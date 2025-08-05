import os
import torch
import numpy as np
import pickle
import math
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler

# Define the model architecture (same as in training)
class GLULayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLULayer, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim * 2)
        self.bn = torch.nn.BatchNorm1d(output_dim)
    
    def forward(self, x):
        x_proj = self.fc(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        x_glu = x1 * torch.sigmoid(x2)
        x_glu = self.bn(x_glu)
        return x_glu

class FeatureTransformer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, n_glu_layers, shared_layers=None):
        super().__init__()
        self.shared = shared_layers
        self.blocks = torch.nn.ModuleList()
        if shared_layers is None:
            for i in range(n_glu_layers):
                in_dim = input_dim if i == 0 else output_dim
                self.blocks.append(GLULayer(in_dim, output_dim))
        else:
            for i in range(n_glu_layers - len(shared_layers)):
                self.blocks.append(GLULayer(output_dim, output_dim))
    
    def forward(self, x):
        if self.shared is not None:
            for layer in self.shared:
                residual = x
                x = layer(x)
                if residual.shape == x.shape:
                    x = (x + residual) * math.sqrt(0.5)
        for layer in self.blocks:
            residual = x
            x = layer(x)
            if residual.shape == x.shape:
                x = (x + residual) * math.sqrt(0.5)
        return x

class Sparsemax(torch.nn.Module):
    def forward(self, input):
        input = input - input.max(dim=1, keepdim=True)[0]
        z_sorted, _ = torch.sort(input, dim=1, descending=True)
        k = torch.arange(1, input.size(1)+1, device=input.device).float()
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        k_mask = 1 + k * z_sorted > z_cumsum
        k_max = k_mask.sum(dim=1, keepdim=True)
        tau_sum = torch.gather(z_cumsum, 1, k_max.long() - 1)
        tau = (tau_sum - 1) / k_max
        output = torch.clamp(input - tau, min=0)
        return output

class AttentiveTransformer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentiveTransformer, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.bn = torch.nn.BatchNorm1d(output_dim)
        self.sparsemax = Sparsemax()
    
    def forward(self, x, prior):
        x = self.fc(x)
        x = self.bn(x)
        x = x * prior
        x = self.sparsemax(x)
        return x

class DecisionStep(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, shared_layers=None, n_glu=4):
        super().__init__()
        self.feature_transformer = FeatureTransformer(
            input_dim=input_dim,
            output_dim=feature_dim,
            n_glu_layers=n_glu,
            shared_layers=shared_layers
        )
        self.attentive_transformer = AttentiveTransformer(feature_dim, input_dim)
    
    def forward(self, x, prior):
        transformed = self.feature_transformer(x)
        transformed = torch.relu(transformed)
        mask = self.attentive_transformer(transformed, prior)
        masked_x = x * mask
        return transformed, mask, masked_x

class CustomTabNetEncoder(torch.nn.Module):
    def __init__(self, input_dim, n_d=64, n_a=64, feature_dim=64, output_dim=64,
                 n_steps=5, gamma=1.3, n_glu=2, n_shared=2, n_independent=2):
        super().__init__()
        self.n_steps = n_steps
        self.gamma = gamma
        self.shared_layers = torch.nn.ModuleList()
        for i in range(n_shared):
            in_dim = input_dim if i == 0 else feature_dim
            self.shared_layers.append(GLULayer(in_dim, feature_dim))
        self.initial_transform = FeatureTransformer(input_dim, feature_dim, n_glu_layers=n_shared + n_independent)
        self.decision_steps = torch.nn.ModuleList([
            DecisionStep(input_dim, feature_dim, shared_layers=self.shared_layers, n_glu=n_shared + n_independent)
            for _ in range(n_steps)
        ])
        self.output_dim = output_dim
        self.bn = torch.nn.BatchNorm1d(input_dim)
        self.fc = torch.nn.Linear(feature_dim, output_dim)
    
    def forward(self, x):
        x = self.bn(x)
        prior = torch.ones_like(x)
        x = self.initial_transform(x)
        steps_out = []
        for step in self.decision_steps:
            transformed, mask, masked_x = step(x, prior)
            prior = prior * (self.gamma - mask)
            out = self.fc(transformed)
            steps_out.append(out)
        aggregated = torch.stack(steps_out, dim=0).sum(dim=0)
        return aggregated

class CustomTabNetClassifier(torch.nn.Module):
    def __init__(self, input_dim, feature_dim=64, output_dim=64, n_steps=5, gamma=1.3, n_glu=2,  
                 n_shared=2, n_independent=2, num_classes=7):
        super(CustomTabNetClassifier, self).__init__()
        self.encoder = CustomTabNetEncoder(
            input_dim=input_dim,
            feature_dim=feature_dim,
            output_dim=output_dim,
            n_steps=n_steps,
            gamma=gamma,
            n_glu=n_glu
        )
        self.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(output_dim),
            torch.nn.Linear(output_dim, num_classes)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        return self.head(encoded)

# Initialize Flask app
app = Flask(__name__)

# Load headache model and scaler
headache_model_path = os.path.join('models', 'headache_model.pth')
headache_scaler_path = os.path.join('models', 'headache_scaler.pkl')

try:
    headache_checkpoint = torch.load(headache_model_path, map_location=torch.device('cpu'), weights_only=True)
    headache_model_config = headache_checkpoint['model_config']
    headache_model = CustomTabNetClassifier(
        input_dim=headache_model_config['input_dim'],
        feature_dim=headache_model_config['feature_dim'],
        output_dim=headache_model_config['output_dim'],
        n_steps=headache_model_config['n_steps'],
        gamma=headache_model_config['gamma'],
        n_glu=headache_model_config['n_glu'],
        n_shared=headache_model_config['n_shared'],
        n_independent=headache_model_config['n_independent'],
        num_classes=headache_model_config['num_classes']
    )
    headache_model.load_state_dict(headache_checkpoint['model_state_dict'])
    headache_model.eval()
except Exception as e:
    print(f"Error loading headache model: {e}")
    headache_model = None

# Load headache scaler with error handling
try:
    with open(headache_scaler_path, 'rb') as f:
        headache_scaler = pickle.load(f)
except Exception as e:
    print(f"Error loading headache scaler: {e}")
    # Create a new scaler as fallback
    headache_scaler = StandardScaler()

# Load migraine model and scaler
migraine_model_path = os.path.join('models', 'migraine_model.pth')
migraine_scaler_path = os.path.join('models', 'migraine_scaler.pkl')

try:
    migraine_checkpoint = torch.load(migraine_model_path, map_location=torch.device('cpu'), weights_only=True)
    migraine_model_config = migraine_checkpoint['model_config']
    migraine_model = CustomTabNetClassifier(
        input_dim=migraine_model_config['input_dim'],
        feature_dim=migraine_model_config['feature_dim'],
        output_dim=migraine_model_config['output_dim'],
        n_steps=migraine_model_config['n_steps'],
        gamma=migraine_model_config['gamma'],
        n_glu=migraine_model_config['n_glu'],
        n_shared=migraine_model_config['n_shared'],
        n_independent=migraine_model_config['n_independent'],
        num_classes=migraine_model_config['num_classes']
    )
    migraine_model.load_state_dict(migraine_checkpoint['model_state_dict'])
    migraine_model.eval()
except Exception as e:
    print(f"Error loading migraine model: {e}")
    migraine_model = None

# Load migraine scaler with error handling
try:
    with open(migraine_scaler_path, 'rb') as f:
        migraine_scaler = pickle.load(f)
except Exception as e:
    print(f"Error loading migraine scaler: {e}")
    # Create a new scaler as fallback
    migraine_scaler = StandardScaler()

# Define the feature order for headache classification
headache_feature_order = [
    'characterisation_pulsating', 'characterisation_pressing', 'photophobia', 'aggravation',
    'location_bilateral', 'nausea', 'location_unilateral', 'severity', 'phonophobia',
    'characterisation_stabbing', 'lacrimation', 'durationGroup', 'rhinorrhoea',
    'conjunctival_injection', 'pericranial', 'sweating', 'location_orbital', 'eyelid_oedema',
    'miosis', 'nasal_congestion', 'agitation', 'ptosis', 'vomitting', 'hemiplegic',
    'headache_with_aura', 'aura_duration', 'visual_symptomps', 'aura_development',
    'sensory_symptomps', 'homonymous_symptomps', 'diplopia', 'vertigo'
]

# Define the feature order for migraine classification
migraine_feature_order = [
    'Age', 'Frequency', 'Location', 'Character', 'Intensity', 
    'Vomit', 'Phonophobia', 'Photophobia', 'Visual', 'Sensory', 
    'Dysphasia', 'Vertigo', 'Tinnitus', 'Hypoacusis', 'Visual_defect', 
    'Conscience', 'DPF'
]

# Define the mapping for headache categorical features
headache_categorical_mappings = {
    'characterisation_pulsating': {'No': 0, 'Yes': 1},
    'characterisation_pressing': {'No': 0, 'Yes': 1},
    'photophobia': {'No': 0, 'Yes': 1},
    'aggravation': {'No': 0, 'Yes': 1},
    'location_bilateral': {'No': 0, 'Yes': 1},
    'nausea': {'No': 0, 'Yes': 1},
    'location_unilateral': {'No': 0, 'Yes': 1},
    'severity': {'Mild': 0, 'Moderate': 1, 'Severe': 2},
    'phonophobia': {'No': 0, 'Yes': 1},
    'characterisation_stabbing': {'No': 0, 'Yes': 1},
    'lacrimation': {'No': 0, 'Yes': 1},
    'durationGroup': {
        'A': 0,  # Less than 1 minute
        'B': 1,  # 3-5 minutes
        'C': 2,  # 2-4 minutes
        'D': 3,  # 4-15 minutes
        'E': 4,  # 15-30 minutes
        'F': 5,  # 30 minutes - 3 hours
        'G': 6,  # 3-4 hours
        'H': 7,  # 4 hours - 3 days
        'I': 8,  # 3-7 days
        'J': 9   # More than 1 week
    },
    'rhinorrhoea': {'No': 0, 'Yes': 1},
    'conjunctival_injection': {'No': 0, 'Yes': 1},
    'pericranial': {'No': 0, 'Yes': 1},
    'sweating': {'No': 0, 'Yes': 1},
    'location_orbital': {'No': 0, 'Yes': 1},
    'eyelid_oedema': {'No': 0, 'Yes': 1},
    'miosis': {'No': 0, 'Yes': 1},
    'nasal_congestion': {'No': 0, 'Yes': 1},
    'agitation': {'No': 0, 'Yes': 1},
    'ptosis': {'No': 0, 'Yes': 1},
    'vomitting': {'No': 0, 'Yes': 1},
    'hemiplegic': {'No': 0, 'Yes': 1},
    'headache_with_aura': {'No': 0, 'Yes': 1},
    'aura_duration': {'None': 0, 'Hour': 1, 'Day': 2},
    'visual_symptomps': {'No': 0, 'Yes': 1},
    'aura_development': {'No': 0, 'Yes': 1},
    'sensory_symptomps': {'No': 0, 'Yes': 1},
    'homonymous_symptomps': {'No': 0, 'Yes': 1},
    'diplopia': {'No': 0, 'Yes': 1},
    'vertigo': {'No': 0, 'Yes': 1}
}

# Define the mapping for migraine categorical features
migraine_categorical_mappings = {
    'Location': {'None': 0, 'Unilateral': 1, 'Bilateral': 2},
    'Character': {'None': 0, 'Throbbing': 1, 'Constant': 2},
    'Intensity': {'None': 0, 'Mild': 1, 'Medium': 2, 'Severe': 3},
    'Vomit': {'Not': 0, 'Yes': 1},
    'Phonophobia': {'Not': 0, 'Yes': 1},
    'Photophobia': {'Not': 0, 'Yes': 1},
    'Dysphasia': {'Not': 0, 'Yes': 1},
    'Vertigo': {'Not': 0, 'Yes': 1},
    'Tinnitus': {'Not': 0, 'Yes': 1},
    'Hypoacusis': {'Not': 0, 'Yes': 1},
    'Visual_defect': {'Not': 0, 'Yes': 1},
    'Conscience': {'Not': 0, 'Yes': 1},
    'DPF': {'Not': 0, 'Yes': 1}
}

# Define the mapping for the target classes
headache_class_mapping = {
    1: "Migraine",
    2: "Tension",
    0: "Cluster"
}

migraine_class_mapping = {
    0: "Basilar-type aura",
    1: "Familial hemiplegic migraine",
    2: "Migraine without aura",
    3: "Other",
    4: "Sporadic hemiplegic migraine",
    5: "Typical aura with migraine",
    6: "Typical aura without migraine"
}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/headache')
def headache():
    return render_template('headache.html', mappings=headache_categorical_mappings)

@app.route('/migraine')
def migraine():
    return render_template('migraine.html', mappings=migraine_categorical_mappings)

@app.route('/predict_headache', methods=['POST'])
def predict_headache():
    if headache_model is None:
        return jsonify({'error': 'Headache model not loaded properly'})
    
    try:
        # Get the form data
        form_data = request.form
        
        # Convert form data to the model input format
        input_data = []
        for feature in headache_feature_order:
            value = form_data[feature]
            if feature in headache_categorical_mappings:
                # Map categorical value to numeric
                numeric_value = headache_categorical_mappings[feature][value]
            else:
                # Convert numeric string to float
                numeric_value = float(value)
            input_data.append(numeric_value)
        
        # Convert to numpy array and reshape for a single sample
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale the input
        scaled_input = headache_scaler.transform(input_array)
        
        # Convert to tensor
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            outputs = headache_model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
        
        # Map the predicted class to the headache type
        headache_type = headache_class_mapping[predicted_class]
        
        # Return the result as JSON
        return jsonify({'headache_type': headache_type})
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'})

@app.route('/predict_migraine', methods=['POST'])
def predict_migraine():
    if migraine_model is None:
        return jsonify({'error': 'Migraine model not loaded properly'})
    
    try:
        # Get the form data
        form_data = request.form
        
        # Convert form data to the model input format
        input_data = []
        for feature in migraine_feature_order:
            value = form_data[feature]
            if feature in migraine_categorical_mappings:
                # Map categorical value to numeric
                numeric_value = migraine_categorical_mappings[feature][value]
            else:
                # Convert numeric string to float
                numeric_value = float(value)
            input_data.append(numeric_value)
        
        # Convert to numpy array and reshape for a single sample
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale the input
        scaled_input = migraine_scaler.transform(input_array)
        
        # Convert to tensor
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            outputs = migraine_model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
        
        # Map the predicted class to the migraine type
        migraine_type = migraine_class_mapping[predicted_class]
        
        # Return the result as JSON
        return jsonify({'migraine_type': migraine_type})
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
