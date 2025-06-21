# app.py - Flask Application for Brain Tumor Classification
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os
import json
import numpy as np
import sys

# Add debug prints at startup
print("🚀 STARTING BRAIN TUMOR CLASSIFIER", flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"Current working directory: {os.getcwd()}", flush=True)

app = Flask(__name__)

# Model Classes (same as your training code)
class DenseNet121Model(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.2, pretrained=True):
        super(DenseNet121Model, self).__init__()
        if pretrained:
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            self.densenet = models.densenet121(weights=None)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.densenet(x)

class EfficientNetB3Model(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.2, pretrained=True):
        super(EfficientNetB3Model, self).__init__()
        if pretrained:
            self.efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        else:
            self.efficientnet = models.efficientnet_b3(weights=None)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)

class ResNet50Model(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.2, pretrained=True):
        super(ResNet50Model, self).__init__()
        if pretrained:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.resnet = models.resnet50(weights=None)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

class VGG16Model(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.2, pretrained=True):
        super(VGG16Model, self).__init__()
        if pretrained:
            self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        else:
            self.vgg = models.vgg16(weights=None)
        self.vgg.classifier[6] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        return self.vgg(x)

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}", flush=True)
models_dict = {}
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_models():
    """Load all trained models with extensive debugging"""
    global models_dict
    
    print("🔄 LOAD_MODELS FUNCTION CALLED", flush=True)
    print(f"Current directory: {os.getcwd()}", flush=True)
    
    try:
        print(f"Files in current directory: {os.listdir('.')}", flush=True)
    except Exception as e:
        print(f"Error listing current directory: {e}", flush=True)
    
    # Check if models folder exists
    models_exists = os.path.exists('models')
    print(f"Models folder exists: {models_exists}", flush=True)
    
    if models_exists:
        try:
            models_files = os.listdir('models')
            print(f"Files in models folder: {models_files}", flush=True)
        except Exception as e:
            print(f"Error reading models folder: {e}", flush=True)
            return
    else:
        print("❌ MODELS FOLDER NOT FOUND!", flush=True)
        return
    
    # Model configurations
    model_configs = {
        'densenet121': {'class': DenseNet121Model, 'file': 'models/densenet121_best_state_dict.pth'},
        'efficientnet_b3': {'class': EfficientNetB3Model, 'file': 'models/efficientnet_b3_best_state_dict.pth'},
        'resnet50': {'class': ResNet50Model, 'file': 'models/resnet50_best_state_dict.pth'},
        'vgg16': {'class': VGG16Model, 'file': 'models/vgg16_best_state_dict.pth'}
    }
    
    for model_name, config in model_configs.items():
        try:
            print(f"🏗️ Attempting to load {model_name}...", flush=True)
            
            # Check if model file exists
            model_file = config['file']
            file_exists = os.path.exists(model_file)
            print(f"Model file {model_file} exists: {file_exists}", flush=True)
            
            if not file_exists:
                print(f"❌ Skipping {model_name} - file not found", flush=True)
                continue
            
            # Load model info to get hyperparameters
            info_file = config['file'].replace('_state_dict.pth', '_info.pth')
            if os.path.exists(info_file):
                print(f"📥 Loading info file: {info_file}", flush=True)
                model_info = torch.load(info_file, map_location=device)
                dropout_rate = model_info['hyperparameters'].get('dropout', 0.2)
                print(f"✅ Loaded dropout rate: {dropout_rate}", flush=True)
            else:
                dropout_rate = 0.2
                print(f"⚠️ Info file not found, using default dropout: {dropout_rate}", flush=True)
            
            # Initialize model WITHOUT pretrained weights
            print(f"🔧 Creating {model_name} architecture...", flush=True)
            model = config['class'](num_classes=4, dropout_rate=dropout_rate, pretrained=False)
            print(f"✅ {model_name} architecture created", flush=True)
            
            # Load state dict
            print(f"📥 Loading state dict for {model_name}...", flush=True)
            state_dict = torch.load(model_file, map_location=device)
            print(f"✅ State dict loaded for {model_name}", flush=True)
            
            print(f"🔗 Loading state dict into model...", flush=True)
            model.load_state_dict(state_dict)
            print(f"✅ State dict loaded into {model_name}", flush=True)
            
            print(f"📱 Moving {model_name} to device: {device}...", flush=True)
            model.to(device)
            model.eval()
            models_dict[model_name] = model
            print(f"✅ {model_name} loaded successfully!", flush=True)
                
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    print(f"🎯 FINAL RESULT: Loaded {len(models_dict)} models: {list(models_dict.keys())}", flush=True)

def predict_image(image, model_name):
    """Make prediction on image using specified model"""
    print(f"🔍 predict_image called with model: {model_name}", flush=True)
    print(f"Available models: {list(models_dict.keys())}", flush=True)
    
    if model_name not in models_dict:
        print(f"❌ Model {model_name} not found in loaded models", flush=True)
        return None, None, None
    
    model = models_dict[model_name]
    
    try:
        print("🖼️ Processing image...", flush=True)
        # Preprocess image
        if isinstance(image, str):
            # If base64 string
            image_data = base64.b64decode(image.split(',')[1])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        print("🔄 Transforming image...", flush=True)
        # Transform image
        input_tensor = transform(image).unsqueeze(0).to(device)
        print(f"Input tensor shape: {input_tensor.shape}", flush=True)
        
        print("🤖 Making prediction...", flush=True)
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)
        
        predicted_label = class_names[predicted_class.item()]
        confidence_score = confidence.item() * 100
        
        print(f"✅ Prediction: {predicted_label}, Confidence: {confidence_score:.2f}%", flush=True)
        
        # Get all class probabilities
        all_probs = {class_names[i]: float(probabilities[i]) * 100 for i in range(len(class_names))}
        
        return predicted_label, confidence_score, all_probs
    
    except Exception as e:
        print(f"❌ Error in predict_image: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None, None, None

# Routes
@app.route('/')
def index():
    print("📄 Index route accessed", flush=True)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("🔮 Predict endpoint called", flush=True)
    try:
        data = request.get_json()
        image_data = data.get('image')
        model_name = data.get('model', 'densenet121')
        
        print(f"Model requested: {model_name}", flush=True)
        print(f"Image data received: {bool(image_data)}", flush=True)
        
        if not image_data:
            print("❌ No image provided", flush=True)
            return jsonify({'error': 'No image provided'}), 400
        
        # Make prediction
        predicted_label, confidence, all_probs = predict_image(image_data, model_name)
        
        # Check if any value is None
        if predicted_label is None or confidence is None or all_probs is None:
            print("❌ Prediction failed - model not available", flush=True)
            return jsonify({
                'error': 'Model not available or prediction failed',
                'available_models': list(models_dict.keys()),
                'requested_model': model_name
            }), 500
        
        result = {
            'prediction': predicted_label,
            'confidence': round(confidence, 2),
            'all_probabilities': {k: round(v, 2) for k, v in all_probs.items()},
            'model_used': model_name
        }
        
        print(f"✅ Returning result: {result}", flush=True)
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in predict endpoint: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/samples')
def get_samples():
    """Get sample images for testing"""
    print("📂 Samples endpoint called", flush=True)
    samples_dir = 'static/samples'
    samples = []
    
    if os.path.exists(samples_dir):
        for filename in os.listdir(samples_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append(filename)
        print(f"Found {len(samples)} sample images", flush=True)
    else:
        print("⚠️ Samples directory not found", flush=True)
    
    return jsonify({'samples': samples})

@app.route('/static/samples/<filename>')
def sample_image(filename):
    print(f"🖼️ Sample image requested: {filename}", flush=True)
    return send_from_directory('static/samples', filename)

@app.route('/debug-info')
def debug_info():
    """Debug endpoint to check system status"""
    print("🔧 Debug info endpoint called", flush=True)
    debug_data = {
        'models_loaded': list(models_dict.keys()),
        'models_count': len(models_dict),
        'current_dir': os.getcwd(),
        'models_exists': os.path.exists('models'),
        'device': str(device),
        'models_files': os.listdir('models') if os.path.exists('models') else [],
        'static_exists': os.path.exists('static'),
        'samples_exists': os.path.exists('static/samples'),
        'python_version': sys.version,
        'torch_version': torch.__version__
    }
    
    print(f"Debug data: {debug_data}", flush=True)
    return jsonify(debug_data)

if __name__ == '__main__':
    print("🎯 MAIN FUNCTION STARTING", flush=True)
    
    try:
        print("📂 Creating static directories...", flush=True)
        os.makedirs('static/samples', exist_ok=True)
        print("✅ Static directories created", flush=True)
        
        print("🤖 Loading models...", flush=True)
        load_models()
        print(f"✅ Model loading complete. Loaded models: {list(models_dict.keys())}", flush=True)
        
        print("🌐 Starting Flask app...", flush=True)
        app.run(debug=True, host='0.0.0.0', port=5002)
        
    except Exception as e:
        print(f"❌ CRITICAL STARTUP ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
else:
    # This runs when deployed (not in debug mode)
    print("🚀 App imported for deployment", flush=True)
    print("📂 Creating static directories...", flush=True)
    os.makedirs('static/samples', exist_ok=True)
    print("✅ Static directories created", flush=True)
    
    print("🤖 Loading models...", flush=True)
    load_models()
    print(f"✅ Model loading complete. Loaded models: {list(models_dict.keys())}", flush=True)