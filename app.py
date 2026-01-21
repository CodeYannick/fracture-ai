import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import base64
import numpy as np
from model import get_resnet_mnist_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load Model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = get_resnet_mnist_model()
checkpoint_path = "checkpoints/resnet_mnist.pth"

try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model from {checkpoint_path}")
except FileNotFoundError:
    print(f"Error: Model checkpoint not found at {checkpoint_path}. Please train the model first.")
    # Initialize with random weights if file not found (just to allow app to start, though predictions will be garbage)

model.to(device)
model.eval()

# Preprocessing transform
# Same normalization as training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return jsonify({
            'digit': int(predicted.item()),
            'confidence': float(confidence.item())
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run on 0.0.0.0 to make it accessible if needed, or just localhost
    # Disable reloader for background execution stability
    app.run(host='0.0.0.0', port=5001, debug=False)
