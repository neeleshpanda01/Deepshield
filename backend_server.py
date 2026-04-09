import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = Path("models/deepshield_video.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
print("Loading model...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(512, 2)
)
model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"Model loaded (ResNet50)")
print(f"Using device: {DEVICE}")
print(f"Server ready on http://localhost:5000\n")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "device": str(DEVICE)})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict deepfake from base64 encoded image
    
    Request JSON:
    {
        "image": "base64_encoded_image"
    }
    
    Response JSON:
    {
        "prediction": 0 or 1,  # 0=fake, 1=real
        "fake_confidence": float,
        "real_confidence": float,
        "label": "FAKE" or "REAL"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Missing 'image' in request"}), 400
        
        print(".", end="", flush=True) # Print dot for every request
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Transform
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            fake_conf = probabilities[0][0].item()
            real_conf = probabilities[0][1].item()
        
        label = "FAKE" if prediction == 0 else "REAL"
        
        return jsonify({
            "prediction": int(prediction),
            "fake_confidence": float(fake_conf),
            "real_confidence": float(real_conf),
            "label": label
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_predict():
    """
    Predict multiple frames at once
    
    Request JSON:
    {
        "images": ["base64_image1", "base64_image2", ...]
    }
    
    Response JSON:
    {
        "predictions": [
            {"label": "FAKE", "fake_confidence": 0.95, ...},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'images' not in data:
            return jsonify({"error": "Missing 'images' in request"}), 400
        
        results = []
        
        for image_b64 in data['images']:
            try:
                image_data = base64.b64decode(image_b64)
                image = Image.open(io.BytesIO(image_data))
                tensor = transform(image).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()
                    fake_conf = probabilities[0][0].item()
                    real_conf = probabilities[0][1].item()
                
                label = "FAKE" if prediction == 0 else "REAL"
                
                results.append({
                    "prediction": int(prediction),
                    "fake_confidence": float(fake_conf),
                    "real_confidence": float(real_conf),
                    "label": label
                })
            except Exception as e:
                results.append({"error": str(e)})
        
        return jsonify({"predictions": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=False)
