import sys
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python test_video_path.py <video_path>")
    print("Example: python test_video_path.py C:/path/to/video.mp4")
    exit(1)

video_path = sys.argv[1]

# Configuration
MODEL_PATH = Path("models/deepshield_video.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.85

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
print(f"Model loaded (ResNet50)\n")

if not Path(video_path).exists():
    print(f"ERROR: File not found: {video_path}")
    exit(1)

print(f"Opening video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit(1)

print("Analyzing video...\n")
print("=== DeepShield Video Analysis ===\n")

frame_count = 0
fake_count = 0
real_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_count += 1
    
    # Prepare frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = transforms.ToPILImage()(frame_rgb)
    tensor = transform(pil_frame).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    label = "FAKE" if prediction == 0 else "REAL"
    
    if prediction == 0:
        fake_count += 1
    else:
        real_count += 1
    
    if frame_count % 30 == 0:
        print(f"Frame {frame_count}: {label} ({confidence*100:.1f}%)")

cap.release()

# Summary
print(f"\n=== Analysis Complete ===")
print(f"Total frames: {frame_count}")
print(f"FAKE: {fake_count} ({100*fake_count/frame_count:.1f}%)")
print(f"REAL: {real_count} ({100*real_count/frame_count:.1f}%)")

# Verdict
fake_ratio = fake_count / frame_count
overall = "DEEPFAKE DETECTED" if fake_ratio > 0.5 else "AUTHENTIC VIDEO"
print(f"\n📊 VERDICT: {overall}")

if fake_ratio > 0.5:
    print(f"⚠️  {fake_ratio*100:.1f}% of frames detected as fake")
else:
    print(f"✓ {real_count/frame_count*100:.1f}% of frames detected as real")
