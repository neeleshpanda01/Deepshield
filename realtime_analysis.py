import sys
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import numpy as np
from collections import deque
import ctypes

# Configuration
MODEL_PATH = Path("models/deepshield_video.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.85
SMOOTHING_WINDOW = 5

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def show_alert(title, message):
    """Show Windows message box alert"""
    ctypes.windll.user32.MessageBoxW(0, message, title, 1)

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
print(f"Using device: {DEVICE}\n")

# Determine input source
if len(sys.argv) > 1:
    source = sys.argv[1]
    if not Path(source).exists():
        print(f"ERROR: File not found: {source}")
        exit(1)
    print(f"Video file: {source}\n")
else:
    source = 1  # Default to camera 1 (OBS VirtualCam)
    print(f"Using camera {source}\n")

# Open source
cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))

if not cap.isOpened():
    print(f"ERROR: Could not open source")
    exit(1)

if isinstance(source, int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

print("Ready. Press 'q' or Escape to quit.\n")
print("=== DeepShield Real-time Analysis ===")
print(f"Alert threshold: {CONFIDENCE_THRESHOLD*100:.0f}% fake confidence\n")

frame_count = 0
import time
start_time = time.time()

prediction_history = deque(maxlen=SMOOTHING_WINDOW)
fake_alert_cooldown = 0
warmup_frames = 30

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("ERROR: Failed to read frame")
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
    
    # Smooth predictions
    prediction_history.append((prediction, confidence))
    
    if len(prediction_history) > 0:
        avg_confidence = np.mean([c for _, c in prediction_history])
        avg_prediction = int(np.round(np.mean([p for p, _ in prediction_history])))
    else:
        avg_prediction = prediction
        avg_confidence = confidence
    
    # Label
    label = "FAKE" if avg_prediction == 0 else "REAL"
    color = (0, 0, 255) if avg_prediction == 0 else (0, 255, 0)
    
    # Alert
    fake_is_detected = avg_prediction == 0 and avg_confidence >= CONFIDENCE_THRESHOLD
    if fake_is_detected and fake_alert_cooldown <= 0 and frame_count > warmup_frames:
        print(f"\n⚠️  DEEPFAKE ALERT! Confidence: {avg_confidence*100:.1f}%")
        show_alert("⚠️ DEEPFAKE DETECTED", 
                   f"High-confidence deepfake detected!\n\nConfidence: {avg_confidence*100:.1f}%")
        fake_alert_cooldown = 60
    
    fake_alert_cooldown -= 1
    
    # Overlay text
    text = f"{label}: {avg_confidence*100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = 20, 40
    cv2.rectangle(frame, (x - 5, y - 25), (x + text_size[0] + 5, y + 5), color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    # Alert banner
    if fake_is_detected:
        alert_text = "⚠️ ALERT: DEEPFAKE DETECTED"
        alert_size = cv2.getTextSize(alert_text, font, 1.0, 2)[0]
        x_alert = (frame.shape[1] - alert_size[0]) // 2
        cv2.rectangle(frame, (x_alert - 5, 50), (x_alert + alert_size[0] + 5, 80), (0, 0, 255), -2)
        cv2.putText(frame, alert_text, (x_alert, 75), font, 1.0, (255, 255, 255), 2)
    
    # FPS
    elapsed = time.time() - start_time
    if elapsed > 0:
        fps = frame_count / elapsed
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (20, frame.shape[0] - 20), font, 0.7, (255, 255, 255), 1)
    
    # Show
    cv2.imshow("DeepShield - Real-time Analysis", frame)
    
    # Quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("\nShutting down...")
        break

cap.release()
cv2.destroyAllWindows()
print("Analysis complete.")
