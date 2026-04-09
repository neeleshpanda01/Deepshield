import sys
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import time

# --- CONFIGURATION ---
MODEL_PATH = Path("models/deepshield_video.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.85

# --- SETUP ---
print("\n" + "="*40)
print("   DEEPSHIELD VIDEO ANALYZER")
print("="*40 + "\n")

if len(sys.argv) < 2:
    print("❌ Usage: python analyze.py <path_to_video_file>")
    print("   Example: python analyze.py C:/Videos/suspect.mp4")
    input("\nPress Enter to exit...")
    exit(1)

video_path = sys.argv[1].strip('"') # Remove quotes if present

if not Path(video_path).exists():
    print(f"❌ Error: File not found: {video_path}")
    input("\nPress Enter to exit...")
    exit(1)

if not MODEL_PATH.exists():
    print(f"❌ Error: Model not found at {MODEL_PATH}")
    print("   Make sure 'models/deepshield_video.pt' exists.")
    input("\nPress Enter to exit...")
    exit(1)

# --- LOAD MODEL ---
print("⏳ Loading AI Model (ResNet50)...")
try:
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
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# --- PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- ANALYSIS LOOP ---
print(f"\n🎥 Opening video: {Path(video_path).name}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video file.")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"   Total Frames: {total_frames}")
print("   Analyzing every 30th frame...\n")

frame_count = 0
analyzed_count = 0
fake_accum = 0.0
real_accum = 0.0
fake_frames = 0
real_frames = 0

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Analyze every 30th frame (approx 1 per second for 30fps)
    if frame_count % 30 != 0:
        continue
        
    analyzed_count += 1
    
    # Transform
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = transforms.ToPILImage()(frame_rgb)
    tensor = transform(pil_frame).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        fake_conf = probabilities[0][0].item()
        real_conf = probabilities[0][1].item()
    
    # Log
    label = "FAKE 🔴" if prediction == 0 else "REAL 🟢"
    conf = fake_conf if prediction == 0 else real_conf
    
    if prediction == 0:
        fake_frames += 1
        fake_accum += fake_conf
    else:
        real_frames += 1
        real_accum += real_conf
        
    print(f"   Frame {frame_count}: {label} ({conf*100:.1f}%)")

cap.release()
duration = time.time() - start_time

# --- RESULTS ---
print("\n" + "-"*40)
print("   ANALYSIS COMPLETE")
print("-" * 40)

if analyzed_count == 0:
    print("❌ No frames analyzed (video too short?)")
    exit(0)

fake_ratio = fake_frames / analyzed_count
avg_fake_conf = (fake_accum / fake_frames) if fake_frames > 0 else 0

print(f"   Processed: {analyzed_count} frames in {duration:.1f}s")
print(f"   Fake Frames: {fake_frames}")
print(f"   Real Frames: {real_frames}")

print("\n   🔍 VERDICT:")
if fake_ratio > 0.4: # Specific threshold
    print(f"   ⚠️  DEEPFAKE DETECTED")
    print(f"   Confidence: {avg_fake_conf*100:.1f}%")
else:
    print(f"   ✅  AUTHENTIC VIDEO")
    print(f"   Confidence: {(real_accum/real_frames)*100:.1f}%" if real_frames > 0 else "   N/A")

print("="*40 + "\n")
input("Press Enter to close window...")
