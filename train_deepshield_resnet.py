import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from pathlib import Path
import time

# Configuration
FRAMES_DIR = Path("frames")
MODELS_DIR = Path("models")
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001  # Lower for fine-tuning
VAL_SPLIT = 0.2

# Create models directory
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transformations with stronger augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset
print("\nLoading dataset...")
dataset = ImageFolder(str(FRAMES_DIR), transform=train_transform)

if len(dataset) == 0:
    print("ERROR: No images found in frames directory!")
    exit(1)

print(f"Total images: {len(dataset)}")
print(f"Classes: {dataset.classes}")

# Split into train and val
train_size = int(len(dataset) * (1 - VAL_SPLIT))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Update val_dataset transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# Model - ResNet50
print("\nInitializing ResNet50...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Replace classifier head
num_classes = 2
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(512, num_classes)
)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

# Training loop
best_val_acc = 0.0
best_epoch = 0

print("\n=== Training ResNet50 ===\n")

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_acc = 100 * train_correct / train_total
    
    # Validate
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = 100 * val_correct / val_total
    
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train Accuracy: {train_acc:.2f}%")
    print(f"  Val Accuracy: {val_acc:.2f}%")
    
    # Adjust learning rate
    scheduler.step(val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        model_path = MODELS_DIR / "deepshield_video.pt"
        torch.save(model.state_dict(), str(model_path))
        print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    print()

print("=== Training Complete ===")
print(f"Best Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
print(f"Model saved to: {MODELS_DIR / 'deepshield_video.pt'}")
