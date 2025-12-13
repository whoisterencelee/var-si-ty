import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import glob

# CONFIGURATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
MODELS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")
IMG_SIZE = (224, 224) # Standard ResNet size
BATCH_SIZE = 32       # Can be higher since it's simple classification
LR = 1e-4
EPOCHS = 10
THRESHOLD = 50        # Images with > 50 people are "Dense"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODELS_DIR, exist_ok=True)

class GateDataset(Dataset):
    def __init__(self, split):
        self.img_dir = os.path.join(DATA_ROOT, split, "images")
        self.gt_dir = os.path.join(DATA_ROOT, split, "gt")
        self.img_files = glob.glob(os.path.join(self.img_dir, "*.jpg"))
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        txt_name = os.path.basename(img_path).replace(".jpg", ".txt")
        txt_path = os.path.join(self.gt_dir, txt_name)
        
        # Get Count
        count = 0
        if os.path.exists(txt_path):
            try:
                gt_data = np.loadtxt(txt_path)
                count = gt_data.shape[0] if gt_data.ndim > 0 else 0
            except: count = 0
            
        # Label: 1 if Dense, 0 if Sparse
        label = 1.0 if count >= THRESHOLD else 0.0
        
        image = cv2.imread(img_path)
        if image is None: image = np.zeros((224,224,3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return self.transform(Image.fromarray(image)), torch.tensor([label], dtype=torch.float32)

from PIL import Image

def train_gate():
    print(f"Training Gating Model (Threshold={THRESHOLD})...")
    
    # Data
    train_ds = GateDataset("train")
    val_ds = GateDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model: ResNet18 (Binary Classification)
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid() # Output 0-1 probability
    )
    model = model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0; total = 0
        
        for img, label in train_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            out = model(img).squeeze()
            loss = criterion(out, label.squeeze())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (out > 0.5).float()
            correct += (preds == label.squeeze()).sum().item()
            total += label.size(0)
            
        # Validation
        model.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                out = model(img).squeeze()
                preds = (out > 0.5).float()
                val_correct += (preds == label.squeeze()).sum().item()
                val_total += label.size(0)
        
        val_acc = val_correct / val_total
        print(f"Ep {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {correct/total:.2f} | Val Acc: {val_acc:.2f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "gate_resnet18.pth"))
            print("Saved Best Gate Model.")

if __name__ == "__main__":
    train_gate()
