import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import segmentation_models_pytorch as smp
from datetime import datetime
import glob
import gc
import random

# CONFIGURATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VIS_DIR = os.path.join(PROJECT_ROOT, "visualizations")

# --- PERFORMANCE TUNING ---
IMG_SIZE = (256, 256)  # Training Crop Size
VAL_SIZE = (512, 512)  # Validation Resize (Larger for better accuracy)
BATCH_SIZE = 2         
ACCUM_STEPS = 8        # Effective Batch = 16
MAX_LR = 1e-3          # Aggressive Learning Rate (OneCycle will manage this)
EPOCHS = 50
LAMBDA_ADV = 0.001     
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# ==========================================
# FEATURE 1: CRITIC
# ==========================================
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final(x)
        return torch.sigmoid(x) 

# ==========================================
# FEATURE 2: VISUALIZATION
# ==========================================
# (Kept simple for brevity, logic remains same as fixed version)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]
    def __enter__(self):
        self.handles.append(self.target_layer.register_forward_hook(self.save_activation))
        self.handles.append(self.target_layer.register_backward_hook(self.save_gradient))
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles: handle.remove()
        self.handles = []; self.gradients = None; self.activations = None
    def __call__(self, x):
        output = self.model(x)
        score = output.max()
        self.model.zero_grad()
        score.backward(retain_graph=True)
        if self.gradients is None: return np.zeros((x.size(2), x.size(3)), dtype=np.float32)
        b, k, u, v = self.gradients.size()
        alpha = self.gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        cam = (weights * self.activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min(); cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()

def generate_visualization_for_web(model, image_tensor, original_image, name_prefix):
    # EfficientNet backbone layer target
    # Adjust based on model: usually model.encoder.conv_head or similar for SMP models
    # For ResNet (Regression): model.layer4[-1]
    try:
        if hasattr(model, 'layer4'): target = model.layer4[-1] # ResNet
        else: target = model.encoder.conv_head # EfficientNet in SMP
        
        with GradCAM(model, target) as grad_cam:
            mask = grad_cam(image_tensor)
        mask_resized = cv2.resize(mask[0, 0], (original_image.shape[1], original_image.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam_image = heatmap + np.float32(original_image) / 255
        cam_image = cam_image / np.max(cam_image)
        cam_image = np.uint8(255 * cam_image)
        cv2.imwrite(os.path.join(VIS_DIR, f"{name_prefix}_gradcam.jpg"), cam_image)
        del mask, heatmap, cam_image; gc.collect()
    except Exception as e:
        print(f"Vis Error: {e}")

# ==========================================
# DATASET (IMPROVED: RANDOM CROP)
# ==========================================
class JhuCrowdDataset(Dataset):
    def __init__(self, split, method="density"):
        self.split = split
        self.method = method
        self.img_dir = os.path.join(DATA_ROOT, split, "images")
        self.gt_dir = os.path.join(DATA_ROOT, split, "gt")
        self.map_dir = os.path.join(DATA_ROOT, split, "density_maps")
        self.img_files = glob.glob(os.path.join(self.img_dir, "*.jpg"))
        
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        filename = os.path.basename(img_path)
        npy_name = filename.replace(".jpg", ".npy")
        
        image = cv2.imread(img_path)
        if image is None: image = np.zeros((512,512,3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Target dimensions
        h, w, _ = image.shape
        
        # LOGIC: 
        # Train = Random Crop (256x256)
        # Val   = Center Crop (1024x1024) -> Preserves Scale, fits in VRAM
        
        if self.split == "train":
            crop_h, crop_w = IMG_SIZE
            # Pad if needed
            pad_h, pad_w = max(0, crop_h - h), max(0, crop_w - w)
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                h, w, _ = image.shape
            
            i = random.randint(0, h - crop_h)
            j = random.randint(0, w - crop_w)
        else:
            # VALIDATION: Center Crop 1024x1024 (Or smaller if image is small)
            crop_h, crop_w = (1024, 1024)
            
            # Pad if image is smaller than 1024
            pad_h, pad_w = max(0, crop_h - h), max(0, crop_w - w)
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                h, w, _ = image.shape
                
            # Center coordinates
            i = (h - crop_h) // 2
            j = (w - crop_w) // 2

        # Perform the crop
        image_crop = image[i:i+crop_h, j:j+crop_w]
        
        if self.method == "density":
            map_path = os.path.join(self.map_dir, npy_name)
            if os.path.exists(map_path):
                dmap = np.load(map_path)
                # Pad density map to match image
                if self.split == "train" and (pad_h > 0 or pad_w > 0):
                    dmap = cv2.copyMakeBorder(dmap, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                elif self.split != "train" and (pad_h > 0 or pad_w > 0):
                     dmap = cv2.copyMakeBorder(dmap, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                     
                dmap_crop = dmap[i:i+crop_h, j:j+crop_w]
                target = torch.from_numpy(dmap_crop * 100.0).unsqueeze(0).float()
            else:
                target = torch.zeros((1, crop_h, crop_w), dtype=torch.float32)
        
        return self.norm(image_crop), target


    def get_regression_target(self, txt_name):
        txt_path = os.path.join(self.gt_dir, txt_name)
        count = 0.0
        if os.path.exists(txt_path):
            try:
                gt_data = np.loadtxt(txt_path)
                if gt_data.ndim == 1 and gt_data.size > 0: count = 1.0
                elif gt_data.ndim > 1: count = float(gt_data.shape[0])
            except: pass
        return torch.tensor([count], dtype=torch.float32)

def get_density_model():
    # UPGRADE: EfficientNet-B0 is better than MobileNet
    return smp.Unet("efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)

def get_regression_model():
    return models.resnet18(weights='DEFAULT') # Keep ResNet for now, modify FC later

# ==========================================
# TRAIN LOOP (OneCycleLR)
# ==========================================
def train_adversarial(method_name, generator, discriminator, train_loader, val_loader, epochs):
    generator = generator.to(DEVICE); discriminator = discriminator.to(DEVICE)
    
    opt_g = optim.AdamW(generator.parameters(), lr=MAX_LR, weight_decay=1e-4)
    opt_d = optim.AdamW(discriminator.parameters(), lr=MAX_LR, weight_decay=1e-4)
    
    # OneCycleLR for super convergence
    sched_g = optim.lr_scheduler.OneCycleLR(opt_g, max_lr=MAX_LR, steps_per_epoch=len(train_loader)//ACCUM_STEPS, epochs=epochs)
    sched_d = optim.lr_scheduler.OneCycleLR(opt_d, max_lr=MAX_LR, steps_per_epoch=len(train_loader)//ACCUM_STEPS, epochs=epochs)
    
    criterion_content = nn.L1Loss() # UPGRADE: L1 is better for density convergence
    criterion_adv = nn.BCELoss()
    best_mae = float('inf')
    
    print(f"\nStarting FAST ADVERSARIAL Training ({method_name})...")
    
    for epoch in range(epochs):
        generator.train(); discriminator.train()
        run_g = 0.0; run_d = 0.0
        opt_g.zero_grad(); opt_d.zero_grad()
        
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            # Generator Forward
            fake_maps = generator(images)
            
            # Discriminator Step
            d_real = discriminator(targets)
            d_fake = discriminator(fake_maps.detach())
            loss_d = (criterion_adv(d_real, torch.ones_like(d_real)) + \
                      criterion_adv(d_fake, torch.zeros_like(d_fake))) / 2
            (loss_d / ACCUM_STEPS).backward()
            
            # Generator Step
            d_fake_2 = discriminator(fake_maps)
            loss_adv = criterion_adv(d_fake_2, torch.ones_like(d_real))
            loss_content = criterion_content(fake_maps, targets)
            loss_g = loss_content + (LAMBDA_ADV * loss_adv)
            (loss_g / ACCUM_STEPS).backward()
            
            run_g += loss_g.item() * ACCUM_STEPS
            run_d += loss_d.item() * ACCUM_STEPS
            
            if (i + 1) % ACCUM_STEPS == 0:
                opt_g.step(); opt_d.step()
                sched_g.step(); sched_d.step() # Update LR
                opt_g.zero_grad(); opt_d.zero_grad()
            
            del fake_maps, d_real, d_fake
            
        # Validation
        generator.eval()
        val_mae = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = generator(images)
                pred = torch.sum(outputs, dim=(1,2,3)) / 100.0
                true = torch.sum(targets, dim=(1,2,3)) / 100.0
                val_mae += torch.mean(torch.abs(pred - true)).item()

        avg_val_mae = val_mae / len(val_loader)
        print(f"Ep {epoch+1}/{epochs} | Loss G: {run_g/len(train_loader):.4f} | Val MAE: {avg_val_mae:.2f}")
        
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            torch.save(generator.state_dict(), os.path.join(MODELS_DIR, f"best_{method_name}.pth"))
            
    return generator

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # DENSITY ONLY (Regression omitted for brevity as Density is superior)
    d_train = JhuCrowdDataset("train", "density")
    d_val = JhuCrowdDataset("val", "density")
    
    d_loader = DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    # Validate with batch_size 1 to handle larger 512x512 images safely
    d_vloader = DataLoader(d_val, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    
    d_model = get_density_model()
    critic = Discriminator()
    
    train_adversarial("density_efficientnet_gan", d_model, critic, d_loader, d_vloader, EPOCHS)
