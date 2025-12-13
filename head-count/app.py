import os
import time
import numpy as np
import cv2
import base64
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from torchvision import models

app = Flask(__name__)

# --- Configuration ---
MODEL_PATHS = {
    "density": "models/best_density_efficientnet_gan.pth",
    "regression": "models/20251125_214552_regression_resnet18.onnx",
    "gate": "models/gate_resnet18.pth",
    "yolos": "models/yolov8s.pt",
    "yolox": "models/yolov8x.pt"
}

device = "cpu" # Spaces use CPU by default

# --- Load Density Model (PyTorch) ---
def get_density_model():
    return smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=1, activation=None)

print("Loading models...")
try:
    density_model = get_density_model()
    state_dict = torch.load(MODEL_PATHS["density"], map_location=torch.device('cpu'))
    density_model.load_state_dict(state_dict)
    density_model.to(device)
    density_model.eval()
    print("Density model loaded.")
except Exception as e:
    print(f"Error loading density: {e}"); density_model = None

# --- Load Gatekeeper Model (PyTorch) ---
try:
    gate_model = models.resnet18()
    # FIX: Must match training architecture EXACTLY (including Dropout)
    gate_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    gate_model.load_state_dict(torch.load(MODEL_PATHS["gate"], map_location=torch.device('cpu')))
    gate_model.to(device)
    gate_model.eval()
    print("Gatekeeper model loaded.")
except Exception as e:
    print(f"Error loading gatekeeper: {e}"); gate_model = None

# --- Load Regression Model (ONNX) ---
try:
    regression_net = cv2.dnn.readNetFromONNX(MODEL_PATHS["regression"])
    print("Regression model loaded.")
except Exception as e:
    print(f"Error loading regression: {e}"); regression_net = None

# --- Load YOLO Models ---
yolo_small = YOLO(MODEL_PATHS["yolos"])
yolo_large = YOLO(MODEL_PATHS["yolox"])
print("All models loaded.")

# --- Transforms ---
norm_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
gate_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
reg_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Inference Functions ---
def run_density_inference(image_pil):
    if density_model is None: return 0, 0, "", 0.0
    start = time.time()
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]
    pad_h, pad_w = (32 - h % 32) % 32, (32 - w % 32) % 32
    if pad_h or pad_w: image_np = cv2.copyMakeBorder(image_np, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    
    img_tensor = norm_transform(Image.fromarray(image_np)).unsqueeze(0).to(device)
    with torch.no_grad(): output = density_model(img_tensor)
    
    raw_map = output[0, 0, :h, :w].cpu().numpy()
    count = np.sum(raw_map) / 100.0
    
    heatmap_norm = cv2.normalize(raw_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    _, buffer = cv2.imencode('.jpg', heatmap_color)
    return int(round(count)), round((time.time()-start)*1000, 1), base64.b64encode(buffer).decode('utf-8')

def run_regression_inference(image_pil):
    if regression_net is None: return 0, 0
    start = time.time()
    img_tensor = reg_transform(image_pil).unsqueeze(0).numpy()
    regression_net.setInput(img_tensor)
    count = regression_net.forward()[0][0]
    return int(max(0, round(count))), round((time.time()-start)*1000, 1)

def run_yolo_inference(model, image_pil):
    start = time.time()
    img_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    results = model(image_pil, classes=[0], conf=0.15, iou=0.3, verbose=False)
    count = len(results[0].boxes)
    
    mask = np.zeros_like(img_cv2, dtype=np.float32)
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        radius = int(max(int(x2-x1), int(y2-y1))/1.5)
        cv2.circle(mask, (cx, cy), radius, (0,0,255), -1)
    mask = cv2.GaussianBlur(mask, (99, 99), 0)
    mask = mask / (mask.max() + 1e-6)
    overlay = np.clip(img_cv2.astype(np.float32) + (mask * 255 * 0.6), 0, 255).astype(np.uint8)
    _, buffer = cv2.imencode('.jpg', overlay)
    return count, round((time.time()-start)*1000, 1), base64.b64encode(buffer).decode('utf-8')

# --- Logic Chain ---
def get_best_estimate(density_count, yolo_count, gate_prob):
    if gate_prob > 0.6:
        return density_count, "density", f"Scene Classified as Dense (Confidence: {gate_prob:.2f})"
    elif gate_prob < 0.4:
        return yolo_count, "yolox", f"Scene Classified as Sparse (Confidence: {1-gate_prob:.2f})"
    else:
        diff_ratio = density_count / (yolo_count + 1e-6)
        if diff_ratio > 2.0:
             return density_count, "density", "Uncertain Scene, but Density Map detected heavy crowding."
        elif diff_ratio < 0.5:
             return yolo_count, "yolox", "Uncertain Scene, but YOLO detected clear objects."
        else:
             avg = int((density_count + yolo_count) / 2)
             return avg, "hybrid", f"Scene Ambiguous (Conf: {gate_prob:.2f}). Using Ensemble Average."

@app.route('/')
def index(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    
    try:
        image = Image.open(file.stream).convert('RGB')
        t_start = time.time()
        
        # 1. Run Gatekeeper First
        gate_prob = 0.5 # Default to uncertain if model fails
        if gate_model is not None:
            gate_input = gate_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                gate_prob = gate_model(gate_input).item()
        else:
            print("Warning: Gatekeeper model is None. Using default prob 0.5")
        
        # 2. Run All Models
        d_c, d_t, d_map = run_density_inference(image)
        r_c, r_t = run_regression_inference(image)
        ys_c, ys_t, ys_map = run_yolo_inference(yolo_small, image)
        yx_c, yx_t, yx_map = run_yolo_inference(yolo_large, image)
        
        # 3. Decision Logic
        best_count, best_model, reason = get_best_estimate(d_c, yx_c, gate_prob)
        
        return jsonify({
            'density': {'count': d_c, 'time': d_t, 'heatmap': d_map},
            'regression': {'count': r_c, 'time': r_t},
            'yolos': {'count': ys_c, 'time': ys_t, 'heatmap': ys_map},
            'yolox': {'count': yx_c, 'time': yx_t, 'heatmap': yx_map},
            'recommendation': {
                'count': best_count,
                'model': best_model,
                'reason': reason
            },
            'gate_prob': gate_prob,
            'total_time': round((time.time()-t_start)*1000, 1)
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=7860, debug=False)
