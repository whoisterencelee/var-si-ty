import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import glob

# CONFIGURATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SPLIT = "val"  # Which folder to test: 'val' or 'test'
CONF_THRESHOLD = 0.25  # Confidence to count a person

def main():
    # 1. Load Pre-trained YOLOv8 (Extra Large)
    print("[INFO] Loading YOLOv8x (Extra Large)...")
    model = YOLO("yolov8x.pt")
    
    # 2. Locate Images and Labels
    img_dir = os.path.join(DATA_ROOT, SPLIT, "images")
    gt_dir = os.path.join(DATA_ROOT, SPLIT, "gt")
    
    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    print(f"[INFO] Found {len(img_files)} images in {SPLIT} set.")

    # 3. Run Inference
    print("[INFO] Running inference (this may take a moment)...")
    # stream=True makes it a generator, saving memory
    results = model.predict(source=img_dir, classes=[0], conf=CONF_THRESHOLD, verbose=False, stream=True)
    
    # 4. Calculate MAE
    total_error = 0.0
    count = 0
    
    # We iterate through the generator
    for res in tqdm(results, total=len(img_files)):
        # A. Get Predicted Count
        pred_count = len(res.boxes)
        
        # B. Get Ground Truth Count
        filename = os.path.basename(res.path)
        txt_name = filename.replace(".jpg", ".txt")
        gt_path = os.path.join(gt_dir, txt_name)
        
        gt_count = 0
        if os.path.exists(gt_path):
            try:
                gt_data = np.loadtxt(gt_path)
                if gt_data.size > 0:
                    # If it's 1D (one person), shape[0] gives number of elements, 
                    # so check dimension or assume rows if 2D.
                    gt_count = gt_data.shape[0] if gt_data.ndim > 1 else 1
            except:
                gt_count = 0
        
        # C. Accumulate Error
        total_error += abs(pred_count - gt_count)
        count += 1

    # 5. Final Result
    mae = total_error / count
    print("\n" + "="*30)
    print(f"FINAL MAE: {mae:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
