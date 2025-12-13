import os
import cv2
import numpy as np
import scipy.ndimage as ndimage
from tqdm import tqdm
import glob

# ---------------- CONFIGURATION ----------------
# Path to your main dataset folder
DATA_ROOT = "data" 
# Sub-directories to process
SPLITS = ["train", "val", "test"]
# Minimum sigma (blur) to prevent dots from disappearing
MIN_SIGMA = 1  
# If head size is missing (w,h=0), fallback to this fixed sigma
FIXED_SIGMA = 15 
# -----------------------------------------------

def generate_adaptive_density_map(img_shape, gt_data):
    height, width = img_shape
    density_map = np.zeros((height, width), dtype=np.float32)
    
    # Safety check for empty data
    if gt_data is None or len(gt_data) == 0:
        return density_map

    # Ensure gt_data is iterable (handling the 1-person case safely)
    if isinstance(gt_data, np.ndarray) and gt_data.ndim == 1:
        gt_data = [gt_data]

    for row in gt_data:
        # Parse with safety checks
        try:
            x, y, w, h = row[0], row[1], row[2], row[3]
        except IndexError:
            continue # Skip malformed rows

        # Coordinates
        y_int = int(y)
        x_int = int(x)
        
        # SKIP points outside image (don't clamp, just ignore them)
        # Clamping piles up density at the edges, confusing the model.
        if x_int < 0 or x_int >= width or y_int < 0 or y_int >= height:
            continue

        # --- ADAPTIVE SIGMA LOGIC ---
        # If head size is available and valid
        if w > 0 and h > 0:
            sigma = min(max(1, 0.3 * ((w + h) / 2)), 25) # Cap max sigma to prevent giant blobs
        else:
            # Fallback: Use K-Nearest Neighbors (KNN) if size is missing
            # But for simplicity now, use a SMALLER fixed sigma
            sigma = 4  # Much tighter than 15!
            
        # Optimization: Instead of blurring the WHOLE image for every point (slow),
        # we generate a small Gaussian patch and add it to the map.
        
        # Determine patch size (3-sigma rule covers ~99% of the distribution)
        k_size = int(3 * sigma) * 2 + 1
        
        # Create Gaussian patch
        # We create a delta function at the center and blur it
        patch_size = (k_size, k_size)
        patch_center_x, patch_center_y = k_size // 2, k_size // 2
        
        # Define the Gaussian Kernel manually or via scipy
        # Here we use a simple 2D gaussian function on a grid
        y_grid, x_grid = np.ogrid[-patch_center_y:k_size-patch_center_y, 
                                  -patch_center_x:k_size-patch_center_x]
        gaussian_patch = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        
        # Normalize? Standard practice is sum of map = count.
        # So the patch should sum to 1.
        if np.sum(gaussian_patch) > 0:
            gaussian_patch = gaussian_patch / np.sum(gaussian_patch)
            
        # --- PASTE PATCH ONTO DENSITY MAP ---
        # Calculate ranges on the main density map
        y1 = max(0, y_int - patch_center_y)
        y2 = min(height, y_int + patch_center_y + 1)
        x1 = max(0, x_int - patch_center_x)
        x2 = min(width, x_int + patch_center_x + 1)
        
        # Calculate ranges on the patch (handle edge cases where patch is cropped)
        py1 = max(0, patch_center_y - (y_int - y1))
        py2 = py1 + (y2 - y1)
        px1 = max(0, patch_center_x - (x_int - x1))
        px2 = px1 + (x2 - x1)
        
        # Add patch to density map
        density_map[y1:y2, x1:x2] += gaussian_patch[py1:py2, px1:px2]

    return density_map

def main():
    for split in SPLITS:
        img_dir = os.path.join(DATA_ROOT, split, "images")
        gt_dir = os.path.join(DATA_ROOT, split, "gt")
        
        # We will save density maps in a new 'density_maps' folder inside the split
        save_dir = os.path.join(DATA_ROOT, split, "density_maps")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Processing split: {split}...")
        
        # Get all text files
        txt_files = glob.glob(os.path.join(gt_dir, "*.txt"))
        
        for txt_path in tqdm(txt_files):
            filename = os.path.basename(txt_path)
            # Image filename corresponds to gt filename usually by replacing extension
            # Check your files: usually gt is "0001.txt" and img is "0001.jpg"
            img_name = filename.replace(".txt", ".jpg") 
            img_path = os.path.join(img_dir, img_name)
            
            # Skip if image doesn't exist
            if not os.path.exists(img_path):
                continue
                
            # Read Image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading image: {img_path}")
                continue
            img_h, img_w = img.shape[:2]
            
            # Read GT file
            try:
                # Load x, y, w, h (first 4 cols)
                # JHU format: x y w h o b
                gt_data = np.loadtxt(txt_path)
                
                # Handle case where there is only 1 line (1D array) or 0 lines
                if gt_data.ndim == 1 and gt_data.size > 0:
                    gt_data = gt_data[np.newaxis, :]
                elif gt_data.size == 0:
                    gt_data = []
                    
            except Exception as e:
                print(f"Error reading GT {txt_path}: {e}")
                gt_data = []

            # Generate Map
            dmap = generate_adaptive_density_map((img_h, img_w), gt_data)
            
            # Save as .npy (much faster to load than saving as image)
            save_name = filename.replace(".txt", ".npy")
            np.save(os.path.join(save_dir, save_name), dmap)

if __name__ == "__main__":
    main()
