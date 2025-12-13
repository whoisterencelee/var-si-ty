#!/usr/bin/env python3
"""Show a random training example side-by-side: the raw image and the density map heatmap.

This script no longer runs as a CLI for arbitrary files. When executed it will:
 - pick a random .npy density map from data/train/density_maps
 - try to find the corresponding image in data/train/images (common extensions: .jpg/.png)
 - display the density heatmap next to the original image

Run from the project root:
    python scripts/view_density.py

Dependencies: numpy, matplotlib, pillow
"""
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_density(path: Path):
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}, got shape={arr.shape}")
    return arr


def show_density(dmap, cmap="viridis", vmin=None, vmax=None, title=None):
    plt.imshow(dmap, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="density")
    if title:
        plt.title(title)
    plt.axis('off')


def overlay_on_image(img_path: Path, dmap, alpha=0.5, cmap="jet", vmin=None, vmax=None, title=None):
    img = Image.open(img_path).convert('RGB')
    img_arr = np.asarray(img)

    # Resize dmap to image size if shapes don't match
    if dmap.shape != img_arr.shape[:2]:
        # Use PIL to resize floating point arrays so we don't require scikit-image
        orig_sum = dmap.sum() if dmap.sum() != 0 else 1.0
        # Image.fromarray supports 'F' mode for floats
        im = Image.fromarray(dmap.astype('float32'), mode='F')
        # PIL resize wants size=(width, height)
        new_size = (img_arr.shape[1], img_arr.shape[0])
        try:
            # RESAMPLING constants changed names across Pillow versions
            resample = Image.Resampling.LANCZOS
        except Exception:
            # fallback for older pillow versions
            resample = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.BICUBIC

        resized = im.resize(new_size, resample)
        dmap_resized = np.asarray(resized, dtype=np.float32)
        if dmap_resized.sum() != 0:
            dmap_resized *= (orig_sum / dmap_resized.sum())
        dmap = dmap_resized

    plt.imshow(img_arr)
    plt.imshow(dmap, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    plt.axis('off')
    if title:
        plt.title(title)


def list_npy_files(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix == '.npy'])


def find_random_train_pair(data_root: Path = Path('data')):
    """Pick a random density map in data/train/density_maps and return paths (density, image)
    If the corresponding image file is missing, image_path will be None.
    """
    dmap_dir = data_root / 'train' / 'density_maps'
    img_dir = data_root / 'train' / 'images'
    files = [p for p in sorted(dmap_dir.iterdir()) if p.suffix == '.npy']
    if not files:
        raise FileNotFoundError(f"No .npy files found in {dmap_dir}")
    dpath = random.choice(files)
    base = dpath.stem
    # look for common image extensions
    for ext in ('.jpg', '.jpeg', '.png'):
        candidate = img_dir / (base + ext)
        if candidate.exists():
            return dpath, candidate
    # not found
    return dpath, None


def show_train_pair(dmap_path: Path, img_path: Path | None):
    dmap = load_density(dmap_path)
    title = f"{dmap_path.name} — sum={dmap.sum():.3f} shape={dmap.shape}"

    if img_path and img_path.exists():
        img = Image.open(img_path).convert('RGB')
        img_arr = np.asarray(img)

        # if shapes mismatch, resize dmap to image size (preserving sum so counts still make sense visually)
        if dmap.shape != img_arr.shape[:2]:
            orig_sum = dmap.sum() if dmap.sum() != 0 else 1.0
            im = Image.fromarray(dmap.astype('float32'), mode='F')
            try:
                resample = Image.Resampling.LANCZOS
            except Exception:
                resample = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.BICUBIC
            resized = im.resize((img_arr.shape[1], img_arr.shape[0]), resample)
            dmap = np.asarray(resized, dtype=np.float32)
            if dmap.sum() != 0:
                dmap *= (orig_sum / dmap.sum())

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_arr)
        axes[0].axis('off')
        axes[0].set_title(f'image: {img_path.name}')

        im2 = axes[1].imshow(dmap, cmap='viridis')
        axes[1].set_title(title)
        axes[1].axis('off')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='density')
        plt.tight_layout()
        plt.show()
    else:
        # no image available — show only the density map
        plt.figure(figsize=(8, 6))
        show_density(dmap, cmap='viridis', title=title)
        plt.show()


if __name__ == '__main__':
    # Main behavior: pick a random training example and display image and density side-by-side.
    try:
        dmap_path, img_path = find_random_train_pair(Path('data'))
        print(f"Showing: density={dmap_path} image={img_path}")
        show_train_pair(dmap_path, img_path)
    except Exception as e:
        print('Error while picking/viewing random training pair:', e)
        raise
