import os
from pathlib import Path

from scripts import view_density


def test_random_train_pair_loads():
    """Ensure that pick_random_train_pair returns a density .npy file and optionally a matching image path.

    This is a lightweight test: it does not open figures (no GUI) but verifies the loader works
    and that the density file is a 2D array. If an image exists for the same base name we confirm
    that file also exists (we do not require images to exist for every density file).
    """
    dpath, img_path = view_density.find_random_train_pair(Path('data'))
    assert dpath.exists(), f"Density path should exist: {dpath}"
    arr = view_density.load_density(dpath)
    assert arr.ndim == 2
    assert arr.sum() >= 0
    if img_path is not None:
        assert img_path.exists(), f"Expected paired image to exist: {img_path}"
