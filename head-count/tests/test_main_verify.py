import os
import tempfile
import numpy as np
from main import verify_outputs, expected_npy_files


def touch_npy(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # write a tiny array
    np.save(path, np.array([[1.0, 2.0]]))


def test_verify_outputs_missing_and_present():
    with tempfile.TemporaryDirectory() as tmp:
        # create GT directory and a corresponding density_maps directory
        split = "train"
        gt_dir = os.path.join(tmp, split, "gt")
        dm_dir = os.path.join(tmp, split, "density_maps")
        os.makedirs(gt_dir, exist_ok=True)

        # add 2 gt files
        gt1 = os.path.join(gt_dir, "0001.txt")
        gt2 = os.path.join(gt_dir, "0002.txt")
        open(gt1, "w").write("1 1 2 2\n")
        open(gt2, "w").write("5 5 2 2\n")

        # Now create only one corresponding .npy (simulate partially generated)
        touch_npy(os.path.join(dm_dir, "0001.npy"))

        # Expect verify_outputs to be False because 0002.npy is missing
        assert verify_outputs(tmp, [split]) is False

        # Now add the missing file and re-check
        touch_npy(os.path.join(dm_dir, "0002.npy"))
        assert verify_outputs(tmp, [split]) is True

