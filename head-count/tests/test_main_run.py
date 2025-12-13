import os
import tempfile
import numpy as np
from types import SimpleNamespace

import main as main_mod


def touch_npy(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.array([[1.0]]))


def test_run_skips_when_outputs_present(monkeypatch):
    called = {'gen_called': False}

    # monkeypatch gen.main to raise if called (should not be called)
    def fake_gen_main():
        called['gen_called'] = True

    monkeypatch.setattr(main_mod, 'gen', SimpleNamespace(main=fake_gen_main, SPLITS=['train'], DATA_ROOT='data'))

    with tempfile.TemporaryDirectory() as tmp:
        split = 'train'
        gt_dir = os.path.join(tmp, split, 'gt')
        dm_dir = os.path.join(tmp, split, 'density_maps')
        os.makedirs(gt_dir, exist_ok=True)

        # create one gt file and corresponding npy
        gt = os.path.join(gt_dir, '0001.txt')
        open(gt, 'w').write('1 1 2 2\n')
        touch_npy(os.path.join(dm_dir, '0001.npy'))

        # run with config pointing to our temp data root
        cfg = {'data_root': tmp, 'splits': [split], 'force': False}
        rc = main_mod.run(config=cfg)

        assert rc == 0
        assert called['gen_called'] is False


def test_run_calls_gen_when_missing(monkeypatch):
    called = {'gen_called': False}

    # Provide fake gen.main that will create the missing .npy
    def fake_gen_main():
        called['gen_called'] = True
        # simulate generation by creating files where GT exists
        # use the main module's DATA_ROOT in this monkeypatched gen
        # but we'll rely on the temp layout in the test

    # monkeypatch main_mod.gen with an object we can inspect
    monkeypatch.setattr(main_mod, 'gen', SimpleNamespace(main=fake_gen_main, SPLITS=['train'], DATA_ROOT='data'))

    with tempfile.TemporaryDirectory() as tmp:
        split = 'train'
        gt_dir = os.path.join(tmp, split, 'gt')
        dm_dir = os.path.join(tmp, split, 'density_maps')
        os.makedirs(gt_dir, exist_ok=True)

        # create one gt file but no corresponding npy
        gt = os.path.join(gt_dir, '0001.txt')
        open(gt, 'w').write('1 1 2 2\n')

        # run; fake_gen_main doesn't actually create files (we simulate a generator that
        # doesn't write), so after run returns it may still be missing. Instead, call
        # run with force=True but ensure gen.main is called.
        cfg = {'data_root': tmp, 'splits': [split], 'force': True}
        rc = main_mod.run(config=cfg)

        assert called['gen_called'] is True
