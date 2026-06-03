# PR-DDOLF code package

This package provides the core source code for the PR-DDOLF framework.
The experimental DMA data are not included because they are subject to
confidentiality and institutional data-use restrictions.

## Contents

```text
stage1/models/
  baselines/      Stage-I Pure-MLP and Soft-PINN scripts.
  cnn/            Stage-I CNN baseline scripts.
  transformer/    Stage-I Transformer scripts.

stage2/models/
  baselines/      Stage-II MLP residual baseline.
  fno/            Stage-II FNO residual-field correction scripts.
```

Documentation:

- `RUN_CONFIGS.md`: data-path assumptions and representative run settings.
- `METRICS.md`: metric definitions used for model evaluation.
- `CODE_AVAILABILITY.md`: code organization.
- `RELEASE_MANIFEST.md`: file inventory.

## Basic checks

Run commands from the package root after installing dependencies.

```bash
pip install -r requirements.txt
python -m py_compile stage2/models/fno/common_stage2_fno.py
```

Training and evaluation scripts expect processed DMA `.npz` files and split
JSON files under the project-relative paths described in `RUN_CONFIGS.md`.
