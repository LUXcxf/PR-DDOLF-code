# Run Configurations

The scripts are organized around the original stage-wise project layout. They
expect external processed data and split files at the following project-relative
locations when training or evaluation is run:

```text
stage1/data/npz_converted/
stage1/data/splits/
stage2/data/splits/
```

The code uses environment variables for most training hyperparameters. The
settings below document the representative configuration used for the main
PR-DDOLF workflow.

## Stage-I Transformer

```bash
STAGE1_USE_EXISTING_SPLIT=1
STAGE1_FEATURE_MODE=ep_only
STAGE1_EPOCHS=320
STAGE1_BATCH_TRAIN=192
STAGE1_LR=8e-4
STAGE1_WEIGHT_DECAY=2e-4
STAGE1_MAIN_PARAM_VEC_CONS_WEIGHT=0.35
STAGE1_FOCUS10_PARAM_VEC_CONS_WEIGHT=0.60
STAGE1_MAIN_SPECTRUM_VEC_CONS_WEIGHT=0.06
STAGE1_FOCUS10_SPECTRUM_VEC_CONS_WEIGHT=0.12
STAGE1_SPECTRUM_SHAPE_PRIOR_WEIGHT=0.0015
STAGE1_HARD_SAMPLE_IDS=
```

`STAGE1_HARD_SAMPLE_IDS` is left empty for the main workflow. The hard-sample
interface is retained in the source code as an optional experiment hook.

## Stage-I CNN Baseline

```bash
STAGE1_CNN_USE_EXISTING_SPLIT=1
```

In this code package, the CNN consistency probe used for checkpoint tradeoff is
drawn from the validation split. The test split should be reserved for final
reporting.

## Stage-I End-to-End Baselines

```bash
STAGE1_BASELINE_USE_EXISTING_SPLIT=1
```

The Pure-MLP and Soft-PINN scripts use the same stage-wise split convention.

## Stage-II FNO Residual Correction

```bash
STAGE2_STAGE1_MODEL=<path_to_stage1_checkpoint>
STAGE2_EPOCHS=160
STAGE2_BATCH_SIZE=16
STAGE2_LR=5e-4
STAGE2_WIDTH=48
STAGE2_MODES1=4
STAGE2_MODES2=20
STAGE2_DEPTH=4
STAGE2_RESID_L2_WEIGHT=0.01
STAGE2_SMOOTH_WEIGHT=0.005
STAGE2_INVALID_RESID_WEIGHT=0.05
```

Stage-II model selection is based on validation-set refined RMSE tradeoff. The
test split is used for final evaluation.
