import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from common_stage2_fno import (
    DEFAULT_STAGE1_MODEL,
    STAGE2_DIR,
    STAGE2_SPLIT_JSON,
    Stage2ResidualDataset,
    build_trimmed_valid_mask,
    build_stage2_model,
    compute_stage2_losses,
    env_float,
    env_int,
    env_str,
    evaluate_stage2_model,
    load_stage2_split,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_TAG = env_str("STAGE2_RUN_TAG", "fno_residual_v1")
STAGE1_MODEL_PATH = Path(env_str("STAGE2_STAGE1_MODEL", str(DEFAULT_STAGE1_MODEL)))
RESULTS_DIR = STAGE2_DIR / "results" / "fno"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
PLOT_DIR = RESULTS_DIR / "plots"
EVAL_DIR = RESULTS_DIR / "evaluation"
HISTORY_DIR = RESULTS_DIR / "history"

EPOCHS = env_int("STAGE2_EPOCHS", 160)
BATCH_SIZE = env_int("STAGE2_BATCH_SIZE", 16)
LR = env_float("STAGE2_LR", 2e-3)
WEIGHT_DECAY = env_float("STAGE2_WEIGHT_DECAY", 1e-4)
PATIENCE = env_int("STAGE2_PATIENCE", 40)
WIDTH = env_int("STAGE2_WIDTH", 48)
MODES1 = env_int("STAGE2_MODES1", 4)
MODES2 = env_int("STAGE2_MODES2", 20)
DEPTH = env_int("STAGE2_DEPTH", 4)
RESID_L2_WEIGHT = env_float("STAGE2_RESID_L2_WEIGHT", 0.01)
SMOOTH_WEIGHT = env_float("STAGE2_SMOOTH_WEIGHT", 0.005)
INVALID_RESID_WEIGHT = env_float("STAGE2_INVALID_RESID_WEIGHT", 0.05)
SUPERVISION_TRIM_RATIO = env_float("STAGE2_SUPERVISION_TRIM_RATIO", 0.0)

MODEL_PATH = CHECKPOINT_DIR / f"fno_stage2_best_{RUN_TAG}.pth"
HISTORY_PATH = HISTORY_DIR / f"history_fno_stage2_{RUN_TAG}.npz"
SUMMARY_PATH = EVAL_DIR / f"summary_fno_stage2_{RUN_TAG}.json"
PLOT_PATH = PLOT_DIR / f"loss_fno_stage2_{RUN_TAG}.png"


def main():
    set_seed(42)
    for folder in (CHECKPOINT_DIR, PLOT_DIR, EVAL_DIR, HISTORY_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    split = load_stage2_split()
    train_dataset = Stage2ResidualDataset(split["train_files"], STAGE1_MODEL_PATH, device=DEVICE)
    val_dataset = Stage2ResidualDataset(split["val_files"], STAGE1_MODEL_PATH, device=DEVICE)
    test_dataset = Stage2ResidualDataset(split["test_files"], STAGE1_MODEL_PATH, device=DEVICE)

    material_mean, material_std = train_dataset.material_stats()
    material_mean = material_mean.to(DEVICE)
    material_std = material_std.to(DEVICE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_stage2_model(
        material_dim=int(train_dataset.samples[0].material_vec.numel()),
        width=WIDTH,
        modes1=MODES1,
        modes2=MODES2,
        depth=DEPTH,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    history = {
        "train_total": [],
        "train_fit": [],
        "train_residual": [],
        "train_bound": [],
        "train_invalid": [],
        "train_smooth": [],
        "val_refined_ep": [],
        "val_refined_edp": [],
        "val_base_ep": [],
        "val_base_edp": [],
        "val_tradeoff": [],
    }
    best_tradeoff = float("inf")
    best_epoch = -1

    print("========== Stage-2 FNO Residual ==========")
    print(f"device = {DEVICE}")
    print(f"stage1_backbone = {STAGE1_MODEL_PATH}")
    print(f"stage2_split = {STAGE2_SPLIT_JSON}")
    print(f"train/val/test = {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    print(
        f"weights: resid_l2={RESID_L2_WEIGHT:.4f}, smooth={SMOOTH_WEIGHT:.4f}, invalid_resid={INVALID_RESID_WEIGHT:.4f}, "
        f"trim_ratio={SUPERVISION_TRIM_RATIO:.3f}"
    )

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        train_total = 0.0
        train_fit = 0.0
        train_residual = 0.0
        train_bound = 0.0
        train_invalid = 0.0
        train_smooth = 0.0

        for batch in train_loader:
            grid_input = batch["grid_input"].to(DEVICE)
            material_vec = batch["material_vec"].to(DEVICE)
            base_grid = batch["base_grid"].to(DEVICE)
            target_grid = batch["target_grid"].to(DEVICE)
            residual_grid = batch["residual_grid"].to(DEVICE)
            valid_mask = batch["valid_mask"].to(DEVICE)
            supervision_mask = build_trimmed_valid_mask(valid_mask, SUPERVISION_TRIM_RATIO)

            material_vec = (material_vec - material_mean) / material_std

            optimizer.zero_grad(set_to_none=True)
            pred_residual = model(grid_input, material_vec)
            losses = compute_stage2_losses(
                pred_residual,
                base_grid,
                target_grid,
                residual_grid,
                supervision_mask,
                resid_l2_weight=RESID_L2_WEIGHT,
                smooth_weight=SMOOTH_WEIGHT,
                invalid_resid_weight=INVALID_RESID_WEIGHT,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_total += float(losses["total"].detach().cpu())
            train_fit += float(losses["fit"].detach().cpu())
            train_residual += float(losses["residual"].detach().cpu())
            train_bound += float(losses["bound"].detach().cpu())
            train_invalid += float(losses["invalid"].detach().cpu())
            train_smooth += float(losses["smooth"].detach().cpu())

        scheduler.step()

        val_metrics = evaluate_stage2_model(model, val_loader, material_mean, material_std, device=DEVICE)
        tradeoff = val_metrics["refined_rmse_ep_mean"] + val_metrics["refined_rmse_edp_mean"]

        history["train_total"].append(train_total / max(len(train_loader), 1))
        history["train_fit"].append(train_fit / max(len(train_loader), 1))
        history["train_residual"].append(train_residual / max(len(train_loader), 1))
        history["train_bound"].append(train_bound / max(len(train_loader), 1))
        history["train_invalid"].append(train_invalid / max(len(train_loader), 1))
        history["train_smooth"].append(train_smooth / max(len(train_loader), 1))
        history["val_refined_ep"].append(val_metrics["refined_rmse_ep_mean"])
        history["val_refined_edp"].append(val_metrics["refined_rmse_edp_mean"])
        history["val_base_ep"].append(val_metrics["base_rmse_ep_mean"])
        history["val_base_edp"].append(val_metrics["base_rmse_edp_mean"])
        history["val_tradeoff"].append(tradeoff)

        if tradeoff < best_tradeoff:
            best_tradeoff = tradeoff
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "material_mean": material_mean.detach().cpu(),
                    "material_std": material_std.detach().cpu(),
                    "stage1_model_path": str(STAGE1_MODEL_PATH),
                    "run_tag": RUN_TAG,
                    "config": {
                        "width": WIDTH,
                        "modes1": MODES1,
                        "modes2": MODES2,
                        "depth": DEPTH,
                        "resid_l2_weight": RESID_L2_WEIGHT,
                        "smooth_weight": SMOOTH_WEIGHT,
                        "invalid_resid_weight": INVALID_RESID_WEIGHT,
                        "supervision_trim_ratio": SUPERVISION_TRIM_RATIO,
                    },
                },
                MODEL_PATH,
            )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1:03d}/{EPOCHS}] | "
                f"Time {time.time() - t0:.2f}s | "
                f"Train total={history['train_total'][-1]:.5f} fit={history['train_fit'][-1]:.5f} "
                f"resid={history['train_residual'][-1]:.5f} invalid={history['train_invalid'][-1]:.5f} | "
                f"Val base=({val_metrics['base_rmse_ep_mean']:.4f}, {val_metrics['base_rmse_edp_mean']:.4f}) "
                f"refined=({val_metrics['refined_rmse_ep_mean']:.4f}, {val_metrics['refined_rmse_edp_mean']:.4f}) | "
                f"best={best_tradeoff:.5f}@{best_epoch}"
            )

        if best_epoch > 0 and (epoch + 1) - best_epoch >= PATIENCE:
            print(f"Early stop at epoch {epoch + 1}")
            break

    np.savez(HISTORY_PATH, **{k: np.asarray(v, dtype=np.float32) for k, v in history.items()})

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_total"], label="Train total")
    plt.plot(history["val_refined_ep"], label="Val refined E'")
    plt.plot(history["val_refined_edp"], label="Val refined E''")
    plt.plot(history["val_base_ep"], "--", label="Val base E'")
    plt.plot(history["val_base_edp"], "--", label="Val base E''")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=240)
    plt.close()

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    test_metrics = evaluate_stage2_model(model, test_loader, material_mean, material_std, device=DEVICE)
    summary = {
        "stage1_model_path": str(STAGE1_MODEL_PATH),
        "best_epoch": best_epoch,
        "best_val_tradeoff": best_tradeoff,
        "supervision_trim_ratio": SUPERVISION_TRIM_RATIO,
        **test_metrics,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== Stage-2 Test Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved best model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
