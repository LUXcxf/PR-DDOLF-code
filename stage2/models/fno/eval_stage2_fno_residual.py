import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from common_stage2_fno import (
    DEFAULT_STAGE1_MODEL,
    STAGE2_DIR,
    T_STANDARD,
    Stage2ResidualDataset,
    build_holdout_edge_mask,
    build_stage2_model,
    env_float,
    env_int,
    env_str,
    evaluate_stage2_model,
    load_stage2_split,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_TAG = env_str("STAGE2_RUN_TAG", "fno_residual_masked_ext_v1")
STAGE2_MODEL_PATH = Path(
    env_str(
        "STAGE2_MODEL_PATH",
        str(STAGE2_DIR / "results" / "fno" / "checkpoints" / f"fno_stage2_best_{RUN_TAG}.pth"),
    )
)
OUTPUT_TAG = env_str("STAGE2_OUTPUT_TAG", STAGE2_MODEL_PATH.stem)
PLOT_SAMPLE_IDS = [x.strip() for x in env_str("STAGE2_SAMPLE_IDS", "").split(",") if x.strip()]
MAX_SAMPLES = env_int("STAGE2_MAX_SAMPLES", 3)
OUTPUT_DIR = STAGE2_DIR / "results" / "fno" / "evaluation" / OUTPUT_TAG


def add_mask_spans(ax, mask_1d: np.ndarray, color: str, alpha: float):
    idx = np.where(mask_1d > 0.5)[0]
    if len(idx) == 0:
        return
    start = idx[0]
    prev = idx[0]
    spans = []
    for pos in idx[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        spans.append((start, prev))
        start = pos
        prev = pos
    spans.append((start, prev))
    for left_idx, right_idx in spans:
        left = float(T_STANDARD[max(left_idx, 0)])
        right = float(T_STANDARD[min(right_idx, len(T_STANDARD) - 1)])
        ax.axvspan(left, right, color=color, alpha=alpha, linewidth=0)


def masked_curve(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    y_plot = np.asarray(y, dtype=np.float32).copy()
    y_plot[np.asarray(mask) < 0.5] = np.nan
    return y_plot


def support_bounds(mask_2d: np.ndarray):
    idx = np.where(np.any(np.asarray(mask_2d) > 0.5, axis=0))[0]
    if len(idx) == 0:
        return float(T_STANDARD[0]), float(T_STANDARD[-1])
    left = float(T_STANDARD[idx[0]])
    right = float(T_STANDARD[idx[-1]])
    pad = max(2.0, 0.04 * max(right - left, 1.0))
    left = max(float(T_STANDARD[0]), left - pad)
    right = min(float(T_STANDARD[-1]), right + pad)
    return left, right


def plot_sample(sample: dict, pred_residual: torch.Tensor, save_dir: Path, trim_ratio: float):
    base = sample["base_grid"].cpu().numpy()
    target = sample["target_grid"].cpu().numpy()
    valid_mask = sample["valid_mask"].cpu().numpy()
    holdout_mask = build_holdout_edge_mask(sample["valid_mask"], trim_ratio).cpu().numpy()
    refined = base + pred_residual.cpu().numpy()
    sample_id = sample["sample_id"]
    freq_labels = ["1", "2", "5", "10"]

    save_dir.mkdir(parents=True, exist_ok=True)
    for i, freq_label in enumerate(freq_labels):
        true_ep = masked_curve(target[0, i], valid_mask[0, i])
        true_edp = masked_curve(target[1, i], valid_mask[1, i])
        base_ep = masked_curve(base[0, i], valid_mask[0, i])
        base_edp = masked_curve(base[1, i], valid_mask[1, i])
        refined_ep = masked_curve(refined[0, i], valid_mask[0, i])
        refined_edp = masked_curve(refined[1, i], valid_mask[1, i])
        x_left, x_right = support_bounds(valid_mask[:, i])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(T_STANDARD, true_ep, color="black", linewidth=2, label="True E'")
        axes[0].plot(T_STANDARD, base_ep, color="royalblue", linestyle="--", linewidth=1.8, label="Stage1 E'")
        axes[0].plot(T_STANDARD, refined_ep, color="crimson", linewidth=1.8, label="Stage2 E'")
        add_mask_spans(axes[0], holdout_mask[0, i], color="gold", alpha=0.14)
        axes[0].set_title(f"{sample_id} | {freq_label}Hz | E'")
        axes[0].set_xlim(x_left, x_right)
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()

        axes[1].plot(T_STANDARD, true_edp, color="black", linewidth=2, label="True E''")
        axes[1].plot(T_STANDARD, base_edp, color="royalblue", linestyle="--", linewidth=1.8, label="Stage1 E''")
        axes[1].plot(T_STANDARD, refined_edp, color="crimson", linewidth=1.8, label="Stage2 E''")
        add_mask_spans(axes[1], holdout_mask[1, i], color="gold", alpha=0.14)
        axes[1].set_title(f"{sample_id} | {freq_label}Hz | E''")
        axes[1].set_xlim(x_left, x_right)
        axes[1].grid(True, alpha=0.25)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(save_dir / f"{sample_id}_{freq_label}Hz_compare.png", dpi=220)
        plt.close(fig)


def main():
    if not STAGE2_MODEL_PATH.exists():
        raise FileNotFoundError(f"missing stage2 checkpoint: {STAGE2_MODEL_PATH}")

    checkpoint = torch.load(STAGE2_MODEL_PATH, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["config"]
    trim_ratio = env_float("STAGE2_SUPERVISION_TRIM_RATIO", float(cfg.get("supervision_trim_ratio", 0.0)))
    stage1_model_path = Path(checkpoint.get("stage1_model_path", str(DEFAULT_STAGE1_MODEL)))
    split = load_stage2_split()
    test_dataset = Stage2ResidualDataset(split["test_files"], stage1_model_path, device=DEVICE)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = build_stage2_model(
        material_dim=int(test_dataset.samples[0].material_vec.numel()),
        width=int(cfg["width"]),
        modes1=int(cfg["modes1"]),
        modes2=int(cfg["modes2"]),
        depth=int(cfg["depth"]),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    material_mean = checkpoint["material_mean"].to(DEVICE)
    material_std = checkpoint["material_std"].to(DEVICE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_stage2_model(model, test_loader, material_mean, material_std, device=DEVICE)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    plotted = 0
    sample_id_set = set(PLOT_SAMPLE_IDS)
    with torch.no_grad():
        for sample in test_dataset:
            if sample_id_set:
                if sample["sample_id"] not in sample_id_set:
                    continue
            elif plotted >= MAX_SAMPLES:
                break
            grid_input = sample["grid_input"].unsqueeze(0).to(DEVICE)
            material_vec = sample["material_vec"].unsqueeze(0).to(DEVICE)
            pred_residual = model(grid_input, (material_vec - material_mean) / material_std)[0].cpu()
            plot_sample(sample, pred_residual, OUTPUT_DIR / "plots" / sample["sample_id"], trim_ratio)
            plotted += 1
            if sample_id_set and plotted >= len(sample_id_set):
                break
            if (not sample_id_set) and plotted >= MAX_SAMPLES:
                break

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved stage2 evaluation to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
