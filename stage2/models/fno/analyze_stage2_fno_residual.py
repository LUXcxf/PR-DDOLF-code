import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common_stage2_fno import (
    DEFAULT_STAGE1_MODEL,
    STAGE2_DIR,
    Stage2ResidualDataset,
    build_holdout_edge_mask,
    build_trimmed_valid_mask,
    build_stage2_model,
    env_float,
    env_str,
    load_stage2_split,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_TAG = env_str("STAGE2_RUN_TAG", "fno_residual_masked_v1")
STAGE2_MODEL_PATH = Path(
    env_str(
        "STAGE2_MODEL_PATH",
        str(STAGE2_DIR / "results" / "fno" / "checkpoints" / f"fno_stage2_best_{RUN_TAG}.pth"),
    )
)
OUTPUT_TAG = env_str("STAGE2_OUTPUT_TAG", STAGE2_MODEL_PATH.stem)
EDGE_RATIO = env_float("STAGE2_EDGE_RATIO", 0.15)
OUTPUT_DIR = STAGE2_DIR / "results" / "fno" / "analysis" / OUTPUT_TAG


def masked_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    mask_sum = float(np.sum(mask))
    if mask_sum <= 0:
        return float("nan")
    return float(np.sqrt(np.sum(((pred - target) ** 2) * mask) / mask_sum))


def masked_mean_abs(x: np.ndarray, mask: np.ndarray) -> float:
    mask_sum = float(np.sum(mask))
    if mask_sum <= 0:
        return float("nan")
    return float(np.sum(np.abs(x) * mask) / mask_sum)


def invalid_mean_abs(x: np.ndarray, mask: np.ndarray) -> float:
    invalid = 1.0 - mask
    return masked_mean_abs(x, invalid)


def safe_nanmean(values) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def edge_mask_from_valid(mask_1d: np.ndarray, edge_ratio: float) -> np.ndarray:
    valid_idx = np.where(mask_1d > 0.5)[0]
    out = np.zeros_like(mask_1d, dtype=np.float32)
    if len(valid_idx) == 0:
        return out
    edge_n = max(1, int(np.ceil(len(valid_idx) * edge_ratio)))
    chosen = np.concatenate([valid_idx[:edge_n], valid_idx[-edge_n:]])
    out[np.unique(chosen)] = 1.0
    return out


def main():
    if not STAGE2_MODEL_PATH.exists():
        raise FileNotFoundError(f"missing stage2 checkpoint: {STAGE2_MODEL_PATH}")

    checkpoint = torch.load(STAGE2_MODEL_PATH, map_location=DEVICE, weights_only=False)
    cfg = checkpoint["config"]
    trim_ratio = env_float("STAGE2_SUPERVISION_TRIM_RATIO", float(cfg.get("supervision_trim_ratio", 0.0)))
    stage1_model_path = Path(checkpoint.get("stage1_model_path", str(DEFAULT_STAGE1_MODEL)))

    split = load_stage2_split()
    test_dataset = Stage2ResidualDataset(split["test_files"], stage1_model_path, device=DEVICE)
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

    rows = []
    freq_rows = []
    for sample in test_dataset:
        grid_input = sample["grid_input"].unsqueeze(0).to(DEVICE)
        material_vec = sample["material_vec"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_residual = model(grid_input, (material_vec - material_mean) / material_std)[0].cpu().numpy()

        base = sample["base_grid"].cpu().numpy()
        target = sample["target_grid"].cpu().numpy()
        valid_mask = sample["valid_mask"].cpu().numpy()
        support_mask = build_trimmed_valid_mask(sample["valid_mask"], trim_ratio).cpu().numpy()
        holdout_mask = build_holdout_edge_mask(sample["valid_mask"], trim_ratio).cpu().numpy()
        refined = base + pred_residual

        sample_metrics = {"sample_id": sample["sample_id"]}
        channel_names = ["ep", "edp"]
        base_rmses = []
        refined_rmses = []
        support_base_rmses = []
        support_refined_rmses = []
        holdout_base_rmses = []
        holdout_refined_rmses = []
        for c_idx, channel in enumerate(channel_names):
            channel_mask = valid_mask[c_idx]
            channel_support_mask = support_mask[c_idx]
            channel_holdout_mask = holdout_mask[c_idx]
            base_rmse = masked_rmse(base[c_idx], target[c_idx], channel_mask)
            refined_rmse = masked_rmse(refined[c_idx], target[c_idx], channel_mask)
            support_base_rmse = masked_rmse(base[c_idx], target[c_idx], channel_support_mask)
            support_refined_rmse = masked_rmse(refined[c_idx], target[c_idx], channel_support_mask)
            holdout_base_rmse = masked_rmse(base[c_idx], target[c_idx], channel_holdout_mask)
            holdout_refined_rmse = masked_rmse(refined[c_idx], target[c_idx], channel_holdout_mask)
            residual_scale = masked_mean_abs(pred_residual[c_idx], channel_mask) / max(
                masked_mean_abs(base[c_idx], channel_mask), 1e-8
            )
            invalid_residual_scale = invalid_mean_abs(pred_residual[c_idx], channel_mask) / max(
                invalid_mean_abs(base[c_idx], channel_mask), 1e-8
            )
            sample_metrics[f"base_rmse_{channel}"] = base_rmse
            sample_metrics[f"refined_rmse_{channel}"] = refined_rmse
            sample_metrics[f"improve_ratio_{channel}"] = float((base_rmse - refined_rmse) / max(base_rmse, 1e-8))
            sample_metrics[f"support_base_rmse_{channel}"] = support_base_rmse
            sample_metrics[f"support_refined_rmse_{channel}"] = support_refined_rmse
            sample_metrics[f"support_improve_ratio_{channel}"] = float(
                (support_base_rmse - support_refined_rmse) / max(support_base_rmse, 1e-8)
            ) if not np.isnan(support_base_rmse) and not np.isnan(support_refined_rmse) else float("nan")
            sample_metrics[f"holdout_base_rmse_{channel}"] = holdout_base_rmse
            sample_metrics[f"holdout_refined_rmse_{channel}"] = holdout_refined_rmse
            sample_metrics[f"holdout_improve_ratio_{channel}"] = float(
                (holdout_base_rmse - holdout_refined_rmse) / max(holdout_base_rmse, 1e-8)
            ) if not np.isnan(holdout_base_rmse) and not np.isnan(holdout_refined_rmse) else float("nan")
            sample_metrics[f"residual_scale_{channel}"] = residual_scale
            sample_metrics[f"invalid_residual_scale_{channel}"] = invalid_residual_scale
            base_rmses.append(base_rmse)
            refined_rmses.append(refined_rmse)
            support_base_rmses.append(support_base_rmse)
            support_refined_rmses.append(support_refined_rmse)
            holdout_base_rmses.append(holdout_base_rmse)
            holdout_refined_rmses.append(holdout_refined_rmse)

            for f_idx, freq_label in enumerate(["1", "2", "5", "10"]):
                freq_mask = channel_mask[f_idx]
                freq_support_mask = channel_support_mask[f_idx]
                freq_holdout_mask = channel_holdout_mask[f_idx]
                edge_mask = edge_mask_from_valid(freq_mask, EDGE_RATIO)
                interior_mask = np.clip(freq_mask - edge_mask, a_min=0.0, a_max=1.0)
                freq_rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "channel": channel,
                        "freq_hz": freq_label,
                        "base_rmse": masked_rmse(base[c_idx, f_idx], target[c_idx, f_idx], freq_mask),
                        "refined_rmse": masked_rmse(refined[c_idx, f_idx], target[c_idx, f_idx], freq_mask),
                        "support_base_rmse": masked_rmse(base[c_idx, f_idx], target[c_idx, f_idx], freq_support_mask),
                        "support_refined_rmse": masked_rmse(refined[c_idx, f_idx], target[c_idx, f_idx], freq_support_mask),
                        "holdout_base_rmse": masked_rmse(base[c_idx, f_idx], target[c_idx, f_idx], freq_holdout_mask),
                        "holdout_refined_rmse": masked_rmse(refined[c_idx, f_idx], target[c_idx, f_idx], freq_holdout_mask),
                        "edge_base_rmse": masked_rmse(base[c_idx, f_idx], target[c_idx, f_idx], edge_mask),
                        "edge_refined_rmse": masked_rmse(refined[c_idx, f_idx], target[c_idx, f_idx], edge_mask),
                        "interior_base_rmse": masked_rmse(base[c_idx, f_idx], target[c_idx, f_idx], interior_mask),
                        "interior_refined_rmse": masked_rmse(refined[c_idx, f_idx], target[c_idx, f_idx], interior_mask),
                    }
                )

        sample_metrics["base_rmse_mean"] = float(np.mean(base_rmses))
        sample_metrics["refined_rmse_mean"] = float(np.mean(refined_rmses))
        sample_metrics["improve_ratio_mean"] = float(
            (sample_metrics["base_rmse_mean"] - sample_metrics["refined_rmse_mean"]) / max(sample_metrics["base_rmse_mean"], 1e-8)
        )
        sample_metrics["support_base_rmse_mean"] = safe_nanmean(support_base_rmses)
        sample_metrics["support_refined_rmse_mean"] = safe_nanmean(support_refined_rmses)
        sample_metrics["support_improve_ratio_mean"] = float(
            (sample_metrics["support_base_rmse_mean"] - sample_metrics["support_refined_rmse_mean"])
            / max(sample_metrics["support_base_rmse_mean"], 1e-8)
        ) if not np.isnan(sample_metrics["support_base_rmse_mean"]) and not np.isnan(sample_metrics["support_refined_rmse_mean"]) else float("nan")
        sample_metrics["holdout_base_rmse_mean"] = safe_nanmean(holdout_base_rmses)
        sample_metrics["holdout_refined_rmse_mean"] = safe_nanmean(holdout_refined_rmses)
        sample_metrics["holdout_improve_ratio_mean"] = float(
            (sample_metrics["holdout_base_rmse_mean"] - sample_metrics["holdout_refined_rmse_mean"])
            / max(sample_metrics["holdout_base_rmse_mean"], 1e-8)
        ) if not np.isnan(sample_metrics["holdout_base_rmse_mean"]) and not np.isnan(sample_metrics["holdout_refined_rmse_mean"]) else float("nan")
        rows.append(sample_metrics)

    summary_df = pd.DataFrame(rows).sort_values("improve_ratio_mean", ascending=False)
    freq_df = pd.DataFrame(freq_rows)
    summary_df.to_csv(OUTPUT_DIR / "per_sample_summary.csv", index=False, encoding="utf-8-sig")
    freq_df.to_csv(OUTPUT_DIR / "per_frequency_summary.csv", index=False, encoding="utf-8-sig")

    aggregate = {
        "n_samples": int(len(summary_df)),
        "supervision_trim_ratio": float(trim_ratio),
        "base_rmse_ep_mean": float(summary_df["base_rmse_ep"].mean()),
        "refined_rmse_ep_mean": float(summary_df["refined_rmse_ep"].mean()),
        "base_rmse_edp_mean": float(summary_df["base_rmse_edp"].mean()),
        "refined_rmse_edp_mean": float(summary_df["refined_rmse_edp"].mean()),
        "improve_ratio_ep_mean": float(summary_df["improve_ratio_ep"].mean()),
        "improve_ratio_edp_mean": float(summary_df["improve_ratio_edp"].mean()),
        "support_base_rmse_ep_mean": float(summary_df["support_base_rmse_ep"].mean()),
        "support_refined_rmse_ep_mean": float(summary_df["support_refined_rmse_ep"].mean()),
        "support_base_rmse_edp_mean": float(summary_df["support_base_rmse_edp"].mean()),
        "support_refined_rmse_edp_mean": float(summary_df["support_refined_rmse_edp"].mean()),
        "support_improve_ratio_ep_mean": float(summary_df["support_improve_ratio_ep"].mean()),
        "support_improve_ratio_edp_mean": float(summary_df["support_improve_ratio_edp"].mean()),
        "holdout_base_rmse_ep_mean": float(summary_df["holdout_base_rmse_ep"].mean()),
        "holdout_refined_rmse_ep_mean": float(summary_df["holdout_refined_rmse_ep"].mean()),
        "holdout_base_rmse_edp_mean": float(summary_df["holdout_base_rmse_edp"].mean()),
        "holdout_refined_rmse_edp_mean": float(summary_df["holdout_refined_rmse_edp"].mean()),
        "holdout_improve_ratio_ep_mean": float(summary_df["holdout_improve_ratio_ep"].mean()),
        "holdout_improve_ratio_edp_mean": float(summary_df["holdout_improve_ratio_edp"].mean()),
        "residual_scale_ep_mean": float(summary_df["residual_scale_ep"].mean()),
        "residual_scale_edp_mean": float(summary_df["residual_scale_edp"].mean()),
        "invalid_residual_scale_ep_mean": float(summary_df["invalid_residual_scale_ep"].mean()),
        "invalid_residual_scale_edp_mean": float(summary_df["invalid_residual_scale_edp"].mean()),
        "edge_base_rmse_mean": float(freq_df[["edge_base_rmse"]].mean().iloc[0]),
        "edge_refined_rmse_mean": float(freq_df[["edge_refined_rmse"]].mean().iloc[0]),
        "interior_base_rmse_mean": float(freq_df[["interior_base_rmse"]].mean().iloc[0]),
        "interior_refined_rmse_mean": float(freq_df[["interior_refined_rmse"]].mean().iloc[0]),
    }
    (OUTPUT_DIR / "aggregate_summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"saved stage2 analysis to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
