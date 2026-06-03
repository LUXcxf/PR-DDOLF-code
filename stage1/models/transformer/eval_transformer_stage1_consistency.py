import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train_transformer_stage1 import (
    NUM_MAXWELL,
    PINNPolymerTransformer,
    REPO_ROOT,
    STAGE1_DIR,
    T_GRID_POINTS,
    choose_closest_freq,
    infer_transformer_config_from_state_dict,
    load_npz_freqs,
    make_feat_curve,
    safe_log10_np,
)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value is not None else default


def env_path(name: str, default: Path) -> Path:
    value = env_str(name, "")
    if not value:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


MODEL_PATH = Path(
    env_str(
        "STAGE1_TRANSFORMER_MODEL",
        str(STAGE1_DIR / "results" / "transformer" / "checkpoints" / "transformer_stage1_best_consistency_anchormix2edp_clean_v1.pth"),
    )
)
OUTPUT_TAG = env_str("STAGE1_OUTPUT_TAG", MODEL_PATH.stem)
SPLIT_JSON = env_path("STAGE1_SPLIT_JSON_OVERRIDE", STAGE1_DIR / "data" / "splits" / "split_stage1_transformer.json")
DATA_DIR = STAGE1_DIR / "data" / "npz_converted"
OUTPUT_DIR = STAGE1_DIR / "results" / "transformer" / "evaluation" / OUTPUT_TAG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_TEST_SPLIT = env_int("STAGE1_USE_TEST_SPLIT", 1) != 0
SPLIT_NAME = env_str("STAGE1_SPLIT_NAME", "test" if USE_TEST_SPLIT else "all").lower()
PLOT_SAMPLE_LIMIT = env_int("STAGE1_PLOT_SAMPLE_LIMIT", 6)
PLOT_POINTS = env_int("STAGE1_PLOT_POINTS", 220)
SAVE_HEATMAPS = env_int("STAGE1_SAVE_HEATMAPS", 1) != 0
SAVE_CURVE_PLOTS = env_int("STAGE1_SAVE_CURVE_PLOTS", 1) != 0
SELECTED_SAMPLE_IDS = {x.strip() for x in env_str("STAGE1_SAMPLE_IDS", "").split(",") if x.strip()}

THRESHOLDS = {
    "param_cv_mean": 0.12,
    "pairwise_rel_l2_mean": 0.05,
    "cosine_min": 0.985,
    "rmse_ep_mean": 0.35,
    "rmse_edp_mean": 0.45,
}

T_STANDARD = np.linspace(20.0, 180.0, T_GRID_POINTS, dtype=np.float32)


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_eval_files():
    if SPLIT_NAME in {"train", "val", "test"} and SPLIT_JSON.exists():
        with open(SPLIT_JSON, "r", encoding="utf-8") as f:
            split = json.load(f)
        paths = [resolve_repo_path(p) for p in split.get(f"{SPLIT_NAME}_files", [])]
        paths = [str(p.resolve()) for p in paths if p.exists()]
        if paths:
            return paths
    if USE_TEST_SPLIT and SPLIT_JSON.exists():
        with open(SPLIT_JSON, "r", encoding="utf-8") as f:
            split = json.load(f)
        paths = [resolve_repo_path(p) for p in split.get("test_files", [])]
        paths = [str(p.resolve()) for p in paths if p.exists()]
        if paths:
            return paths
    return sorted(glob.glob(os.path.join(str(DATA_DIR), "*.npz")))


def freq_to_float(freq_str: str) -> float:
    return float(freq_str.replace("_", "."))


def named_param_dict(params):
    out = {
        "C1": float(params["C1"].detach().cpu().view(-1)[0]),
        "C2": float(params["C2"].detach().cpu().view(-1)[0]),
        "E_e": float(params["E_e"].detach().cpu().view(-1)[0]),
    }
    e_i = params["E_i"].detach().cpu().numpy().reshape(-1)
    tau_i = params["tau_i"].detach().cpu().numpy().reshape(-1)
    for i, value in enumerate(e_i, 1):
        out[f"E_{i}"] = float(value)
    for i, value in enumerate(tau_i, 1):
        out[f"tau_{i}"] = float(value)
    return out


def build_anchor_inputs(npz, anchor_freq: str, feature_mode: str):
    feat_curve = make_feat_curve(npz, anchor_freq, T_STANDARD, feature_mode=feature_mode)
    if feat_curve is None:
        return None, None
    omega_feat = torch.tensor([[np.log10(max(freq_to_float(anchor_freq), 0.1))]], dtype=torch.float32)
    return feat_curve.unsqueeze(0), omega_feat


def select_temperature_grid(npz, freq_str: str):
    keys = [f"E_prime_temp_{freq_str}Hz", f"E_double_prime_temp_{freq_str}Hz"]
    temps = []
    for key in keys:
        if key in npz.files:
            arr = np.asarray(npz[key], dtype=np.float32).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                temps.append(arr)
    if not temps:
        raise ValueError(f"missing temperature arrays for {freq_str}Hz")
    merged = np.unique(np.concatenate(temps))
    if merged.size < 2:
        return merged
    return np.linspace(float(merged.min()), float(merged.max()), max(PLOT_POINTS, merged.size)).astype(np.float32)


def repeat_params_for_curve(params, n_points: int):
    out = {}
    for key in ("C1", "C2", "E_e", "E_i", "tau_i"):
        value = params[key]
        repeat_dims = [n_points] + [1] * (value.dim() - 1)
        out[key] = value.repeat(*repeat_dims)
    return out


def reconstruct_curve(model, params, target_freq: str, t_values: np.ndarray):
    t_norm = torch.tensor(((t_values - 20.0) / 160.0).reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    omega_target = torch.full(
        (len(t_values), 1),
        np.log10(max(freq_to_float(target_freq), 0.1)),
        dtype=torch.float32,
        device=DEVICE,
    )
    params_batched = repeat_params_for_curve(params, len(t_values))
    pred_ep, pred_edp = model.physics_decode(params_batched, t_norm, omega_target)
    return pred_ep.detach().cpu().numpy().reshape(-1), pred_edp.detach().cpu().numpy().reshape(-1)


def cosine_similarity(a, b, eps=1e-12):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < eps or norm_b < eps:
        return np.nan
    return float(np.dot(a, b) / (norm_a * norm_b))


def relative_l2(a, b, eps=1e-12):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + eps))


def summarize_parameter_consistency(sample_id: str, freq_param_records: dict):
    freqs = list(freq_param_records.keys())
    param_names = list(freq_param_records[freqs[0]].keys())
    mat = np.array([[freq_param_records[f][p] for p in param_names] for f in freqs], dtype=np.float64)

    mean_vec = np.mean(mat, axis=0)
    std_vec = np.std(mat, axis=0)
    cv_vec = std_vec / (np.abs(mean_vec) + 1e-12)

    core_names = ["C1", "C2", "E_e"]
    core_idx = [param_names.index(name) for name in core_names]

    pair_cos = []
    pair_l2 = []
    pair_l2_10 = []
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            a = mat[i]
            b = mat[j]
            pair_cos.append(cosine_similarity(a, b))
            rel = relative_l2(a, b)
            pair_l2.append(rel)
            if freqs[i] in {"10", "10_0"} or freqs[j] in {"10", "10_0"}:
                pair_l2_10.append(rel)

    result = {
        "sample_id": sample_id,
        "n_freqs": len(freqs),
        "param_cv_mean": float(np.nanmean(cv_vec)),
        "core_param_cv_mean": float(np.nanmean(cv_vec[core_idx])),
        "pairwise_rel_l2_mean": float(np.nanmean(pair_l2)) if pair_l2 else np.nan,
        "pairwise_rel_l2_max": float(np.nanmax(pair_l2)) if pair_l2 else np.nan,
        "probe_rel10_mean": float(np.nanmean(pair_l2_10)) if pair_l2_10 else np.nan,
        "cosine_mean": float(np.nanmean(pair_cos)) if pair_cos else np.nan,
        "cosine_min": float(np.nanmin(pair_cos)) if pair_cos else np.nan,
    }
    return result, param_names, mat


def readiness_flag(row):
    return (
        row["param_cv_mean"] <= THRESHOLDS["param_cv_mean"]
        and row["pairwise_rel_l2_mean"] <= THRESHOLDS["pairwise_rel_l2_mean"]
        and row["cosine_min"] >= THRESHOLDS["cosine_min"]
        and row["rmse_ep_mean"] <= THRESHOLDS["rmse_ep_mean"]
        and row["rmse_edp_mean"] <= THRESHOLDS["rmse_edp_mean"]
    )


def save_parameter_heatmap(sample_id: str, param_names, param_matrix, freq_names):
    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 5))
    plt.imshow(np.log10(np.clip(np.abs(param_matrix), 1e-12, None)), aspect="auto")
    plt.colorbar(label="log10(|parameter|)")
    plt.yticks(range(len(freq_names)), [f"{f}Hz" for f in freq_names])
    plt.xticks(range(len(param_names)), param_names, rotation=90)
    plt.title(f"Transformer parameter consistency - sample {sample_id}")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{sample_id}_parameter_heatmap.png", dpi=220)
    plt.close()


def save_curve_plots(model, npz, sample_id: str, anchor_freq: str, params, valid_freqs):
    plot_dir = OUTPUT_DIR / "plots" / "curves" / sample_id
    plot_dir.mkdir(parents=True, exist_ok=True)

    for target_freq in valid_freqs:
        t_plot = select_temperature_grid(npz, target_freq)
        pred_ep, pred_edp = reconstruct_curve(model, params, target_freq, t_plot)

        plt.figure(figsize=(10, 6))
        plt.plot(t_plot, pred_ep, color="red", linewidth=2, label=f"Pred E' ({target_freq}Hz)")
        plt.plot(t_plot, pred_edp, color="blue", linestyle="--", linewidth=2, label=f"Pred E'' ({target_freq}Hz)")

        ep_temp_key = f"E_prime_temp_{target_freq}Hz"
        ep_val_key = f"E_prime_val_{target_freq}Hz"
        if ep_temp_key in npz.files and ep_val_key in npz.files:
            t_ep = np.asarray(npz[ep_temp_key], dtype=np.float32)
            y_ep = safe_log10_np(np.asarray(npz[ep_val_key], dtype=np.float32))
            plt.scatter(t_ep, y_ep, color="darkred", s=28, label=f"True E' ({target_freq}Hz)")

        edp_temp_key = f"E_double_prime_temp_{target_freq}Hz"
        edp_val_key = f"E_double_prime_val_{target_freq}Hz"
        if edp_temp_key in npz.files and edp_val_key in npz.files:
            t_edp = np.asarray(npz[edp_temp_key], dtype=np.float32)
            y_edp = safe_log10_np(np.asarray(npz[edp_val_key], dtype=np.float32))
            plt.scatter(t_edp, y_edp, color="darkblue", marker="x", s=34, label=f"True E'' ({target_freq}Hz)")

        plt.title(f"Transformer fit - sample {sample_id} | anchor {anchor_freq}Hz -> target {target_freq}Hz")
        plt.xlabel("Temperature (C)")
        plt.ylabel("log10 Modulus")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{sample_id}_anchor_{anchor_freq}Hz_target_{target_freq}Hz.png", dpi=240)
        plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = load_eval_files()
    if not files:
        raise FileNotFoundError(f"no evaluation files found under {DATA_DIR}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"checkpoint not found: {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model_config = infer_transformer_config_from_state_dict(state)
    feature_mode = str(model_config["feature_mode"])
    model = PINNPolymerTransformer(
        seq_len=T_GRID_POINTS,
        num_maxwell=NUM_MAXWELL,
        d_model=int(model_config["d_model"]),
        nhead=int(model_config["nhead"]),
        num_layers=int(model_config["num_layers"]),
        dim_feedforward=int(model_config["dim_feedforward"]),
        dropout=float(model_config["dropout"]),
        feature_dim=int(model_config["feature_dim"]),
        token_stem=str(model_config["token_stem"]),
    ).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    per_sample_summary = []
    per_sample_recon_dfs = []
    per_frequency_params = []

    plotted = 0

    for path in files:
        sample_id = Path(path).stem
        npz = np.load(path, allow_pickle=True)
        valid_freqs = load_npz_freqs(npz)
        if len(valid_freqs) < 2:
            continue

        freq_param_records = {}
        freq_param_tensors = {}

        for anchor_freq in valid_freqs:
            feat_curve, omega_feat = build_anchor_inputs(npz, anchor_freq, feature_mode=feature_mode)
            if feat_curve is None:
                continue
            feat_curve = feat_curve.to(DEVICE)
            omega_feat = omega_feat.to(DEVICE)

            with torch.no_grad():
                params = model.infer_params(feat_curve, omega_feat)

            freq_param_records[anchor_freq] = named_param_dict(params)
            freq_param_tensors[anchor_freq] = {
                "C1": params["C1"],
                "C2": params["C2"],
                "E_e": params["E_e"],
                "E_i": params["E_i"],
                "tau_i": params["tau_i"],
            }

            row = {"sample_id": sample_id, "input_freq_hz": freq_to_float(anchor_freq)}
            row.update(freq_param_records[anchor_freq])
            per_frequency_params.append(row)

        if len(freq_param_records) < 2:
            continue

        param_summary, param_names, param_matrix = summarize_parameter_consistency(sample_id, freq_param_records)
        if SAVE_HEATMAPS:
            save_parameter_heatmap(sample_id, param_names, param_matrix, list(freq_param_records.keys()))

        recon_rows = []
        for anchor_freq, params in freq_param_tensors.items():
            for target_freq in valid_freqs:
                ep_temp_key = f"E_prime_temp_{target_freq}Hz"
                ep_val_key = f"E_prime_val_{target_freq}Hz"
                edp_temp_key = f"E_double_prime_temp_{target_freq}Hz"
                edp_val_key = f"E_double_prime_val_{target_freq}Hz"
                if not all(key in npz.files for key in (ep_temp_key, ep_val_key, edp_temp_key, edp_val_key)):
                    continue

                target_t = np.asarray(npz[ep_temp_key], dtype=np.float32)
                target_ep = safe_log10_np(np.asarray(npz[ep_val_key], dtype=np.float32))
                target_edp = safe_log10_np(np.asarray(npz[edp_val_key], dtype=np.float32))

                pred_ep, pred_edp = reconstruct_curve(model, params, target_freq, target_t)

                recon_rows.append(
                    {
                        "sample_id": sample_id,
                        "input_freq_hz": freq_to_float(anchor_freq),
                        "target_freq_hz": freq_to_float(target_freq),
                        "rmse_ep": float(np.sqrt(np.mean((pred_ep - target_ep) ** 2))),
                        "rmse_edp": float(np.sqrt(np.mean((pred_edp - target_edp) ** 2))),
                        "mae_ep": float(np.mean(np.abs(pred_ep - target_ep))),
                        "mae_edp": float(np.mean(np.abs(pred_edp - target_edp))),
                    }
                )

        if not recon_rows:
            continue

        recon_df = pd.DataFrame(recon_rows)
        per_sample_recon_dfs.append(recon_df)

        recon_summary = {
            "sample_id": sample_id,
            "rmse_ep_mean": float(recon_df["rmse_ep"].mean()),
            "rmse_edp_mean": float(recon_df["rmse_edp"].mean()),
            "mae_ep_mean": float(recon_df["mae_ep"].mean()),
            "mae_edp_mean": float(recon_df["mae_edp"].mean()),
            "probe_rmse10_mean": float(recon_df.loc[np.isclose(recon_df["target_freq_hz"], 10.0), "rmse_edp"].mean())
            if np.any(np.isclose(recon_df["target_freq_hz"], 10.0))
            else np.nan,
        }

        merged = {**param_summary, **recon_summary}
        merged["ready_for_stage2"] = readiness_flag(merged)
        per_sample_summary.append(merged)

        should_plot = sample_id in SELECTED_SAMPLE_IDS or (not SELECTED_SAMPLE_IDS and plotted < PLOT_SAMPLE_LIMIT)
        if should_plot and SAVE_CURVE_PLOTS:
            anchor_freq = choose_closest_freq(valid_freqs, 1.0) or valid_freqs[0]
            save_curve_plots(model, npz, sample_id, anchor_freq, freq_param_tensors[anchor_freq], valid_freqs)
            plotted += 1

    if not per_sample_summary:
        raise RuntimeError("no valid samples were evaluated")

    summary_df = pd.DataFrame(per_sample_summary).sort_values(
        by=["ready_for_stage2", "param_cv_mean", "rmse_ep_mean"],
        ascending=[False, True, True],
    )
    summary_df.to_csv(OUTPUT_DIR / "per_sample_summary.csv", index=False, encoding="utf-8-sig")

    recon_df = pd.concat(per_sample_recon_dfs, axis=0, ignore_index=True)
    recon_df.to_csv(OUTPUT_DIR / "per_sample_reconstruction_error.csv", index=False, encoding="utf-8-sig")

    raw_param_df = pd.DataFrame(per_frequency_params)
    raw_param_df.to_csv(OUTPUT_DIR / "per_sample_per_frequency_predicted_parameters.csv", index=False, encoding="utf-8-sig")

    global_summary = {
        "n_samples_evaluated": int(len(summary_df)),
        "ready_ratio": float(summary_df["ready_for_stage2"].mean()),
        "param_cv_mean": float(summary_df["param_cv_mean"].mean()),
        "core_param_cv_mean": float(summary_df["core_param_cv_mean"].mean()),
        "pairwise_rel_l2_mean": float(summary_df["pairwise_rel_l2_mean"].mean()),
        "cosine_mean": float(summary_df["cosine_mean"].mean()),
        "rmse_ep_mean": float(summary_df["rmse_ep_mean"].mean()),
        "rmse_edp_mean": float(summary_df["rmse_edp_mean"].mean()),
        "mae_ep_mean": float(summary_df["mae_ep_mean"].mean()),
        "mae_edp_mean": float(summary_df["mae_edp_mean"].mean()),
        "probe_rel10_mean": float(summary_df["probe_rel10_mean"].mean()),
        "probe_rmse10_mean": float(summary_df["probe_rmse10_mean"].mean()),
        "param_cv_mean_global": float(summary_df["param_cv_mean"].mean()),
        "pairwise_rel_l2_mean_global": float(summary_df["pairwise_rel_l2_mean"].mean()),
        "rmse_ep_mean_global": float(summary_df["rmse_ep_mean"].mean()),
        "rmse_edp_mean_global": float(summary_df["rmse_edp_mean"].mean()),
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(global_summary, ensure_ascii=False, indent=2))
    print(f"saved transformer evaluation to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
