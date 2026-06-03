import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_JSON = STAGE1_DIR / "data" / "splits" / "split_stage1_transformer.json"
DATA_DIR = STAGE1_DIR / "data" / "npz_converted"
MODEL_PATH = Path(
    env_str(
        "STAGE1_TRANSFORMER_MODEL",
        str(STAGE1_DIR / "results" / "transformer" / "checkpoints" / "transformer_stage1_best_consistency_anchormix2edp_clean_v1.pth"),
    )
)
OUTPUT_TAG = env_str("STAGE1_OUTPUT_TAG", MODEL_PATH.stem)
OUTPUT_DIR = STAGE1_DIR / "results" / "transformer" / "plots" / OUTPUT_TAG
PLOT_POINTS = env_int("STAGE1_PLOT_POINTS", 220)
ANCHOR_TARGET_HZ = float(env_str("STAGE1_ANCHOR_HZ", "1.0"))
USE_TEST_SPLIT = env_int("STAGE1_USE_TEST_SPLIT", 1) != 0
MAX_SAMPLES = env_int("STAGE1_MAX_SAMPLES", 6)
SELECTED_SAMPLE_IDS = [x.strip() for x in env_str("STAGE1_SAMPLE_IDS", "").split(",") if x.strip()]
T_STANDARD = np.linspace(20.0, 180.0, T_GRID_POINTS, dtype=np.float32)


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_candidate_files():
    if USE_TEST_SPLIT and SPLIT_JSON.exists():
        with open(SPLIT_JSON, "r", encoding="utf-8") as f:
            split = json.load(f)
        paths = [resolve_repo_path(p) for p in split.get("test_files", [])]
        paths = [str(p.resolve()) for p in paths if p.exists()]
        if paths:
            return paths
    return sorted(str(p.resolve()) for p in DATA_DIR.glob("*.npz"))


def select_temperature_grid(npz, freq_str: str):
    temp_keys = [f"E_prime_temp_{freq_str}Hz", f"E_double_prime_temp_{freq_str}Hz"]
    temps = []
    for key in temp_keys:
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


def build_anchor_inputs(npz, anchor_freq: str, feature_mode: str):
    feat_curve = make_feat_curve(npz, anchor_freq, T_STANDARD, feature_mode=feature_mode)
    if feat_curve is None:
        return None, None
    omega_feat = torch.tensor(
        [[np.log10(max(float(anchor_freq.replace("_", ".")), 0.1))]],
        dtype=torch.float32,
        device=DEVICE,
    )
    return feat_curve.unsqueeze(0).to(DEVICE), omega_feat


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
        np.log10(max(float(target_freq.replace("_", ".")), 0.1)),
        dtype=torch.float32,
        device=DEVICE,
    )
    params_batched = repeat_params_for_curve(params, len(t_values))
    pred_ep, pred_edp = model.physics_decode(params_batched, t_norm, omega_target)
    return pred_ep.detach().cpu().numpy().reshape(-1), pred_edp.detach().cpu().numpy().reshape(-1)


def choose_sample_files():
    all_files = load_candidate_files()
    if SELECTED_SAMPLE_IDS:
        wanted = set(SELECTED_SAMPLE_IDS)
        return [path for path in all_files if Path(path).stem in wanted]
    return all_files[:MAX_SAMPLES]


def plot_one_sample(model, sample_path: str, feature_mode: str):
    sample_id = Path(sample_path).stem
    npz = np.load(sample_path, allow_pickle=True)
    valid_freqs = load_npz_freqs(npz)
    if len(valid_freqs) < 1:
        print(f"skip {sample_id}: no valid frequencies")
        return

    anchor_freq = choose_closest_freq(valid_freqs, ANCHOR_TARGET_HZ) or valid_freqs[0]
    feat_curve, omega_feat = build_anchor_inputs(npz, anchor_freq, feature_mode=feature_mode)
    if feat_curve is None:
        print(f"skip {sample_id}: failed to build anchor feature")
        return

    with torch.no_grad():
        params = model.infer_params(feat_curve, omega_feat)

    save_dir = OUTPUT_DIR / sample_id
    save_dir.mkdir(parents=True, exist_ok=True)

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
            plt.scatter(t_ep, y_ep, color="darkred", s=26, label=f"True E' ({target_freq}Hz)")

        edp_temp_key = f"E_double_prime_temp_{target_freq}Hz"
        edp_val_key = f"E_double_prime_val_{target_freq}Hz"
        if edp_temp_key in npz.files and edp_val_key in npz.files:
            t_edp = np.asarray(npz[edp_temp_key], dtype=np.float32)
            y_edp = safe_log10_np(np.asarray(npz[edp_val_key], dtype=np.float32))
            plt.scatter(t_edp, y_edp, color="darkblue", marker="x", s=30, label=f"True E'' ({target_freq}Hz)")

        plt.title(f"Transformer Stage1 Fit | sample {sample_id} | anchor {anchor_freq}Hz -> target {target_freq}Hz")
        plt.xlabel("Temperature (C)")
        plt.ylabel("log10 Modulus")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_path = save_dir / f"{sample_id}_anchor_{anchor_freq}Hz_target_{target_freq}Hz.png"
        plt.savefig(save_path, dpi=240)
        plt.close()
        print(f"saved {save_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"checkpoint not found: {MODEL_PATH}")

    sample_files = choose_sample_files()
    if not sample_files:
        raise FileNotFoundError("no sample files selected for plotting")

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

    print(f"device = {DEVICE}")
    print(f"model = {MODEL_PATH}")
    print(f"n_samples = {len(sample_files)}")
    for sample_path in sample_files:
        plot_one_sample(model, sample_path, feature_mode=feature_mode)


if __name__ == "__main__":
    main()
