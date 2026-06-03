import os
import glob
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

"""
Public-release English description.
- Public-release English description.
- Public-release English description.
- Public-release English description.
  1. Public-release evaluation goal.
  2. Public-release evaluation goal.
"""

# =========================
# CONFIG
# =========================
REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE1_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = STAGE1_DIR / "data" / "npz_converted"
MODEL_PATH = STAGE1_DIR / "results" / "cnn" / "checkpoints" / "cnn_core_anchor_best_fit.pth"
OUTPUT_DIR = STAGE1_DIR / "results" / "cnn" / "evaluation" / "single_channel_consistency"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CANONICAL_FREQS = [0.5, 1.0, 2.0, 5.0, 10.0]
T_MIN, T_MAX = 20.0, 180.0
T_GRID_POINTS = 100
T_STANDARD = np.linspace(T_MIN, T_MAX, T_GRID_POINTS, dtype=np.float32)

NUM_MAXWELL = 13
USE_TEST_SPLIT = True
SPLIT_JSON = STAGE1_DIR / "data" / "splits" / "split_stage1_cnn.json"
MIN_FREQS_PER_SAMPLE = 2

THRESHOLDS = {
    "param_cv_mean": 0.12,
    "param_cosine_min": 0.985,
    "param_rel_l2_mean": 0.05,
    "rmse_ep_max": 0.35,
    "rmse_edp_max": 0.45,
}


# =========================
# Public-release note.
# =========================
def safe_log10(x):
    return np.log10(np.clip(x, 1e-2, None))


def freq_to_key(freq):
    return f"{freq:g}".replace(".", "_")


def clean_xy_for_interp(T_arr, y_arr):
    T_arr = np.asarray(T_arr, dtype=np.float32).reshape(-1)
    y_arr = np.asarray(y_arr, dtype=np.float32).reshape(-1)

    mask = np.isfinite(T_arr) & np.isfinite(y_arr)
    T_arr = T_arr[mask]
    y_arr = y_arr[mask]

    if len(T_arr) < 2:
        return None, None

    order = np.argsort(T_arr)
    T_arr = T_arr[order]
    y_arr = y_arr[order]

    uniq_T, inverse = np.unique(T_arr, return_inverse=True)
    if len(uniq_T) < 2:
        return None, None

    uniq_y = np.zeros_like(uniq_T, dtype=np.float32)
    counts = np.zeros_like(uniq_T, dtype=np.int32)
    for i, idx in enumerate(inverse):
        uniq_y[idx] += y_arr[i]
        counts[idx] += 1
    uniq_y = uniq_y / np.clip(counts, 1, None)

    return uniq_T.astype(np.float32), uniq_y.astype(np.float32)


def interpolate_to_standard(T_arr, y_arr):
    T_clean, y_clean = clean_xy_for_interp(T_arr, y_arr)
    if T_clean is None:
        return None
    f = interp1d(T_clean, y_clean, kind="linear", bounds_error=False, fill_value="extrapolate", assume_sorted=True)
    out = f(T_STANDARD).astype(np.float32)
    if not np.all(np.isfinite(out)):
        return None
    return out


def parse_freqs(npz):
    available = []
    for k in npz.files:
        if k.startswith("E_prime_temp_") and k.endswith("Hz"):
            f_str = k.split("_")[-1].replace("Hz", "")
            if f_str not in available:
                available.append(f_str)
    return sorted(available, key=lambda x: float(x.replace("_", ".")))


def resolve_repo_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_eval_files():
    if USE_TEST_SPLIT and os.path.exists(SPLIT_JSON):
        with open(SPLIT_JSON, "r", encoding="utf-8") as f:
            split = json.load(f)
        test_files = split.get("test_files", [])
        paths = [str(resolve_repo_path(p)) for p in test_files if resolve_repo_path(p).exists()]
        if paths:
            return paths
    return sorted(glob.glob(os.path.join(str(DATA_DIR), "*.npz")))


# =========================
# Public-release note.
# =========================
class Res1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class PINNPolymerModelConsistency(nn.Module):
    def __init__(self, curve_dim=100, num_maxwell=13):
        super().__init__()
        self.num_maxwell = num_maxwell
        self.Tr = 100.0  # Fixed reference temperature used during training

        # Public-release note.
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            Res1DBlock(32, 64, stride=2),
            Res1DBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )

        # Public-release note.
        self.mlp_predictor = nn.Sequential(
            nn.Linear(128 + 1, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_maxwell * 2 + 3)  # C1, C2, E_e, 13*E_i, 13*tau_i = 29
        )

    def encode(self, feat_curve, omega_feat):
        # Public-release note.
        feat_curve_1d = feat_curve.unsqueeze(1)

        # Public-release note.
        z = self.cnn_extractor(feat_curve_1d).squeeze(-1)
        x_enc = torch.cat([z, omega_feat], dim=1)

        # Public-release note.
        phys_params = self.mlp_predictor(x_enc)

        # Public-release note.
        C1 = F.softplus(phys_params[:, 0:1]) + 5.0
        C2 = F.softplus(phys_params[:, 1:2]) + 100.0

        E_e_log = 10.0 * torch.sigmoid(phys_params[:, 2:3])
        E_e = 10.0 ** E_e_log

        E_i_start = 3
        E_i_end = E_i_start + self.num_maxwell
        E_i_log = 10.0 * torch.sigmoid(phys_params[:, E_i_start:E_i_end])
        E_i = 10.0 ** E_i_log

        tau_i_start = E_i_end
        tau_i_end = tau_i_start + self.num_maxwell
        tau_log = 25.0 * torch.sigmoid(phys_params[:, tau_i_start:tau_i_end]) - 15.0
        tau_i = 10.0 ** tau_log

        return {
            "C1": C1,
            "C2": C2,
            "E_e": E_e,
            "E_i": E_i,
            "tau_i": tau_i,
        }

    def reconstruct_surface(self, params, canonical_freqs, t_standard):
        device = params["C1"].device
        b = params["C1"].shape[0]
        f_num = len(canonical_freqs)
        t_num = len(t_standard)

        T_real = torch.tensor(t_standard, dtype=torch.float32, device=device).view(1, 1, t_num).repeat(b, f_num, 1)
        freq_tensor = torch.tensor(canonical_freqs, dtype=torch.float32, device=device).view(1, f_num, 1).repeat(b, 1,
                                                                                                                 t_num)
        omega_real = 2.0 * np.pi * freq_tensor

        C1 = params["C1"].view(b, 1, 1)
        C2 = params["C2"].view(b, 1, 1)

        # Public-release note.
        dT = T_real - self.Tr
        log_aT = -C1 * dT / (C2 + dT + 1e-6)
        log_aT = torch.clamp(log_aT, min=-15.0, max=15.0)
        aT = 10.0 ** log_aT
        omega_reduced = omega_real * aT

        E_e = params["E_e"].view(b, 1, 1)
        E_i = params["E_i"].view(b, 1, self.num_maxwell)
        tau_i = params["tau_i"].view(b, 1, self.num_maxwell)

        wt = omega_reduced.unsqueeze(-1) * tau_i.unsqueeze(2)
        wt = torch.clamp(wt, max=1e15)
        wt2 = wt ** 2
        denom = 1.0 + wt2

        # Public-release note.
        E_prime_linear = E_e + torch.sum(E_i.unsqueeze(2) * (wt2 / denom), dim=-1)
        E_double_prime_linear = torch.sum(E_i.unsqueeze(2) * (wt / denom), dim=-1)

        pred_ep = torch.log10(E_prime_linear + 1e-2)
        pred_edp = torch.log10(E_double_prime_linear + 1e-2)

        pred_ep = torch.nan_to_num(pred_ep, nan=0.0, posinf=12.0, neginf=-12.0)
        pred_edp = torch.nan_to_num(pred_edp, nan=0.0, posinf=12.0, neginf=-12.0)
        return pred_ep, pred_edp


# =========================
# Public-release note.
# =========================
def build_single_frequency_feature(npz, f_str):
    ep_t_key = f"E_prime_temp_{f_str}Hz"
    ep_v_key = f"E_prime_val_{f_str}Hz"

    if ep_t_key not in npz.files or ep_v_key not in npz.files:
        return None, None

    T_arr = npz[ep_t_key].astype(np.float32)
    Ep_arr = npz[ep_v_key].astype(np.float32)

    ep_std_lin = interpolate_to_standard(T_arr, Ep_arr)
    if ep_std_lin is None:
        return None, None

    ep_std = safe_log10(ep_std_lin)

    if not np.all(np.isfinite(ep_std)):
        return None, None

    # Public-release note.
    feat_curve = torch.tensor(ep_std, dtype=torch.float32).unsqueeze(0)
    omega_feat = torch.tensor([[math.log10(max(float(f_str.replace("_", ".")), 0.1))]], dtype=torch.float32)
    return feat_curve, omega_feat


def get_reconstruction_targets(npz, f_str):
    T_arr = npz[f"E_prime_temp_{f_str}Hz"].astype(np.float32)
    Ep_arr = npz[f"E_prime_val_{f_str}Hz"].astype(np.float32)
    Edp_arr = npz[f"E_double_prime_val_{f_str}Hz"].astype(np.float32)

    target_ep = safe_log10(Ep_arr).reshape(-1)
    target_edp = safe_log10(Edp_arr).reshape(-1)
    return T_arr, target_ep, target_edp


# =========================
# Public-release note.
# =========================
def named_param_dict(param_dict):
    out = {
        "C1": float(param_dict["C1"].detach().cpu().numpy().reshape(-1)[0]),
        "C2": float(param_dict["C2"].detach().cpu().numpy().reshape(-1)[0]),
        "E_e": float(param_dict["E_e"].detach().cpu().numpy().reshape(-1)[0]),
    }
    e_i = param_dict["E_i"].detach().cpu().numpy().reshape(-1)
    tau_i = param_dict["tau_i"].detach().cpu().numpy().reshape(-1)
    for i, v in enumerate(e_i, 1):
        out[f"E_{i}"] = float(v)
    for i, v in enumerate(tau_i, 1):
        out[f"tau_{i}"] = float(v)
    return out


def cosine_similarity(a, b, eps=1e-12):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def relative_l2(a, b, eps=1e-12):
    denom = np.linalg.norm(a) + eps
    return float(np.linalg.norm(a - b) / denom)


def summarize_parameter_consistency(sample_id, freq_param_records):
    freqs = list(freq_param_records.keys())
    param_names = list(freq_param_records[freqs[0]].keys())
    mat = np.array([[freq_param_records[f][p] for p in param_names] for f in freqs], dtype=np.float64)

    mean_vec = np.mean(mat, axis=0)
    std_vec = np.std(mat, axis=0)
    cv_vec = std_vec / (np.abs(mean_vec) + 1e-12)

    pair_cos = []
    pair_l2 = []
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            a = mat[i]
            b = mat[j]
            pair_cos.append(cosine_similarity(a, b))
            pair_l2.append(relative_l2(a, b))

    result = {
        "sample_id": sample_id,
        "n_freqs": len(freqs),
        "param_cv_mean": float(np.nanmean(cv_vec)),
        "param_cv_median": float(np.nanmedian(cv_vec)),
        "param_cv_max": float(np.nanmax(cv_vec)),
        "pairwise_cosine_mean": float(np.nanmean(pair_cos)) if pair_cos else np.nan,
        "pairwise_cosine_min": float(np.nanmin(pair_cos)) if pair_cos else np.nan,
        "pairwise_rel_l2_mean": float(np.nanmean(pair_l2)) if pair_l2 else np.nan,
        "pairwise_rel_l2_max": float(np.nanmax(pair_l2)) if pair_l2 else np.nan,
        "C1_cv": float(cv_vec[param_names.index("C1")]),
        "C2_cv": float(cv_vec[param_names.index("C2")]),
        "E_e_cv": float(cv_vec[param_names.index("E_e")]),
    }
    return result, param_names, mat


def readiness_flag(row):
    return (
            row["param_cv_mean"] <= THRESHOLDS["param_cv_mean"]
            and row["pairwise_cosine_min"] >= THRESHOLDS["param_cosine_min"]
            and row["pairwise_rel_l2_mean"] <= THRESHOLDS["param_rel_l2_mean"]
            and row["rmse_ep_mean"] <= THRESHOLDS["rmse_ep_max"]
            and row["rmse_edp_mean"] <= THRESHOLDS["rmse_edp_max"]
    )


# =========================
# Public-release note.
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    files = load_eval_files()
    if not files:
        raise FileNotFoundError(f"No evaluable files found, DATA_DIR={DATA_DIR}")

    model = PINNPolymerModelConsistency(num_maxwell=NUM_MAXWELL).to(DEVICE)
    # Public-release note.
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    per_sample_summary = []
    per_sample_recon_dfs = []
    per_frequency_params = []

    for path in files:
        sample_id = os.path.splitext(os.path.basename(path))[0]
        npz = np.load(path, allow_pickle=True)
        freqs = parse_freqs(npz)

        valid_freqs = []
        for f_str in freqs:
            ep_k = f"E_prime_val_{f_str}Hz"
            edp_k = f"E_double_prime_val_{f_str}Hz"
            if ep_k in npz.files and edp_k in npz.files:
                valid_freqs.append(f_str)

        if len(valid_freqs) < MIN_FREQS_PER_SAMPLE:
            continue

        freq_param_records = {}
        freq_param_tensors = {}

        for f_in in valid_freqs:
            # Public-release note.
            feat_curve, omega_feat = build_single_frequency_feature(npz, f_in)
            if feat_curve is None:
                continue
            feat_curve = feat_curve.to(DEVICE)
            omega_feat = omega_feat.to(DEVICE)

            with torch.no_grad():
                params = model.encode(feat_curve, omega_feat)

            named = named_param_dict(params)
            freq_param_records[f_in] = named
            freq_param_tensors[f_in] = params

            row = {"sample_id": sample_id, "input_freq_hz": float(f_in.replace("_", "."))}
            row.update(named)
            per_frequency_params.append(row)

        if len(freq_param_records) < MIN_FREQS_PER_SAMPLE:
            continue

        param_summary, param_names, mat = summarize_parameter_consistency(sample_id, freq_param_records)

        plt.figure(figsize=(14, 5))
        plt.imshow(np.log10(np.clip(np.abs(mat), 1e-12, None)), aspect="auto")
        plt.colorbar(label="log10(|parameter|)")
        plt.yticks(range(len(freq_param_records)), [f"{f}Hz" for f in freq_param_records.keys()])
        plt.xticks(range(len(param_names)), param_names, rotation=90)
        plt.title(f"Parameter consistency heatmap - sample {sample_id} (1 Channel)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{sample_id}_parameter_heatmap.png"), dpi=220)
        plt.close()

        recon_rows = []
        for f_in, params in freq_param_tensors.items():
            for f_target in valid_freqs:
                T_arr, target_ep, target_edp = get_reconstruction_targets(npz, f_target)

                bsz = len(T_arr)
                params_batched = {k: v.repeat(bsz, 1) for k, v in params.items()}

                with torch.no_grad():
                    pred_ep_surf, pred_edp_surf = model.reconstruct_surface(params_batched,
                                                                            [float(f_target.replace("_", "."))],
                                                                            T_arr.astype(np.float32))

                pred_ep = pred_ep_surf[:, 0, :].diag().detach().cpu().numpy()
                pred_edp = pred_edp_surf[:, 0, :].diag().detach().cpu().numpy()

                rmse_ep = float(np.sqrt(np.mean((pred_ep - target_ep) ** 2)))
                rmse_edp = float(np.sqrt(np.mean((pred_edp - target_edp) ** 2)))
                mae_ep = float(np.mean(np.abs(pred_ep - target_ep)))
                mae_edp = float(np.mean(np.abs(pred_edp - target_edp)))

                recon_rows.append({
                    "sample_id": sample_id,
                    "input_freq_hz": float(f_in.replace("_", ".")),
                    "target_freq_hz": float(f_target.replace("_", ".")),
                    "rmse_ep": rmse_ep,
                    "rmse_edp": rmse_edp,
                    "mae_ep": mae_ep,
                    "mae_edp": mae_edp,
                })

        recon_df = pd.DataFrame(recon_rows)
        per_sample_recon_dfs.append(recon_df)

        recon_summary = {
            "sample_id": sample_id,
            "rmse_ep_mean": float(recon_df["rmse_ep"].mean()),
            "rmse_ep_max": float(recon_df["rmse_ep"].max()),
            "rmse_edp_mean": float(recon_df["rmse_edp"].mean()),
            "rmse_edp_max": float(recon_df["rmse_edp"].max()),
            "mae_ep_mean": float(recon_df["mae_ep"].mean()),
            "mae_edp_mean": float(recon_df["mae_edp"].mean()),
        }

        merged = {**param_summary, **recon_summary}
        merged["ready_for_residual_learning"] = readiness_flag(merged)
        merged["comment"] = (
            "Single-channel descriptors are stable enough for residual training"
            if merged["ready_for_residual_learning"]
            else "Single-channel descriptor stability needs further improvement"
        )
        per_sample_summary.append(merged)

    if not per_sample_summary:
        raise RuntimeError("No evaluable samples were obtained. Check model and data paths.")

    summary_df = pd.DataFrame(per_sample_summary).sort_values(
        by=["ready_for_residual_learning", "param_cv_mean", "rmse_ep_mean"],
        ascending=[False, True, True]
    )
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "parameter_consistency_summary.csv"), index=False, encoding="utf-8-sig")

    recon_all_df = pd.concat(per_sample_recon_dfs, axis=0, ignore_index=True)
    recon_all_df.to_csv(os.path.join(OUTPUT_DIR, "per_sample_reconstruction_error.csv"), index=False,
                        encoding="utf-8-sig")

    raw_param_df = pd.DataFrame(per_frequency_params)
    raw_param_df.to_csv(os.path.join(OUTPUT_DIR, "per_sample_per_frequency_predicted_parameters.csv"), index=False,
                        encoding="utf-8-sig")

    ready_df = summary_df[["sample_id", "ready_for_residual_learning", "comment"]].copy()
    ready_df.to_csv(os.path.join(OUTPUT_DIR, "ready_for_residual_learning.csv"), index=False, encoding="utf-8-sig")

    global_summary = {
        "n_samples_evaluated": int(len(summary_df)),
        "ready_ratio": float(summary_df["ready_for_residual_learning"].mean()),
        "param_cv_mean_global": float(summary_df["param_cv_mean"].mean()),
        "pairwise_cosine_min_global_mean": float(summary_df["pairwise_cosine_min"].mean()),
        "pairwise_rel_l2_mean_global": float(summary_df["pairwise_rel_l2_mean"].mean()),
        "rmse_ep_mean_global": float(summary_df["rmse_ep_mean"].mean()),
        "rmse_edp_mean_global": float(summary_df["rmse_edp_mean"].mean()),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    print("Evaluation complete.")
    print(f"Number of evaluated samples: {global_summary['n_samples_evaluated']}")
    print(f"Ready ratio for residual training: {global_summary['ready_ratio']:.3f}")
    print(f"Mean descriptor CV: {global_summary['param_cv_mean_global']:.4f}")
    print(f"Mean minimum cosine similarity: {global_summary['pairwise_cosine_min_global_mean']:.4f}")
    print(f"Mean descriptor relative L2 difference: {global_summary['pairwise_rel_l2_mean_global']:.4f}")
    print(f"Mean E' RMSE(log10): {global_summary['rmse_ep_mean_global']:.4f}")
    print(f"Mean E'' RMSE(log10): {global_summary['rmse_edp_mean_global']:.4f}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
