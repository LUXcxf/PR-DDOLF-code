
# -*- coding: utf-8 -*-
import os
import glob
import json
import sys
import time
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# =========================================================
# v1.3c:
# Public-release note.
#
# Public-release note.
# Public-release note.
# Public-release note.
# Public-release note.
# Public-release note.
# Public-release note.
# =========================================================

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE1_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = STAGE1_DIR / "data" / "npz_converted"
SPLIT_JSON = Path(
    os.getenv(
        "STAGE1_CNN_SPLIT_JSON",
        str(STAGE1_DIR / "data" / "splits" / "split_stage1_cnn.json"),
    )
)
RESULTS_DIR = Path(
    os.getenv(
        "STAGE1_CNN_RESULTS_DIR",
        str(STAGE1_DIR / "results" / "cnn"),
    )
)
USE_EXISTING_SPLIT = os.getenv("STAGE1_CNN_USE_EXISTING_SPLIT", "1") == "1"

MODEL_SAVE_BEST_FIT = RESULTS_DIR / "checkpoints" / "cnn_light10hz_bell_best_fit.pth"
MODEL_SAVE_BEST_10HZ = RESULTS_DIR / "checkpoints" / "cnn_light10hz_bell_best_consistency.pth"
LOG_TXT_PATH = RESULTS_DIR / "logs" / "log_cnn_light10hz_bell.txt"
LOG_JSONL_PATH = RESULTS_DIR / "logs" / "log_cnn_light10hz_bell.jsonl"
HISTORY_NPZ_PATH = RESULTS_DIR / "history" / "history_cnn_light10hz_bell.npz"
LOSS_PLOT_PATH = RESULTS_DIR / "plots" / "loss_cnn_light10hz_bell.png"
INIT_MODEL_PATH = os.getenv("STAGE1_CNN_INIT_MODEL", "").strip()

TARGET_BASE_FREQ_HZ = 1.0
EPOCHS = int(os.getenv("STAGE1_CNN_EPOCHS", "300"))
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_VAL = 512
LR = float(os.getenv("STAGE1_CNN_LR", "1e-3"))
WEIGHT_DECAY = float(os.getenv("STAGE1_CNN_WEIGHT_DECAY", "1e-4"))
PATIENCE = int(os.getenv("STAGE1_CNN_PATIENCE", "90"))
PROBE_EVERY = 10
PROBE_COUNT = 3

AUX_MAIN_HZ = [2.0, 5.0]
MAIN_AUX_PROB = float(os.getenv("STAGE1_CNN_MAIN_AUX_PROB", "0.50"))
MAIN_AUX_LOSS_MAX = float(os.getenv("STAGE1_CNN_MAIN_AUX_LOSS_MAX", "0.05"))
MAIN_AUX_RECON_LOSS_MAX = float(os.getenv("STAGE1_CNN_MAIN_AUX_RECON_LOSS_MAX", "0.00"))

# Public-release note.
FOCUS10_AUX_PROB = float(os.getenv("STAGE1_CNN_FOCUS10_AUX_PROB", "0.18"))
FOCUS10_AUX_LOSS_MAX = float(os.getenv("STAGE1_CNN_FOCUS10_AUX_LOSS_MAX", "0.015"))
FOCUS10_AUX_RECON_LOSS_MAX = float(os.getenv("STAGE1_CNN_FOCUS10_AUX_RECON_LOSS_MAX", "0.00"))

# Public-release note.
FOCUS10_CORE_WEIGHTS = [0.8, 1.5, 0.35]   # [C1, C2, EeLog]

# Public-release note.
TRADEOFF_W_REL10 = float(os.getenv("STAGE1_CNN_TRADEOFF_W_REL10", "0.7"))
TRADEOFF_W_RMSE10 = float(os.getenv("STAGE1_CNN_TRADEOFF_W_RMSE10", "0.3"))

for path in [
    MODEL_SAVE_BEST_FIT.parent,
    MODEL_SAVE_BEST_10HZ.parent,
    LOG_TXT_PATH.parent,
    LOG_JSONL_PATH.parent,
    HISTORY_NPZ_PATH.parent,
    LOSS_PLOT_PATH.parent,
]:
    path.mkdir(parents=True, exist_ok=True)


class FileLogger:
    def __init__(self, txt_path: str, jsonl_path: str):
        self.txt_path = str(txt_path)
        self.jsonl_path = str(jsonl_path)
        os.makedirs(os.path.dirname(self.txt_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
        open(self.txt_path, "w", encoding="utf-8").close()
        open(self.jsonl_path, "w", encoding="utf-8").close()

    def log(self, msg: str):
        try:
            print(msg)
        except UnicodeEncodeError:
            enc = sys.stdout.encoding or "utf-8"
            print(msg.encode(enc, errors="replace").decode(enc, errors="replace"))
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def log_json(self, payload: dict):
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_closest_freq(freqs: List[str], target_hz: float) -> Optional[str]:
    if len(freqs) == 0:
        return None
    vals = [float(f.replace("_", ".")) for f in freqs]
    idx = int(np.argmin([abs(v - target_hz) for v in vals]))
    return freqs[idx]


def safe_log10_np(x: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(np.asarray(x, dtype=np.float32), a_min=1e-2, a_max=None))


def load_npz_freqs(npz_data) -> List[str]:
    available_keys = npz_data.files
    freqs = sorted(
        list(set([k.split('_')[-1].replace('Hz', '') for k in available_keys if k.startswith('E_prime_temp_')])),
        key=lambda s: float(s.replace('_', '.'))
    )
    valid = []
    for f_str in freqs:
        need = [
            f"E_prime_temp_{f_str}Hz",
            f"E_prime_val_{f_str}Hz",
            f"E_double_prime_val_{f_str}Hz",
        ]
        if all(k in available_keys for k in need):
            valid.append(f_str)
    return valid


def to_repo_relative(path_str: str) -> str:
    path = Path(path_str).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def save_split_info(train_files, val_files, test_files, save_path=SPLIT_JSON):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_files": [to_repo_relative(x) for x in train_files],
        "val_files": [to_repo_relative(x) for x in val_files],
        "test_files": [to_repo_relative(x) for x in test_files],
        "train_ids": [os.path.splitext(os.path.basename(x))[0] for x in train_files],
        "val_ids": [os.path.splitext(os.path.basename(x))[0] for x in val_files],
        "test_ids": [os.path.splitext(os.path.basename(x))[0] for x in test_files],
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def load_split_info(split_path: Path):
    with open(split_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    def _resolve(items):
        out = []
        for item in items:
            candidate = resolve_repo_path(item)
            if candidate.exists():
                out.append(str(candidate))
        return out

    return (
        _resolve(payload.get("train_files", [])),
        _resolve(payload.get("val_files", [])),
        _resolve(payload.get("test_files", [])),
    )


def make_feat_curve(npz_data, freq_str: str, T_standard: np.ndarray) -> Optional[torch.Tensor]:
    try:
        base_T = np.asarray(npz_data[f"E_prime_temp_{freq_str}Hz"], dtype=np.float32)
        base_Ep = np.asarray(npz_data[f"E_prime_val_{freq_str}Hz"], dtype=np.float32)
        order = np.argsort(base_T)
        base_T = base_T[order]
        base_Ep = base_Ep[order]
        _, uniq_idx = np.unique(base_T, return_index=True)
        uniq_idx = np.sort(uniq_idx)
        base_T = base_T[uniq_idx]
        base_Ep = base_Ep[uniq_idx]
        if len(base_T) < 2:
            return None
        interp_func = interp1d(base_T, base_Ep, kind='linear', bounds_error=False, fill_value="extrapolate")
        base_Ep_standard = safe_log10_np(interp_func(T_standard))
        return torch.tensor(base_Ep_standard, dtype=torch.float32)
    except Exception:
        return None


class PolymerDMADataset(Dataset):
    def __init__(self, file_paths: List[str], logger: FileLogger, T_grid_points: int = 100, target_base_freq_hz: float = 1.0):
        super().__init__()
        self.samples = []
        self.T_standard = np.linspace(20, 180, T_grid_points)
        self.sample_meta = []
        self.skip_info = []

        logger.log(f"Processing {len(file_paths)} files...")
        rng = random.Random(SEED)

        for file_path in file_paths:
            sample_name = os.path.splitext(os.path.basename(file_path))[0]
            try:
                data = np.load(file_path, allow_pickle=True)
                freqs = load_npz_freqs(data)
                if not freqs:
                    self.skip_info.append({"file": file_path, "reason": "no_valid_freqs"})
                    continue

                base_freq = choose_closest_freq(freqs, target_base_freq_hz)
                base_feat_curve = make_feat_curve(data, base_freq, self.T_standard)
                if base_feat_curve is None:
                    self.skip_info.append({"file": file_path, "reason": "base_curve_invalid"})
                    continue

                alt_freqs = [f for f in freqs if f != base_freq]

                main_aux_map = {}
                for hz in AUX_MAIN_HZ:
                    chosen = choose_closest_freq(alt_freqs, hz)
                    if chosen is not None and chosen not in main_aux_map:
                        feat = make_feat_curve(data, chosen, self.T_standard)
                        if feat is not None:
                            main_aux_map[chosen] = feat

                focus10_freq = choose_closest_freq(alt_freqs, 10.0)
                focus10_feat = None
                if focus10_freq is not None and focus10_freq not in main_aux_map:
                    feat = make_feat_curve(data, focus10_freq, self.T_standard)
                    if feat is not None:
                        focus10_feat = feat

                base_freq_num = float(base_freq.replace('_', '.'))
                base_f_norm = np.log10(max(base_freq_num, 0.1))

                n_before = len(self.samples)
                main_aux_keys = list(main_aux_map.keys())

                for f_str in freqs:
                    T_arr = np.asarray(data[f"E_prime_temp_{f_str}Hz"], dtype=np.float32)
                    Ep_arr = np.asarray(data[f"E_prime_val_{f_str}Hz"], dtype=np.float32)
                    Edp_arr = np.asarray(data[f"E_double_prime_val_{f_str}Hz"], dtype=np.float32)

                    if not (len(T_arr) == len(Ep_arr) == len(Edp_arr)):
                        self.skip_info.append({"file": file_path, "reason": f"curve_length_mismatch@{f_str}"})
                        continue

                    freq_num = float(f_str.replace('_', '.'))
                    f_norm = np.log10(max(freq_num, 0.1))

                    for i in range(len(T_arr)):
                        t_val = float(T_arr[i])
                        ep_log = float(np.log10(max(Ep_arr[i], 1e-2)))
                        edp_log = float(np.log10(max(Edp_arr[i], 1e-2)))
                        t_norm = (t_val - 20.0) / 160.0

                        main_aux_freq = rng.choice(main_aux_keys) if main_aux_keys else None
                        main_aux_feat = main_aux_map[main_aux_freq] if main_aux_freq is not None else None

                        use_focus10 = (focus10_feat is not None) and (rng.random() < FOCUS10_AUX_PROB)

                        self.samples.append({
                            'feat_curve': base_feat_curve,
                            'omega_feat': torch.tensor([base_f_norm], dtype=torch.float32),
                            'T_target': torch.tensor([t_norm], dtype=torch.float32),
                            'omega_target': torch.tensor([f_norm], dtype=torch.float32),
                            'target_ep': torch.tensor([ep_log], dtype=torch.float32),
                            'target_edp': torch.tensor([edp_log], dtype=torch.float32),

                            'main_aux_feat_curve': main_aux_feat,
                            'main_aux_omega_feat': (
                                torch.tensor([np.log10(max(float(main_aux_freq.replace("_", ".")), 0.1))], dtype=torch.float32)
                                if main_aux_freq is not None else torch.tensor([0.0], dtype=torch.float32)
                            ),
                            'has_main_aux': 1 if main_aux_feat is not None else 0,

                            'focus10_feat_curve': focus10_feat if use_focus10 else None,
                            'focus10_omega_feat': (
                                torch.tensor([np.log10(max(float(focus10_freq.replace("_", ".")), 0.1))], dtype=torch.float32)
                                if (use_focus10 and focus10_freq is not None) else torch.tensor([0.0], dtype=torch.float32)
                            ),
                            'has_focus10': 1 if (use_focus10 and focus10_feat is not None) else 0,
                        })

                added = len(self.samples) - n_before
                if added == 0:
                    self.skip_info.append({"file": file_path, "reason": "no_points_added"})
                    continue

                self.sample_meta.append({
                    "sample_name": sample_name,
                    "file_path": file_path,
                    "base_freq": base_freq,
                    "n_points": added,
                    "available_freqs": freqs,
                    "main_aux_freqs": list(main_aux_map.keys()),
                    "focus10_freq": focus10_freq,
                })
            except Exception as e:
                self.skip_info.append({"file": file_path, "reason": f"{type(e).__name__}: {e}"})

        logger.log(f"Dataset construction complete: {len(self.samples)} point samples from {len(self.sample_meta)} samples.")
        if self.skip_info:
            logger.log(f"Skipped/warning files: {len(self.skip_info)}")
            for item in self.skip_info[:20]:
                logger.log(f"   - {os.path.basename(item['file'])}: {item['reason']}")
            if len(self.skip_info) > 20:
                logger.log(f"   ... {len(self.skip_info) - 20} remaining entries omitted")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        main_aux_feat = s['main_aux_feat_curve'] if s['main_aux_feat_curve'] is not None else torch.zeros_like(s['feat_curve'])
        focus10_feat = s['focus10_feat_curve'] if s['focus10_feat_curve'] is not None else torch.zeros_like(s['feat_curve'])
        return (
            s['feat_curve'],
            s['omega_feat'],
            s['T_target'],
            s['omega_target'],
            s['target_ep'],
            s['target_edp'],

            main_aux_feat,
            s['main_aux_omega_feat'],
            torch.tensor([s['has_main_aux']], dtype=torch.float32),

            focus10_feat,
            s['focus10_omega_feat'],
            torch.tensor([s['has_focus10']], dtype=torch.float32),
        )


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


class PINNPolymerModel(nn.Module):
    def __init__(self, num_maxwell=13):
        super().__init__()
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            Res1DBlock(32, 64, stride=2),
            Res1DBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.mlp_predictor = nn.Sequential(
            nn.Linear(128 + 1, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_maxwell * 2 + 3)
        )
        self.Tr = 100.0
        self.num_maxwell = num_maxwell

    def infer_params(self, feat_curve, omega_feat):
        feat_curve = feat_curve.float()
        omega_feat = omega_feat.float()
        feat_curve_1d = feat_curve.unsqueeze(1)
        cnn_features = self.cnn_extractor(feat_curve_1d).squeeze(-1)
        x_enc = torch.cat([cnn_features, omega_feat], dim=1)
        phys_params = self.mlp_predictor(x_enc)

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
            "C1": C1, "C2": C2, "E_e_log": E_e_log, "E_e": E_e,
            "E_i_log": E_i_log, "E_i": E_i, "tau_log": tau_log, "tau_i": tau_i,
        }

    def forward(self, feat_curve, omega_feat, T_target, omega_target):
        T_target = T_target.float()
        omega_target = omega_target.float()
        params = self.infer_params(feat_curve, omega_feat)

        C1 = params["C1"]
        C2 = params["C2"]
        E_e = params["E_e"]
        E_i = params["E_i"]
        tau_i = params["tau_i"]

        T_real = T_target * 160.0 + 20.0
        omega_real = 2.0 * math.pi * (10.0 ** omega_target)
        log_aT = -C1 * (T_real - self.Tr) / (C2 + (T_real - self.Tr))
        log_aT = torch.clamp(log_aT, min=-15.0, max=15.0)
        a_T = 10.0 ** log_aT

        omega_reduced = omega_real * a_T
        wt = torch.clamp(omega_reduced * tau_i, max=1e15)
        wt2 = wt ** 2
        denom = 1.0 + wt2

        E_prime_linear = E_e + torch.sum(E_i * (wt2 / denom), dim=1, keepdim=True)
        E_double_prime_linear = torch.sum(E_i * (wt / denom), dim=1, keepdim=True)

        pred_ep = torch.log10(E_prime_linear + 1e-2)
        pred_edp = torch.log10(E_double_prime_linear + 1e-2)
        return pred_ep, pred_edp, params


@dataclass
class ProbeSample:
    sample_name: str
    file_path: str
    all_freqs: List[str]


def choose_probe_samples(file_paths: List[str], n_probe: int = 3) -> List[ProbeSample]:
    if len(file_paths) == 0:
        return []
    chosen_idx = np.linspace(0, len(file_paths) - 1, min(n_probe, len(file_paths))).astype(int)
    out = []
    for idx in chosen_idx:
        fp = file_paths[idx]
        sample_name = os.path.splitext(os.path.basename(fp))[0]
        try:
            data = np.load(fp, allow_pickle=True)
            freqs = load_npz_freqs(data)
            if len(freqs) >= 2:
                out.append(ProbeSample(sample_name=sample_name, file_path=fp, all_freqs=freqs))
        except Exception:
            pass
    return out


@torch.no_grad()
def evaluate_probe_consistency(model: PINNPolymerModel, probe: ProbeSample, device: torch.device, T_standard: np.ndarray):
    data = np.load(probe.file_path, allow_pickle=True)
    param_vecs = []
    rows = []

    for anchor_freq in probe.all_freqs:
        feat_curve = make_feat_curve(data, anchor_freq, T_standard)
        if feat_curve is None:
            continue
        feat_curve = feat_curve.unsqueeze(0).to(device)
        omega_feat = torch.tensor([[np.log10(max(float(anchor_freq.replace("_", ".")), 0.1))]], dtype=torch.float32, device=device)
        params = model.infer_params(feat_curve, omega_feat)

        vec = torch.cat([params["C1"], params["C2"], params["E_e_log"], params["E_i_log"], params["tau_log"]], dim=1)
        param_vecs.append(vec.squeeze(0).detach().cpu().numpy())

        f_num = float(anchor_freq.replace("_", "."))
        T_arr = np.asarray(data[f"E_prime_temp_{anchor_freq}Hz"], dtype=np.float32)
        target_ep = safe_log10_np(np.asarray(data[f"E_prime_val_{anchor_freq}Hz"], dtype=np.float32))
        target_edp = safe_log10_np(np.asarray(data[f"E_double_prime_val_{anchor_freq}Hz"], dtype=np.float32))

        T_norm = torch.tensor(((T_arr - 20.0) / 160.0).reshape(-1, 1), dtype=torch.float32, device=device)
        omega_target = torch.full((len(T_arr), 1), np.log10(max(f_num, 0.1)), dtype=torch.float32, device=device)
        feat_rep = feat_curve.repeat(len(T_arr), 1)
        omega_feat_rep = omega_feat.repeat(len(T_arr), 1)
        pred_ep, pred_edp, _ = model(feat_rep, omega_feat_rep, T_norm, omega_target)

        rmse_ep = float(torch.sqrt(torch.mean((pred_ep.squeeze(1) - torch.tensor(target_ep, device=device)) ** 2)).detach().cpu())
        rmse_edp = float(torch.sqrt(torch.mean((pred_edp.squeeze(1) - torch.tensor(target_edp, device=device)) ** 2)).detach().cpu())

        rows.append({
            "anchor_freq": anchor_freq,
            "rmse_ep_self": rmse_ep,
            "rmse_edp_self": rmse_edp,
            "C1": float(params["C1"].detach().cpu().view(-1)[0]),
            "C2": float(params["C2"].detach().cpu().view(-1)[0]),
            "E_e_log": float(params["E_e_log"].detach().cpu().view(-1)[0]),
        })

    if len(param_vecs) >= 2:
        mat = np.stack(param_vecs, axis=0)
        mean_vec = np.mean(mat, axis=0)
        std_vec = np.std(mat, axis=0)
        cv_mean = float(np.mean(std_vec / (np.abs(mean_vec) + 1e-8)))
        rel_l2 = []
        rel_l2_10 = []
        freq_names = [r["anchor_freq"] for r in rows]
        for i in range(len(mat)):
            for j in range(i + 1, len(mat)):
                val = float(np.linalg.norm(mat[i] - mat[j]) / (np.linalg.norm(mat[i]) + 1e-8))
                rel_l2.append(val)
                if (freq_names[i] == "10" or freq_names[i] == "10_0" or freq_names[j] == "10" or freq_names[j] == "10_0"):
                    rel_l2_10.append(val)
        rel_l2_mean = float(np.mean(rel_l2)) if rel_l2 else 0.0
        rel_l2_10_mean = float(np.mean(rel_l2_10)) if rel_l2_10 else rel_l2_mean
    else:
        cv_mean = 0.0
        rel_l2_mean = 0.0
        rel_l2_10_mean = 0.0

    rmse10 = np.nan
    for r in rows:
        if r["anchor_freq"] in ["10", "10_0"]:
            rmse10 = r["rmse_edp_self"]
            break

    return {
        "sample_name": probe.sample_name,
        "n_anchors": len(rows),
        "param_cv_mean": cv_mean,
        "param_rel_l2_mean": rel_l2_mean,
        "param_rel_l2_10_mean": rel_l2_10_mean,
        "rmse10_edp": float(rmse10) if not np.isnan(rmse10) else np.nan,
        "rows": rows,
    }


def get_main_aux_weight(epoch: int) -> float:
    if epoch < 20:
        return 0.0
    if epoch < 80:
        return MAIN_AUX_LOSS_MAX * (epoch - 20) / 60.0
    return MAIN_AUX_LOSS_MAX


def get_main_aux_recon_weight(epoch: int) -> float:
    if epoch < 20:
        return 0.0
    if epoch < 80:
        return MAIN_AUX_RECON_LOSS_MAX * (epoch - 20) / 60.0
    return MAIN_AUX_RECON_LOSS_MAX


def get_focus10_aux_weight(epoch: int) -> float:
    # Public-release note.
    if epoch < 35:
        return 0.0
    if epoch < 90:
        return FOCUS10_AUX_LOSS_MAX * (epoch - 35) / 55.0
    if epoch < 170:
        return FOCUS10_AUX_LOSS_MAX
    if epoch < 240:
        return FOCUS10_AUX_LOSS_MAX * max(0.0, (240 - epoch) / 70.0)
    return 0.0


def get_focus10_aux_recon_weight(epoch: int) -> float:
    if epoch < 35:
        return 0.0
    if epoch < 90:
        return FOCUS10_AUX_RECON_LOSS_MAX * (epoch - 35) / 55.0
    if epoch < 170:
        return FOCUS10_AUX_RECON_LOSS_MAX
    if epoch < 240:
        return FOCUS10_AUX_RECON_LOSS_MAX * max(0.0, (240 - epoch) / 70.0)
    return 0.0


def core_anchor_consistency_loss(main_params, aux_params):
    main_core = torch.cat([main_params["C1"], main_params["C2"], main_params["E_e_log"]], dim=1)
    aux_core = torch.cat([aux_params["C1"], aux_params["C2"], aux_params["E_e_log"]], dim=1)
    scale = torch.tensor([30.0, 300.0, 10.0], device=main_core.device, dtype=main_core.dtype).view(1, 3)
    return F.smooth_l1_loss(main_core / scale, aux_core / scale)


def core_focus10_consistency_loss(main_params, aux_params):
    main_core = torch.cat([main_params["C1"], main_params["C2"], main_params["E_e_log"]], dim=1)
    aux_core = torch.cat([aux_params["C1"], aux_params["C2"], aux_params["E_e_log"]], dim=1)

    scale = torch.tensor([30.0, 300.0, 10.0], device=main_core.device, dtype=main_core.dtype).view(1, 3)
    diff = torch.abs(main_core / scale - aux_core / scale)

    weights = torch.tensor(FOCUS10_CORE_WEIGHTS, device=main_core.device, dtype=main_core.dtype).view(1, 3)
    weighted = diff * weights
    return weighted.mean()


def train_model():
    set_seed(SEED)
    logger = FileLogger(LOG_TXT_PATH, LOG_JSONL_PATH)
    logger.log("Stage-I CNN training: lightweight 10Hz bell correction.")
    logger.log(f"seed={SEED}")
    logger.log(f"data_dir={DATA_DIR}")
    logger.log(f"target_base_freq_hz={TARGET_BASE_FREQ_HZ}")
    logger.log(f"main_aux={AUX_MAIN_HZ}, main_prob={MAIN_AUX_PROB}, main_loss_max={MAIN_AUX_LOSS_MAX}")
    logger.log(f"main_aux_recon_loss_max={MAIN_AUX_RECON_LOSS_MAX}")
    logger.log(f"focus10_prob={FOCUS10_AUX_PROB}, focus10_loss_max={FOCUS10_AUX_LOSS_MAX}")
    logger.log(f"focus10_aux_recon_loss_max={FOCUS10_AUX_RECON_LOSS_MAX}")
    logger.log(f"focus10_core_weights={FOCUS10_CORE_WEIGHTS}")
    logger.log(f"tradeoff weights: rel10={TRADEOFF_W_REL10}, rmse10={TRADEOFF_W_RMSE10}")
    logger.log(f"init_model={INIT_MODEL_PATH if INIT_MODEL_PATH else 'none'}")

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not all_files:
        logger.log("Status updated.")
        return

    if USE_EXISTING_SPLIT and SPLIT_JSON.exists():
        train_files, val_files, test_files = load_split_info(SPLIT_JSON)
        logger.log(f"Reuse split: {SPLIT_JSON}")
    else:
        rng = np.random.default_rng(SEED)
        rng.shuffle(all_files)
        n = len(all_files)
        train_files = all_files[:int(0.8 * n)]
        val_files = all_files[int(0.8 * n):int(0.9 * n)]
        test_files = all_files[int(0.9 * n):]
        save_split_info(train_files, val_files, test_files, save_path=SPLIT_JSON)
    logger.log(f"Data split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    if len(train_files) == 0 or len(val_files) == 0:
        logger.log("Status updated.")
        return

    logger.log("Status updated.")
    train_dataset = PolymerDMADataset(train_files, logger, target_base_freq_hz=TARGET_BASE_FREQ_HZ)
    logger.log("Status updated.")
    val_dataset = PolymerDMADataset(val_files, logger, target_base_freq_hz=TARGET_BASE_FREQ_HZ)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.log("Status updated.")
        return

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False, num_workers=0, pin_memory=pin_memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"🚀 device={device}")

    model = PINNPolymerModel().to(device)
    if INIT_MODEL_PATH:
        init_path = Path(INIT_MODEL_PATH)
        if not init_path.is_absolute():
            init_path = REPO_ROOT / init_path
        if not init_path.exists():
            raise FileNotFoundError(f"Initialization model does not exist: {init_path}")
        init_state = torch.load(init_path, map_location=device)
        model.load_state_dict(init_state, strict=True)
        logger.log(f"Loaded initialization checkpoint: {init_path}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    probe_samples = choose_probe_samples(val_files, n_probe=PROBE_COUNT)
    if probe_samples:
        logger.log("Fixed probe samples:")
        for p in probe_samples:
            logger.log(f"   - {p.sample_name}: anchors={p.all_freqs}")

    best_val_loss = float('inf')
    best_fit_epoch = -1
    best_10hz_tradeoff = float('inf')
    best_10hz_epoch = -1

    history = {
        "train_total": [], "train_ep": [], "train_edp": [],
        "train_main_aux": [], "train_focus10_aux": [],
        "train_main_aux_recon": [], "train_focus10_aux_recon": [],
        "val_total": [], "val_ep": [], "val_edp": [],
        "probe_rel10_mean": [], "probe_rmse10_mean": [],
        "tenhz_tradeoff": [], "lr": [], "main_aux_w": [], "focus10_aux_w": [],
        "main_aux_recon_w": [], "focus10_aux_recon_w": []
    }

    logger.log("Status updated.")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        total_train_ep = 0.0
        total_train_edp = 0.0
        total_train_main_aux = 0.0
        total_train_focus10_aux = 0.0
        total_train_main_aux_recon = 0.0
        total_train_focus10_aux_recon = 0.0
        main_aux_w = get_main_aux_weight(epoch)
        focus10_aux_w = get_focus10_aux_weight(epoch)
        main_aux_recon_w = get_main_aux_recon_weight(epoch)
        focus10_aux_recon_w = get_focus10_aux_recon_weight(epoch)
        start_time = time.time()

        for (
            feat_curve, omega_feat, T_target, omega_target, target_ep, target_edp,
            main_aux_feat_curve, main_aux_omega_feat, has_main_aux,
            focus10_feat_curve, focus10_omega_feat, has_focus10
        ) in train_loader:
            feat_curve = feat_curve.to(device, non_blocking=True)
            omega_feat = omega_feat.to(device, non_blocking=True)
            T_target = T_target.to(device, non_blocking=True)
            omega_target = omega_target.to(device, non_blocking=True)
            target_ep = target_ep.to(device, non_blocking=True)
            target_edp = target_edp.to(device, non_blocking=True)

            main_aux_feat_curve = main_aux_feat_curve.to(device, non_blocking=True)
            main_aux_omega_feat = main_aux_omega_feat.to(device, non_blocking=True)
            has_main_aux = has_main_aux.to(device, non_blocking=True)

            focus10_feat_curve = focus10_feat_curve.to(device, non_blocking=True)
            focus10_omega_feat = focus10_omega_feat.to(device, non_blocking=True)
            has_focus10 = has_focus10.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                pred_ep, pred_edp, params_main = model(feat_curve, omega_feat, T_target, omega_target)
                loss_ep = criterion(pred_ep, target_ep)
                loss_edp = criterion(pred_edp, target_edp)
                loss = loss_ep + loss_edp

                main_aux_loss = torch.tensor(0.0, device=device)
                focus10_aux_loss = torch.tensor(0.0, device=device)
                main_aux_recon_loss = torch.tensor(0.0, device=device)
                focus10_aux_recon_loss = torch.tensor(0.0, device=device)

                if main_aux_w > 0 or main_aux_recon_w > 0:
                    use_mask = (has_main_aux.squeeze(1) > 0.5) & (torch.rand(len(has_main_aux), device=device) < MAIN_AUX_PROB)
                    if torch.any(use_mask):
                        aux_params = model.infer_params(main_aux_feat_curve[use_mask], main_aux_omega_feat[use_mask])
                        if main_aux_w > 0:
                            main_aux_loss = core_anchor_consistency_loss(
                                {k: v[use_mask] if v.shape[0] == feat_curve.shape[0] else v for k, v in params_main.items()},
                                aux_params
                            )
                            loss = loss + main_aux_w * main_aux_loss
                        if main_aux_recon_w > 0:
                            aux_pred_ep, aux_pred_edp, _ = model(
                                main_aux_feat_curve[use_mask],
                                main_aux_omega_feat[use_mask],
                                T_target[use_mask],
                                omega_target[use_mask],
                            )
                            main_aux_recon_loss = criterion(aux_pred_ep, target_ep[use_mask]) + criterion(aux_pred_edp, target_edp[use_mask])
                            loss = loss + main_aux_recon_w * main_aux_recon_loss

                if focus10_aux_w > 0 or focus10_aux_recon_w > 0:
                    use_mask10 = (has_focus10.squeeze(1) > 0.5)
                    if torch.any(use_mask10):
                        aux_params10 = model.infer_params(focus10_feat_curve[use_mask10], focus10_omega_feat[use_mask10])
                        if focus10_aux_w > 0:
                            focus10_aux_loss = core_focus10_consistency_loss(
                                {k: v[use_mask10] if v.shape[0] == feat_curve.shape[0] else v for k, v in params_main.items()},
                                aux_params10
                            )
                            loss = loss + focus10_aux_w * focus10_aux_loss
                        if focus10_aux_recon_w > 0:
                            aux10_pred_ep, aux10_pred_edp, _ = model(
                                focus10_feat_curve[use_mask10],
                                focus10_omega_feat[use_mask10],
                                T_target[use_mask10],
                                omega_target[use_mask10],
                            )
                            focus10_aux_recon_loss = criterion(aux10_pred_ep, target_ep[use_mask10]) + criterion(aux10_pred_edp, target_edp[use_mask10])
                            loss = loss + focus10_aux_recon_w * focus10_aux_recon_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            total_train_ep += loss_ep.item()
            total_train_edp += loss_edp.item()
            total_train_main_aux += float(main_aux_loss.detach().cpu())
            total_train_focus10_aux += float(focus10_aux_loss.detach().cpu())
            total_train_main_aux_recon += float(main_aux_recon_loss.detach().cpu())
            total_train_focus10_aux_recon += float(focus10_aux_recon_loss.detach().cpu())

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        avg_train_ep = total_train_ep / max(len(train_loader), 1)
        avg_train_edp = total_train_edp / max(len(train_loader), 1)
        avg_train_main_aux = total_train_main_aux / max(len(train_loader), 1)
        avg_train_focus10_aux = total_train_focus10_aux / max(len(train_loader), 1)
        avg_train_main_aux_recon = total_train_main_aux_recon / max(len(train_loader), 1)
        avg_train_focus10_aux_recon = total_train_focus10_aux_recon / max(len(train_loader), 1)

        model.eval()
        total_val_loss = 0.0
        total_val_ep = 0.0
        total_val_edp = 0.0
        with torch.no_grad():
            for (
                feat_curve, omega_feat, T_target, omega_target, target_ep, target_edp,
                main_aux_feat_curve, main_aux_omega_feat, has_main_aux,
                focus10_feat_curve, focus10_omega_feat, has_focus10
            ) in val_loader:
                feat_curve = feat_curve.to(device, non_blocking=True)
                omega_feat = omega_feat.to(device, non_blocking=True)
                T_target = T_target.to(device, non_blocking=True)
                omega_target = omega_target.to(device, non_blocking=True)
                target_ep = target_ep.to(device, non_blocking=True)
                target_edp = target_edp.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    pred_ep, pred_edp, _ = model(feat_curve, omega_feat, T_target, omega_target)
                    loss_ep = criterion(pred_ep, target_ep)
                    loss_edp = criterion(pred_edp, target_edp)
                    loss = loss_ep + loss_edp
                total_val_loss += loss.item()
                total_val_ep += loss_ep.item()
                total_val_edp += loss_edp.item()

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        avg_val_ep = total_val_ep / max(len(val_loader), 1)
        avg_val_edp = total_val_edp / max(len(val_loader), 1)

        probe_rel10_vals = []
        probe_rmse10_vals = []
        if probe_samples:
            for probe in probe_samples:
                info = evaluate_probe_consistency(model, probe, device, train_dataset.T_standard)
                probe_rel10_vals.append(info["param_rel_l2_10_mean"])
                if not np.isnan(info["rmse10_edp"]):
                    probe_rmse10_vals.append(info["rmse10_edp"])

        probe_rel10_mean = float(np.mean(probe_rel10_vals)) if probe_rel10_vals else 0.0
        probe_rmse10_mean = float(np.mean(probe_rmse10_vals)) if probe_rmse10_vals else 0.0
        tenhz_tradeoff = avg_val_loss + TRADEOFF_W_REL10 * probe_rel10_mean + TRADEOFF_W_RMSE10 * probe_rmse10_mean

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history["train_total"].append(avg_train_loss)
        history["train_ep"].append(avg_train_ep)
        history["train_edp"].append(avg_train_edp)
        history["train_main_aux"].append(avg_train_main_aux)
        history["train_focus10_aux"].append(avg_train_focus10_aux)
        history["train_main_aux_recon"].append(avg_train_main_aux_recon)
        history["train_focus10_aux_recon"].append(avg_train_focus10_aux_recon)
        history["val_total"].append(avg_val_loss)
        history["val_ep"].append(avg_val_ep)
        history["val_edp"].append(avg_val_edp)
        history["probe_rel10_mean"].append(probe_rel10_mean)
        history["probe_rmse10_mean"].append(probe_rmse10_mean)
        history["tenhz_tradeoff"].append(tenhz_tradeoff)
        history["lr"].append(current_lr)
        history["main_aux_w"].append(main_aux_w)
        history["focus10_aux_w"].append(focus10_aux_w)
        history["main_aux_recon_w"].append(main_aux_recon_w)
        history["focus10_aux_recon_w"].append(focus10_aux_recon_w)

        improved_fit = avg_val_loss < best_val_loss
        if improved_fit:
            best_val_loss = avg_val_loss
            best_fit_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_BEST_FIT)

        improved_10hz = tenhz_tradeoff < best_10hz_tradeoff
        if improved_10hz:
            best_10hz_tradeoff = tenhz_tradeoff
            best_10hz_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_BEST_10HZ)

        logger.log_json({
            "type": "epoch",
            "epoch": epoch + 1,
            "train_total": avg_train_loss,
            "train_ep": avg_train_ep,
            "train_edp": avg_train_edp,
            "train_main_aux": avg_train_main_aux,
            "train_focus10_aux": avg_train_focus10_aux,
            "train_main_aux_recon": avg_train_main_aux_recon,
            "train_focus10_aux_recon": avg_train_focus10_aux_recon,
            "val_total": avg_val_loss,
            "val_ep": avg_val_ep,
            "val_edp": avg_val_edp,
            "probe_rel10_mean": probe_rel10_mean,
            "probe_rmse10_mean": probe_rmse10_mean,
            "tenhz_tradeoff": tenhz_tradeoff,
            "main_aux_w": main_aux_w,
            "focus10_aux_w": focus10_aux_w,
            "main_aux_recon_w": main_aux_recon_w,
            "focus10_aux_recon_w": focus10_aux_recon_w,
            "lr": current_lr,
            "best_fit": best_val_loss,
            "best_fit_epoch": best_fit_epoch,
            "best_10hz_tradeoff": best_10hz_tradeoff,
            "best_10hz_epoch": best_10hz_epoch,
            "time_sec": time.time() - start_time,
            "improved_fit": improved_fit,
            "improved_10hz": improved_10hz,
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.log(
                f"✅ Epoch [{epoch + 1:03d}/{EPOCHS}] | Time {time.time() - start_time:.2f}s | "
                f"Train {avg_train_loss:.4f} (Ep {avg_train_ep:.4f}, Edp {avg_train_edp:.4f}, "
                f"MainAux {avg_train_main_aux:.6f}, F10Aux {avg_train_focus10_aux:.6f}, "
                f"MainRec {avg_train_main_aux_recon:.6f}, F10Rec {avg_train_focus10_aux_recon:.6f}) | "
                f"Val {avg_val_loss:.4f} (Ep {avg_val_ep:.4f}, Edp {avg_val_edp:.4f}) | "
                f"Rel10 {probe_rel10_mean:.4f} | RMSE10 {probe_rmse10_mean:.4f} | "
                f"MainW {main_aux_w:.4f} | F10W {focus10_aux_w:.4f} | "
                f"MainRecW {main_aux_recon_w:.4f} | F10RecW {focus10_aux_recon_w:.4f} | "
                f"LR {current_lr:.2e} | "
                f"BestFit {best_val_loss:.4f}@{best_fit_epoch} | Best10 {best_10hz_tradeoff:.4f}@{best_10hz_epoch}"
            )

        if probe_samples and ((epoch + 1) % PROBE_EVERY == 0 or epoch == 0):
            logger.log(f"🔎 Probe @ epoch {epoch + 1}：")
            for probe in probe_samples:
                try:
                    info = evaluate_probe_consistency(model, probe, device, train_dataset.T_standard)
                    logger.log(
                        f"   - {info['sample_name']} | anchors={info['n_anchors']} | "
                        f"param_cv_mean={info['param_cv_mean']:.4f} | param_rel_l2_mean={info['param_rel_l2_mean']:.4f} | "
                        f"param_rel_l2_10_mean={info['param_rel_l2_10_mean']:.4f} | rmse10_edp={info['rmse10_edp']:.4f}"
                    )
                    for row in info["rows"]:
                        logger.log(
                            f"      anchor={row['anchor_freq']}Hz | selfRMSE E'={row['rmse_ep_self']:.4f} | "
                            f"selfRMSE E''={row['rmse_edp_self']:.4f} | C1={row['C1']:.2f} | "
                            f"C2={row['C2']:.2f} | EeLog={row['E_e_log']:.2f}"
                        )
                    logger.log_json({"type": "probe", "epoch": epoch + 1, **info})
                except Exception as e:
                    logger.log(f"   - probe {probe.sample_name} failed: {type(e).__name__}: {e}")

        if (epoch + 1) - best_fit_epoch >= PATIENCE:
            logger.log(f"🛑 Early stopping at epoch {epoch + 1}, best fit epoch = {best_fit_epoch}")
            break

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_total"], label='Train Total')
    plt.plot(history["val_total"], label='Val Total')
    plt.plot(history["val_ep"], '--', label="Val E'")
    plt.plot(history["val_edp"], '--', label="Val E''")
    plt.plot(history["train_main_aux"], ':', label="Train MainAux")
    plt.plot(history["train_focus10_aux"], ':', label="Train Focus10Aux")
    plt.plot(history["probe_rel10_mean"], '-.', label="Probe Rel10 Mean")
    plt.title('v1.3c light-10hz bell training curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Metric')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=300)

    np.savez(HISTORY_NPZ_PATH, **{k: np.asarray(v, dtype=np.float32) for k, v in history.items()})
    logger.log(f"Loss curve saved: {LOSS_PLOT_PATH}")
    logger.log(f"Training history saved: {HISTORY_NPZ_PATH}")
    logger.log(f"Best-fit model saved: {MODEL_SAVE_BEST_FIT}")
    logger.log(f"Best 10Hz-tradeoff model saved: {MODEL_SAVE_BEST_10HZ}")
    logger.log("Stage-I CNN training: lightweight 10Hz bell correction.")


if __name__ == '__main__':
    train_model()
