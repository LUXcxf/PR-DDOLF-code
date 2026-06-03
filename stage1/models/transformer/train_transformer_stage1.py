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
from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


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


def env_str_list(name: str) -> List[str]:
    value = env_str(name, "")
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def env_float_list(name: str) -> List[float]:
    value = env_str(name, "")
    if not value:
        return []
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out


def with_run_tag(path: Path, run_tag: str) -> Path:
    if not run_tag:
        return path
    return path.with_name(f"{path.stem}_{run_tag}{path.suffix}")


# =========================================================
# Public-release note.
# =========================================================
SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE1_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = STAGE1_DIR / "data" / "npz_converted"
SPLIT_JSON = env_path("STAGE1_SPLIT_JSON_OVERRIDE", STAGE1_DIR / "data" / "splits" / "split_stage1_transformer.json")
RESULTS_DIR = STAGE1_DIR / "results" / "transformer"
RUN_TAG = env_str("STAGE1_RUN_TAG", "")
USE_EXISTING_SPLIT = env_int("STAGE1_USE_EXISTING_SPLIT", 1) != 0

MODEL_SAVE_BEST_FIT = with_run_tag(RESULTS_DIR / "checkpoints" / "transformer_stage1_best_fit.pth", RUN_TAG)
MODEL_SAVE_BEST_CONS = with_run_tag(RESULTS_DIR / "checkpoints" / "transformer_stage1_best_consistency.pth", RUN_TAG)
LOG_TXT_PATH = with_run_tag(RESULTS_DIR / "logs" / "log_transformer_stage1.txt", RUN_TAG)
LOG_JSONL_PATH = with_run_tag(RESULTS_DIR / "logs" / "log_transformer_stage1.jsonl", RUN_TAG)
HISTORY_NPZ_PATH = with_run_tag(RESULTS_DIR / "history" / "history_transformer_stage1.npz", RUN_TAG)
LOSS_PLOT_PATH = with_run_tag(RESULTS_DIR / "plots" / "loss_transformer_stage1.png", RUN_TAG)
EPOCH_CKPT_DIR = RESULTS_DIR / "checkpoints" / "epochs" / (RUN_TAG or "default")

TARGET_BASE_FREQ_HZ = 1.0
EXTRA_ANCHOR_HZ = env_float_list("STAGE1_EXTRA_ANCHOR_HZ")
EXTRA_ANCHOR_MODE = env_str("STAGE1_EXTRA_ANCHOR_MODE", "all")
T_GRID_POINTS = 100

EPOCHS = env_int("STAGE1_EPOCHS", 320)
BATCH_SIZE_TRAIN = env_int("STAGE1_BATCH_TRAIN", 192)
BATCH_SIZE_VAL = env_int("STAGE1_BATCH_VAL", 384)
LR = env_float("STAGE1_LR", 8e-4)
WEIGHT_DECAY = env_float("STAGE1_WEIGHT_DECAY", 2e-4)
PATIENCE = env_int("STAGE1_PATIENCE", 90)
MAX_FILES = env_int("STAGE1_MAX_FILES", 0)

# Public-release note.
NUM_MAXWELL = 13

PROBE_EVERY = env_int("STAGE1_PROBE_EVERY", 10)
PROBE_COUNT = env_int("STAGE1_PROBE_COUNT", 3)
PROBE_SPLIT_NAME = env_str("STAGE1_PROBE_SPLIT_NAME", "val").lower()
SAVE_EPOCH_CHECKPOINTS = env_int("STAGE1_SAVE_EPOCH_CHECKPOINTS", 0) != 0

# Public-release note.
AUX_MAIN_HZ = [2.0, 5.0]
MAIN_AUX_PROB = env_float("STAGE1_MAIN_AUX_PROB", 0.50)
MAIN_AUX_LOSS_MAX = env_float("STAGE1_MAIN_AUX_LOSS_MAX", 0.05)
MAIN_AUX_START_EPOCH = env_int("STAGE1_MAIN_AUX_START_EPOCH", 20)
MAIN_AUX_FULL_EPOCH = env_int("STAGE1_MAIN_AUX_FULL_EPOCH", 90)

# Public-release note.
FOCUS10_AUX_PROB = env_float("STAGE1_FOCUS10_AUX_PROB", 0.18)
FOCUS10_AUX_LOSS_MAX = env_float("STAGE1_FOCUS10_AUX_LOSS_MAX", 0.012)
FOCUS10_START_EPOCH = env_int("STAGE1_FOCUS10_START_EPOCH", 40)
FOCUS10_FULL_EPOCH = env_int("STAGE1_FOCUS10_FULL_EPOCH", 100)
FOCUS10_HOLD_EPOCH = env_int("STAGE1_FOCUS10_HOLD_EPOCH", 180)
FOCUS10_END_EPOCH = env_int("STAGE1_FOCUS10_END_EPOCH", 250)

# Public-release note.
TRADEOFF_W_VAL = env_float("STAGE1_TRADEOFF_W_VAL", 0.20)
TRADEOFF_W_CV = env_float("STAGE1_TRADEOFF_W_CV", 0.55)
TRADEOFF_W_REL = env_float("STAGE1_TRADEOFF_W_REL", 1.00)
TRADEOFF_W_REL10 = env_float("STAGE1_TRADEOFF_W_REL10", 0.65)
TRADEOFF_W_RMSE10 = env_float("STAGE1_TRADEOFF_W_RMSE10", 0.35)
BEST_CONS_START_EPOCH = env_int("STAGE1_BEST_CONS_START_EPOCH", 8)
MAIN_PARAM_VEC_CONS_WEIGHT = env_float("STAGE1_MAIN_PARAM_VEC_CONS_WEIGHT", 0.0)
FOCUS10_PARAM_VEC_CONS_WEIGHT = env_float("STAGE1_FOCUS10_PARAM_VEC_CONS_WEIGHT", 0.0)
MAIN_SPECTRUM_VEC_CONS_WEIGHT = env_float("STAGE1_MAIN_SPECTRUM_VEC_CONS_WEIGHT", 0.0)
FOCUS10_SPECTRUM_VEC_CONS_WEIGHT = env_float("STAGE1_FOCUS10_SPECTRUM_VEC_CONS_WEIGHT", 0.0)
SPECTRUM_SHAPE_PRIOR_WEIGHT = env_float("STAGE1_SPECTRUM_SHAPE_PRIOR_WEIGHT", 0.0)
HARD_SAMPLE_IDS = set(env_str_list("STAGE1_HARD_SAMPLE_IDS"))
HARD_SAMPLE_BOOST = env_float("STAGE1_HARD_SAMPLE_BOOST", 1.0)
HARD_MAIN_CONS_MULT = env_float("STAGE1_HARD_MAIN_CONS_MULT", 1.0)
HARD_FOCUS10_CONS_MULT = env_float("STAGE1_HARD_FOCUS10_CONS_MULT", 1.0)

# Public-release note.
D_MODEL = env_int("STAGE1_D_MODEL", 96)
NHEAD = env_int("STAGE1_NHEAD", 4)
NUM_LAYERS = env_int("STAGE1_NUM_LAYERS", 3)
DIM_FEEDFORWARD = env_int("STAGE1_DIM_FEEDFORWARD", 192)
DROPOUT = env_float("STAGE1_DROPOUT", 0.08)
FEATURE_MODE = env_str("STAGE1_FEATURE_MODE", "ep_only")
TOKEN_STEM = env_str("STAGE1_TOKEN_STEM", "linear")


def get_feature_dim(feature_mode: str) -> int:
    if feature_mode == "ep_only":
        return 1
    if feature_mode == "ep_slope_curvature_temp":
        return 4
    if feature_mode == "ep_edp_derivative":
        return 5
    raise ValueError(f"Unsupported STAGE1_FEATURE_MODE: {feature_mode}")


def infer_feature_mode_from_dim(feature_dim: int) -> str:
    mapping = {
        1: "ep_only",
        4: "ep_slope_curvature_temp",
        5: "ep_edp_derivative",
    }
    if feature_dim not in mapping:
        raise ValueError(f"Cannot infer feature mode from feature_dim={feature_dim}")
    return mapping[feature_dim]


FEATURE_DIM = get_feature_dim(FEATURE_MODE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Public-release note.
# =========================================================
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
            encoding = sys.stdout.encoding or "utf-8"
            sys.stdout.buffer.write((msg + "\n").encode(encoding, errors="replace"))
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def log_json(self, payload: dict):
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_log10_np(x: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(np.asarray(x, dtype=np.float32), a_min=1e-8, a_max=None))


def build_input_feature_curve(
    ep_standard: np.ndarray,
    edp_standard: Optional[np.ndarray],
    T_standard: np.ndarray,
    feature_mode: Optional[str] = None,
) -> np.ndarray:
    feature_mode = feature_mode or FEATURE_MODE
    ep_standard = np.asarray(ep_standard, dtype=np.float32).reshape(-1)
    T_standard = np.asarray(T_standard, dtype=np.float32).reshape(-1)

    if feature_mode == "ep_only":
        return ep_standard

    slope = np.gradient(ep_standard, T_standard).astype(np.float32) * 20.0
    t_norm = ((T_standard - T_standard[0]) / max(float(T_standard[-1] - T_standard[0]), 1e-6)).astype(np.float32)

    if feature_mode == "ep_slope_curvature_temp":
        curvature = np.gradient(slope, T_standard).astype(np.float32) * 120.0
        return np.stack([ep_standard, slope, curvature, t_norm], axis=-1).astype(np.float32)

    if feature_mode == "ep_edp_derivative":
        if edp_standard is None:
            raise ValueError("E'' feature mode requires edp_standard")
        edp_standard = np.asarray(edp_standard, dtype=np.float32).reshape(-1)
        edp_slope = np.gradient(edp_standard, T_standard).astype(np.float32) * 20.0
        return np.stack([ep_standard, edp_standard, slope, edp_slope, t_norm], axis=-1).astype(np.float32)

    raise ValueError(f"Unsupported STAGE1_FEATURE_MODE: {feature_mode}")


def weighted_mean(values: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    values = values.reshape(-1)
    if sample_weights is None:
        return values.mean()
    weights = sample_weights.reshape(-1).to(device=values.device, dtype=values.dtype)
    return torch.sum(values * weights) / (torch.sum(weights) + 1e-8)


def weighted_point_mse(pred: torch.Tensor, target: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    per_sample = ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=1)
    return weighted_mean(per_sample, sample_weights)


def choose_closest_freq(freqs: List[str], target_hz: float) -> Optional[str]:
    if len(freqs) == 0:
        return None
    vals = [float(f.replace("_", ".")) for f in freqs]
    idx = int(np.argmin([abs(v - target_hz) for v in vals]))
    return freqs[idx]


def choose_anchor_freqs(freqs: List[str], target_base_freq_hz: float, extra_anchor_hz: List[float]) -> List[str]:
    chosen = []
    base_freq = choose_closest_freq(freqs, target_base_freq_hz)
    if base_freq is not None:
        chosen.append(base_freq)

    extra_candidates = []
    for hz in extra_anchor_hz:
        freq = choose_closest_freq(freqs, hz)
        if freq is not None and freq not in chosen and freq not in extra_candidates:
            extra_candidates.append(freq)

    if EXTRA_ANCHOR_MODE == "random_one":
        if extra_candidates:
            chosen.append(random.choice(extra_candidates))
        return chosen

    if EXTRA_ANCHOR_MODE == "random_two":
        if extra_candidates:
            n_pick = min(2, len(extra_candidates))
            chosen.extend(random.sample(extra_candidates, n_pick))
        return chosen

    for freq in extra_candidates:
        chosen.append(freq)
    return chosen


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


def make_feat_curve(npz_data, freq_str: str, T_standard: np.ndarray, feature_mode: Optional[str] = None) -> Optional[torch.Tensor]:
    feature_mode = feature_mode or FEATURE_MODE
    try:
        T_arr = np.asarray(npz_data[f"E_prime_temp_{freq_str}Hz"], dtype=np.float32)
        Ep_arr = np.asarray(npz_data[f"E_prime_val_{freq_str}Hz"], dtype=np.float32)
        Edp_arr = None
        if f"E_double_prime_val_{freq_str}Hz" in npz_data.files:
            Edp_arr = np.asarray(npz_data[f"E_double_prime_val_{freq_str}Hz"], dtype=np.float32)

        order = np.argsort(T_arr)
        T_arr = T_arr[order]
        Ep_arr = Ep_arr[order]
        if Edp_arr is not None:
            Edp_arr = Edp_arr[order]

        T_arr, unique_indices = np.unique(T_arr, return_index=True)
        Ep_arr = Ep_arr[unique_indices]
        if Edp_arr is not None:
            Edp_arr = Edp_arr[unique_indices]

        if len(T_arr) < 2:
            return None

        interp_func = interp1d(T_arr, Ep_arr, kind='linear', bounds_error=False, fill_value="extrapolate")
        Ep_standard = safe_log10_np(interp_func(T_standard))
        Edp_standard = None
        if feature_mode == "ep_edp_derivative":
            if Edp_arr is None:
                return None
            interp_edp = interp1d(T_arr, Edp_arr, kind='linear', bounds_error=False, fill_value="extrapolate")
            Edp_standard = safe_log10_np(interp_edp(T_standard))
        feat_curve = build_input_feature_curve(Ep_standard, Edp_standard, T_standard, feature_mode=feature_mode)
        return torch.tensor(feat_curve, dtype=torch.float32)
    except Exception:
        return None


def to_repo_relative(path_str: str) -> str:
    path = Path(path_str).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path.resolve())
    return str((REPO_ROOT / path).resolve())


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


def load_split_info(load_path=SPLIT_JSON):
    load_path = Path(load_path)
    if not load_path.exists():
        return None

    with open(load_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out = {}
    for key in ("train_files", "val_files", "test_files"):
        out[key] = [resolve_repo_path(p) for p in payload.get(key, [])]
    return out


# =========================================================
# Public-release note.
# =========================================================
class PolymerDMADataset(Dataset):
    def __init__(self, file_paths: List[str], logger: FileLogger,
                 T_grid_points: int = 100, target_base_freq_hz: float = 1.0):
        super().__init__()
        self.samples = []
        self.T_standard = np.linspace(20, 180, T_grid_points)
        self.sample_meta = []
        self.skip_info = []
        rng = random.Random(SEED)

        logger.log(f"Processing {len(file_paths)} files...")

        for file_path in file_paths:
            sample_name = os.path.splitext(os.path.basename(file_path))[0]
            try:
                data = np.load(file_path, allow_pickle=True)
                freqs = load_npz_freqs(data)
                if not freqs:
                    self.skip_info.append({"file": file_path, "reason": "no_valid_freqs"})
                    continue

                anchor_freqs = choose_anchor_freqs(freqs, target_base_freq_hz, EXTRA_ANCHOR_HZ)
                n_before = len(self.samples)

                for base_freq in anchor_freqs:
                    anchor_before = len(self.samples)
                    base_feat_curve = make_feat_curve(data, base_freq, self.T_standard)
                    if base_feat_curve is None:
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
                            ep_log = float(np.log10(max(Ep_arr[i], 1e-8)))
                            edp_log = float(np.log10(max(Edp_arr[i], 1e-8)))
                            t_norm = (t_val - 20.0) / 160.0

                            main_aux_freq = rng.choice(main_aux_keys) if main_aux_keys else None
                            main_aux_feat = main_aux_map[main_aux_freq] if main_aux_freq is not None else None

                            use_focus10 = (focus10_feat is not None) and (rng.random() < FOCUS10_AUX_PROB)

                            self.samples.append({
                                "feat_curve": base_feat_curve,
                                "omega_feat": torch.tensor([base_f_norm], dtype=torch.float32),

                                "T_target": torch.tensor([t_norm], dtype=torch.float32),
                                "omega_target": torch.tensor([f_norm], dtype=torch.float32),

                                "target_ep": torch.tensor([ep_log], dtype=torch.float32),
                                "target_edp": torch.tensor([edp_log], dtype=torch.float32),

                                "main_aux_feat_curve": main_aux_feat,
                                "main_aux_omega_feat": (
                                    torch.tensor([np.log10(max(float(main_aux_freq.replace("_", ".")), 0.1))], dtype=torch.float32)
                                    if main_aux_freq is not None else torch.tensor([0.0], dtype=torch.float32)
                                ),
                                "has_main_aux": 1 if main_aux_feat is not None else 0,

                                "focus10_feat_curve": focus10_feat if use_focus10 else None,
                                "focus10_omega_feat": (
                                    torch.tensor([np.log10(max(float(focus10_freq.replace("_", ".")), 0.1))], dtype=torch.float32)
                                    if (use_focus10 and focus10_freq is not None) else torch.tensor([0.0], dtype=torch.float32)
                                ),
                                "has_focus10": 1 if (use_focus10 and focus10_feat is not None) else 0,
                                "sample_weight": HARD_SAMPLE_BOOST if sample_name in HARD_SAMPLE_IDS else 1.0,
                                "is_hard_sample": 1 if sample_name in HARD_SAMPLE_IDS else 0,
                            })

                    anchor_added = len(self.samples) - anchor_before
                    if anchor_added > 0:
                        self.sample_meta.append({
                            "sample_name": sample_name,
                            "file_path": file_path,
                            "base_freq": base_freq,
                            "n_points": anchor_added,
                            "available_freqs": freqs,
                            "main_aux_freqs": list(main_aux_map.keys()),
                            "focus10_freq": focus10_freq,
                        })

                added = len(self.samples) - n_before
                if added == 0:
                    self.skip_info.append({"file": file_path, "reason": "no_points_added"})
                    continue

            except Exception as e:
                self.skip_info.append({"file": file_path, "reason": f"{type(e).__name__}: {e}"})

        logger.log(f"Dataset construction complete: {len(self.samples)} point samples from {len(self.sample_meta)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        main_aux_feat = s["main_aux_feat_curve"] if s["main_aux_feat_curve"] is not None else torch.zeros_like(s["feat_curve"])
        focus10_feat = s["focus10_feat_curve"] if s["focus10_feat_curve"] is not None else torch.zeros_like(s["feat_curve"])

        return (
            s["feat_curve"],
            s["omega_feat"],
            s["T_target"],
            s["omega_target"],
            s["target_ep"],
            s["target_edp"],

            main_aux_feat,
            s["main_aux_omega_feat"],
            torch.tensor([s["has_main_aux"]], dtype=torch.float32),

            focus10_feat,
            s["focus10_omega_feat"],
            torch.tensor([s["has_focus10"]], dtype=torch.float32),
            torch.tensor([s["sample_weight"]], dtype=torch.float32),
            torch.tensor([s["is_hard_sample"]], dtype=torch.float32),
        )


# =========================================================
# Public-release note.
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class CurveTransformerEncoder(nn.Module):
    def __init__(
        self,
        seq_len=100,
        d_model=96,
        nhead=4,
        num_layers=3,
        dim_feedforward=192,
        dropout=0.08,
        feature_dim: int = 1,
        token_stem: str = "linear",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.token_stem = token_stem
        self.feature_norm = nn.LayerNorm(feature_dim) if feature_dim > 1 else nn.Identity()

        if self.token_stem == "conv":
            stem_hidden = max(d_model // 2, feature_dim * 8)
            self.input_proj = nn.Sequential(
                nn.Conv1d(feature_dim, stem_hidden, kernel_size=5, padding=2, bias=False),
                nn.GELU(),
                nn.Conv1d(stem_hidden, d_model, kernel_size=3, padding=1, bias=False),
                nn.GELU(),
            )
        else:
            self.input_proj = nn.Linear(feature_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L] or [B, L, C]
        B = x.size(0)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.feature_norm(x)

        if self.token_stem == "conv":
            x = x.transpose(1, 2)
            x = self.input_proj(x)
            x = x.transpose(1, 2)
        else:
            x = self.input_proj(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.norm(x)
        global_feat = x[:, 0, :]
        token_feat = x[:, 1:, :]
        return global_feat, token_feat


# =========================================================
# Public-release note.
# =========================================================
class ParameterHead(nn.Module):
    """
    Public-release English description.
    """
    def __init__(self, in_dim: int, num_maxwell: int):
        super().__init__()
        self.num_maxwell = num_maxwell

        self.shared = nn.Sequential(
            nn.Linear(in_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(192, 128),
            nn.GELU()
        )

        self.head_wlf = nn.Linear(128, 2)
        self.head_ee = nn.Linear(128, 1)
        self.head_ei = nn.Linear(128, num_maxwell)
        self.head_tau = nn.Linear(128, num_maxwell)

    def forward(self, x):
        h = self.shared(x)
        out_wlf = self.head_wlf(h)
        out_ee = self.head_ee(h)
        out_ei = self.head_ei(h)
        out_tau = self.head_tau(h)
        return out_wlf, out_ee, out_ei, out_tau


class PINNPolymerTransformer(nn.Module):
    def __init__(
        self,
        seq_len=100,
        num_maxwell=13,
        d_model: Optional[int] = None,
        nhead: Optional[int] = None,
        num_layers: Optional[int] = None,
        dim_feedforward: Optional[int] = None,
        dropout: Optional[float] = None,
        feature_dim: Optional[int] = None,
        token_stem: Optional[str] = None,
    ):
        super().__init__()
        self.num_maxwell = num_maxwell
        self.Tr = 100.0
        self.feature_dim = feature_dim or FEATURE_DIM
        self.token_stem = token_stem or TOKEN_STEM
        self.d_model = d_model or D_MODEL
        self.nhead = nhead or NHEAD
        self.num_layers = num_layers or NUM_LAYERS
        self.dim_feedforward = dim_feedforward or DIM_FEEDFORWARD
        self.dropout = DROPOUT if dropout is None else dropout

        self.encoder = CurveTransformerEncoder(
            seq_len=seq_len,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            feature_dim=self.feature_dim,
            token_stem=self.token_stem,
        )

        self.param_head = ParameterHead(self.d_model + 1, num_maxwell)

    def infer_params(self, feat_curve, omega_feat) -> Dict[str, torch.Tensor]:
        feat_curve = feat_curve.float()
        omega_feat = omega_feat.float()

        global_feat, token_feat = self.encoder(feat_curve)
        x = torch.cat([global_feat, omega_feat], dim=1)

        out_wlf, out_ee, out_ei, out_tau = self.param_head(x)

        # ---------- WLF ----------
        # Public-release note.
        C1 = 5.0 + 25.0 * torch.sigmoid(out_wlf[:, 0:1])     # [5, 30]
        C2 = 50.0 + 250.0 * torch.sigmoid(out_wlf[:, 1:2])   # [50, 300]

        # ---------- Ee ----------
        E_e_log = 1.0 + 9.0 * torch.sigmoid(out_ee)          # [1, 10]
        E_e = 10.0 ** E_e_log

        # ---------- Ei ----------
        E_i_log = 0.5 + 9.5 * torch.sigmoid(out_ei)          # [0.5, 10]
        E_i = 10.0 ** E_i_log

        # ---------- tau ----------
        # Public-release note.
        tau_base_log = -15.0 + 10.0 * torch.sigmoid(out_tau[:, 0:1])    # [-15, -5]
        tau_delta_log = 0.08 + 1.8 * torch.sigmoid(out_tau[:, 1:])  # Positive increment
        tau_log_increments = torch.cat([tau_base_log, tau_delta_log], dim=1)
        tau_log = torch.cumsum(tau_log_increments, dim=1)
        tau_i = 10.0 ** tau_log

        return {
            "C1": C1,
            "C2": C2,
            "E_e_log": E_e_log,
            "E_e": E_e,
            "E_i_log": E_i_log,
            "E_i": E_i,
            "tau_log": tau_log,
            "tau_i": tau_i,
            "global_feat": global_feat,
            "token_feat": token_feat,
        }

    def physics_decode(self, params, T_target, omega_target):
        C1 = params["C1"]
        C2 = params["C2"]
        E_e = params["E_e"]
        E_i = params["E_i"]
        tau_i = params["tau_i"]

        T_real = T_target * 160.0 + 20.0
        omega_real = 2.0 * math.pi * (10.0 ** omega_target)

        # Public-release note.
        log_aT = -C1 * (T_real - self.Tr) / (C2 + (T_real - self.Tr) + 1e-6)
        log_aT = torch.clamp(log_aT, min=-15.0, max=15.0)
        a_T = 10.0 ** log_aT

        omega_reduced = omega_real * a_T
        wt = torch.clamp(omega_reduced * tau_i, max=1e15)
        wt2 = wt ** 2
        denom = 1.0 + wt2

        E_prime_linear = E_e + torch.sum(E_i * (wt2 / denom), dim=1, keepdim=True)
        E_double_prime_linear = torch.sum(E_i * (wt / denom), dim=1, keepdim=True)

        pred_ep = torch.log10(E_prime_linear + 1e-8)
        pred_edp = torch.log10(E_double_prime_linear + 1e-8)
        return pred_ep, pred_edp

    def forward(self, feat_curve, omega_feat, T_target, omega_target):
        params = self.infer_params(feat_curve, omega_feat)
        pred_ep, pred_edp = self.physics_decode(params, T_target.float(), omega_target.float())
        return pred_ep, pred_edp, params


# =========================================================
# 5. probe
# =========================================================
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
def evaluate_probe_consistency(model: PINNPolymerTransformer, probe: ProbeSample,
                               device: torch.device, T_standard: np.ndarray):
    data = np.load(probe.file_path, allow_pickle=True)
    param_vecs = []
    rows = []

    for anchor_freq in probe.all_freqs:
        feat_curve = make_feat_curve(data, anchor_freq, T_standard)
        if feat_curve is None:
            continue

        feat_curve = feat_curve.unsqueeze(0).to(device)
        omega_feat = torch.tensor(
            [[np.log10(max(float(anchor_freq.replace("_", ".")), 0.1))]],
            dtype=torch.float32, device=device
        )

        params = model.infer_params(feat_curve, omega_feat)
        vec = torch.cat([params["C1"], params["C2"], params["E_e_log"], params["E_i_log"], params["tau_log"]], dim=1)
        param_vecs.append(vec.squeeze(0).detach().cpu().numpy())

        f_num = float(anchor_freq.replace("_", "."))
        T_arr = np.asarray(data[f"E_prime_temp_{anchor_freq}Hz"], dtype=np.float32)
        target_ep = safe_log10_np(np.asarray(data[f"E_prime_val_{anchor_freq}Hz"], dtype=np.float32))
        target_edp = safe_log10_np(np.asarray(data[f"E_double_prime_val_{anchor_freq}Hz"], dtype=np.float32))

        T_norm = torch.tensor(((T_arr - 20.0) / 160.0).reshape(-1, 1), dtype=torch.float32, device=device)
        omega_target = torch.full((len(T_arr), 1), np.log10(max(f_num, 0.1)), dtype=torch.float32, device=device)
        feat_repeat_dims = [len(T_arr)] + [1] * (feat_curve.dim() - 1)
        feat_rep = feat_curve.repeat(*feat_repeat_dims)
        omega_feat_rep = omega_feat.repeat(len(T_arr), 1)

        pred_ep, pred_edp, _ = model(feat_rep, omega_feat_rep, T_norm, omega_target)

        rmse_ep = float(torch.sqrt(torch.mean((pred_ep.squeeze(1) - torch.tensor(target_ep, device=device)) ** 2)).cpu())
        rmse_edp = float(torch.sqrt(torch.mean((pred_edp.squeeze(1) - torch.tensor(target_edp, device=device)) ** 2)).cpu())

        rows.append({
            "anchor_freq": anchor_freq,
            "rmse_ep_self": rmse_ep,
            "rmse_edp_self": rmse_edp,
            "C1": float(params["C1"].cpu().view(-1)[0]),
            "C2": float(params["C2"].cpu().view(-1)[0]),
            "E_e_log": float(params["E_e_log"].cpu().view(-1)[0]),
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
                if (freq_names[i] in ["10", "10_0"]) or (freq_names[j] in ["10", "10_0"]):
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


# =========================================================
# Public-release note.
# =========================================================
def get_main_aux_weight(epoch: int) -> float:
    if epoch < MAIN_AUX_START_EPOCH:
        return 0.0
    if epoch < MAIN_AUX_FULL_EPOCH:
        return MAIN_AUX_LOSS_MAX * (epoch - MAIN_AUX_START_EPOCH) / max(MAIN_AUX_FULL_EPOCH - MAIN_AUX_START_EPOCH, 1)
    return MAIN_AUX_LOSS_MAX


def get_focus10_aux_weight(epoch: int) -> float:
    if epoch < FOCUS10_START_EPOCH:
        return 0.0
    if epoch < FOCUS10_FULL_EPOCH:
        return FOCUS10_AUX_LOSS_MAX * (epoch - FOCUS10_START_EPOCH) / max(FOCUS10_FULL_EPOCH - FOCUS10_START_EPOCH, 1)
    if epoch < FOCUS10_HOLD_EPOCH:
        return FOCUS10_AUX_LOSS_MAX
    if epoch < FOCUS10_END_EPOCH:
        return FOCUS10_AUX_LOSS_MAX * max(0.0, (FOCUS10_END_EPOCH - epoch) / max(FOCUS10_END_EPOCH - FOCUS10_HOLD_EPOCH, 1))
    return 0.0


def core_anchor_consistency_loss(main_params, aux_params, sample_weights: Optional[torch.Tensor] = None):
    """
    Public-release English description.
    """
    main_core = torch.cat([main_params["C1"], main_params["C2"], main_params["E_e_log"]], dim=1)
    aux_core = torch.cat([aux_params["C1"], aux_params["C2"], aux_params["E_e_log"]], dim=1)

    scale = torch.tensor([30.0, 300.0, 10.0], device=main_core.device, dtype=main_core.dtype).view(1, 3)
    per_sample = F.smooth_l1_loss(main_core / scale, aux_core / scale, reduction="none").mean(dim=1)
    return weighted_mean(per_sample, sample_weights)


def spectrum_stat_consistency_loss(main_params, aux_params, sample_weights: Optional[torch.Tensor] = None):
    """
    Public-release English description.
    Public-release English description.
    """
    main_e_mean = main_params["E_i_log"].mean(dim=1, keepdim=True)
    aux_e_mean = aux_params["E_i_log"].mean(dim=1, keepdim=True)

    main_e_std = main_params["E_i_log"].std(dim=1, keepdim=True)
    aux_e_std = aux_params["E_i_log"].std(dim=1, keepdim=True)

    main_t_mean = main_params["tau_log"].mean(dim=1, keepdim=True)
    aux_t_mean = aux_params["tau_log"].mean(dim=1, keepdim=True)

    main_t_std = main_params["tau_log"].std(dim=1, keepdim=True)
    aux_t_std = aux_params["tau_log"].std(dim=1, keepdim=True)

    loss = (
        F.smooth_l1_loss(main_e_mean, aux_e_mean, reduction="none").mean(dim=1) +
        F.smooth_l1_loss(main_e_std, aux_e_std, reduction="none").mean(dim=1) +
        F.smooth_l1_loss(main_t_mean / 15.0, aux_t_mean / 15.0, reduction="none").mean(dim=1) +
        F.smooth_l1_loss(main_t_std / 10.0, aux_t_std / 10.0, reduction="none").mean(dim=1)
    )
    return weighted_mean(loss, sample_weights)


def normalized_param_vector(params):
    return torch.cat(
        [
            params["C1"] / 30.0,
            params["C2"] / 300.0,
            params["E_e_log"] / 10.0,
            params["E_i_log"] / 10.0,
            params["tau_log"] / 15.0,
        ],
        dim=1,
    )


def normalized_spectrum_vector(params):
    return torch.cat(
        [
            params["E_i_log"] / 10.0,
            params["tau_log"] / 15.0,
        ],
        dim=1,
    )


def relative_param_consistency_loss(main_params, aux_params, sample_weights: Optional[torch.Tensor] = None):
    main_vec = normalized_param_vector(main_params)
    aux_vec = normalized_param_vector(aux_params)
    diff_norm = torch.sqrt(torch.sum((main_vec - aux_vec) ** 2, dim=1) + 1e-8)
    ref_norm = torch.sqrt(torch.sum(main_vec ** 2, dim=1) + 1e-8)
    return weighted_mean(diff_norm / (ref_norm + 1e-6), sample_weights)


def relative_spectrum_consistency_loss(main_params, aux_params, sample_weights: Optional[torch.Tensor] = None):
    main_vec = normalized_spectrum_vector(main_params)
    aux_vec = normalized_spectrum_vector(aux_params)
    diff_norm = torch.sqrt(torch.sum((main_vec - aux_vec) ** 2, dim=1) + 1e-8)
    ref_norm = torch.sqrt(torch.sum(main_vec ** 2, dim=1) + 1e-8)
    return weighted_mean(diff_norm / (ref_norm + 1e-6), sample_weights)


def focus10_consistency_loss(main_params, aux_params, sample_weights: Optional[torch.Tensor] = None):
    """
    Public-release English description.
    """
    main_core = torch.cat([main_params["C1"], main_params["C2"], main_params["E_e_log"]], dim=1)
    aux_core = torch.cat([aux_params["C1"], aux_params["C2"], aux_params["E_e_log"]], dim=1)
    scale = torch.tensor([30.0, 300.0, 10.0], device=main_core.device, dtype=main_core.dtype).view(1, 3)

    diff = torch.abs(main_core / scale - aux_core / scale)
    weights = torch.tensor([0.8, 1.5, 0.35], device=main_core.device, dtype=main_core.dtype).view(1, 3)
    per_sample = (diff * weights).mean(dim=1)
    return weighted_mean(per_sample, sample_weights)


def tau_smooth_prior(params):
    """
    Public-release English description.
    """
    tau_log = params["tau_log"]
    d = tau_log[:, 1:] - tau_log[:, :-1]
    target_mid = 0.9
    return ((d - target_mid) ** 2).mean()


def spectrum_shape_prior(params):
    """
    Public-release English description.
    """
    e_log = params["E_i_log"]
    tau_log = params["tau_log"]

    if e_log.size(1) < 3:
        return torch.tensor(0.0, device=e_log.device, dtype=e_log.dtype)

    d2_e = e_log[:, 2:] - 2.0 * e_log[:, 1:-1] + e_log[:, :-2]
    d2_tau = tau_log[:, 2:] - 2.0 * tau_log[:, 1:-1] + tau_log[:, :-2]
    return (d2_e ** 2).mean() + 0.35 * (d2_tau ** 2).mean()


def infer_transformer_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, object]:
    if "encoder.cls_token" not in state_dict:
        raise KeyError("checkpoint is missing encoder.cls_token")

    d_model = int(state_dict["encoder.cls_token"].shape[-1])
    num_layers = len(
        {
            key.split(".")[3]
            for key in state_dict.keys()
            if key.startswith("encoder.transformer.layers.") and key.count(".") >= 4
        }
    )
    dim_feedforward = int(state_dict["encoder.transformer.layers.0.linear1.weight"].shape[0])
    dropout = DROPOUT

    if "encoder.input_proj.weight" in state_dict:
        token_stem = "linear"
        feature_dim = int(state_dict["encoder.input_proj.weight"].shape[1])
    elif "encoder.input_proj.0.weight" in state_dict:
        token_stem = "conv"
        feature_dim = int(state_dict["encoder.input_proj.0.weight"].shape[1])
    else:
        raise KeyError("checkpoint is missing encoder.input_proj weights")

    return {
        "d_model": d_model,
        "nhead": NHEAD,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "feature_dim": feature_dim,
        "feature_mode": infer_feature_mode_from_dim(feature_dim),
        "token_stem": token_stem,
    }


# =========================================================
# Public-release note.
# =========================================================
def train_model():
    set_seed(SEED)
    logger = FileLogger(LOG_TXT_PATH, LOG_JSONL_PATH)

    logger.log("========== Stage-1 Transformer: Stable Parameter Inversion ==========")
    logger.log(f"device = {DEVICE}")
    logger.log(f"run_tag = {RUN_TAG or 'default'}")
    logger.log(f"num_maxwell = {NUM_MAXWELL}")
    logger.log(f"transformer = d_model:{D_MODEL}, nhead:{NHEAD}, layers:{NUM_LAYERS}")
    logger.log(f"feature_mode = {FEATURE_MODE}, feature_dim = {FEATURE_DIM}, token_stem = {TOKEN_STEM}")
    logger.log(f"extra_anchor_hz = {EXTRA_ANCHOR_HZ or 'none'} | extra_anchor_mode = {EXTRA_ANCHOR_MODE}")
    logger.log(
        f"main_aux_prob={MAIN_AUX_PROB:.2f}, main_aux_max={MAIN_AUX_LOSS_MAX:.3f}, "
        f"focus10_prob={FOCUS10_AUX_PROB:.2f}, focus10_max={FOCUS10_AUX_LOSS_MAX:.3f}"
    )
    logger.log(
        f"main_aux_schedule=({MAIN_AUX_START_EPOCH}->{MAIN_AUX_FULL_EPOCH}), "
        f"focus10_schedule=({FOCUS10_START_EPOCH}->{FOCUS10_FULL_EPOCH}->{FOCUS10_HOLD_EPOCH}->{FOCUS10_END_EPOCH})"
    )
    logger.log(
        f"main_param_vec_w={MAIN_PARAM_VEC_CONS_WEIGHT:.3f}, "
        f"focus10_param_vec_w={FOCUS10_PARAM_VEC_CONS_WEIGHT:.3f}, "
        f"main_spectrum_vec_w={MAIN_SPECTRUM_VEC_CONS_WEIGHT:.3f}, "
        f"focus10_spectrum_vec_w={FOCUS10_SPECTRUM_VEC_CONS_WEIGHT:.3f}, "
        f"spectrum_shape_prior_w={SPECTRUM_SHAPE_PRIOR_WEIGHT:.4f}"
    )
    logger.log(
        f"hard_samples={len(HARD_SAMPLE_IDS)} | hard_sample_boost={HARD_SAMPLE_BOOST:.2f} | "
        f"hard_main_cons_mult={HARD_MAIN_CONS_MULT:.2f} | hard_focus10_cons_mult={HARD_FOCUS10_CONS_MULT:.2f}"
    )
    if HARD_SAMPLE_IDS:
        preview = sorted(HARD_SAMPLE_IDS)[:12]
        logger.log(f"hard_sample_preview = {preview}{' ...' if len(HARD_SAMPLE_IDS) > 12 else ''}")
    logger.log(
        f"tradeoff_weights: val={TRADEOFF_W_VAL:.2f}, cv={TRADEOFF_W_CV:.2f}, "
        f"rel={TRADEOFF_W_REL:.2f}, rel10={TRADEOFF_W_REL10:.2f}, rmse10={TRADEOFF_W_RMSE10:.2f}, "
        f"best_cons_start={BEST_CONS_START_EPOCH}"
    )
    logger.log(f"probe_split = {PROBE_SPLIT_NAME}, probe_every = {PROBE_EVERY}, probe_count = {PROBE_COUNT}")
    logger.log(f"save_epoch_checkpoints = {SAVE_EPOCH_CHECKPOINTS}")
    logger.log(
        f"epochs={EPOCHS}, batch_train={BATCH_SIZE_TRAIN}, batch_val={BATCH_SIZE_VAL}, "
        f"lr={LR:.2e}, wd={WEIGHT_DECAY:.2e}, max_files={MAX_FILES or 'all'}"
    )

    all_files = sorted(glob.glob(os.path.join(str(DATA_DIR), "*.npz")))
    if not all_files:
        logger.log("Status updated.")
        return
    existing_split = load_split_info(SPLIT_JSON) if USE_EXISTING_SPLIT and MAX_FILES <= 0 else None
    if existing_split is not None:
        train_files = [p for p in existing_split["train_files"] if os.path.exists(p)]
        val_files = [p for p in existing_split["val_files"] if os.path.exists(p)]
        test_files = [p for p in existing_split["test_files"] if os.path.exists(p)]
        logger.log(f"Using existing split: {SPLIT_JSON}")
    else:
        if MAX_FILES > 0:
            all_files = all_files[:MAX_FILES]

        rng = np.random.default_rng(SEED)
        rng.shuffle(all_files)
        n = len(all_files)

        train_files = all_files[:int(0.8 * n)]
        val_files = all_files[int(0.8 * n):int(0.9 * n)]
        test_files = all_files[int(0.9 * n):]

        if MAX_FILES <= 0:
            save_split_info(train_files, val_files, test_files)
        else:
            temp_split_path = with_run_tag(RESULTS_DIR / "logs" / "split_transformer_stage1_tmp.json", RUN_TAG)
            save_split_info(train_files, val_files, test_files, save_path=temp_split_path)
            logger.log(f"Saved temporary split: {temp_split_path}")

    logger.log(f"Data split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    logger.log("Status updated.")
    train_dataset = PolymerDMADataset(train_files, logger, T_grid_points=T_GRID_POINTS, target_base_freq_hz=TARGET_BASE_FREQ_HZ)

    logger.log("Status updated.")
    val_dataset = PolymerDMADataset(val_files, logger, T_grid_points=T_GRID_POINTS, target_base_freq_hz=TARGET_BASE_FREQ_HZ)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.log("Status updated.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())

    model = PINNPolymerTransformer(seq_len=T_GRID_POINTS, num_maxwell=NUM_MAXWELL).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    mse = nn.MSELoss()

    if PROBE_SPLIT_NAME == "train":
        probe_source_files = train_files
    elif PROBE_SPLIT_NAME == "test":
        probe_source_files = test_files
    else:
        probe_source_files = val_files

    probe_samples = choose_probe_samples(probe_source_files, n_probe=PROBE_COUNT)
    if probe_samples:
        logger.log("Fixed probe samples:")
        for p in probe_samples:
            logger.log(f"   - {p.sample_name}: {p.all_freqs}")

    best_val_loss = float("inf")
    best_tradeoff = float("inf")
    best_fit_epoch = -1
    best_cons_epoch = -1

    history = {
        "train_total": [], "train_ep": [], "train_edp": [],
        "train_main_cons": [], "train_focus10_cons": [], "train_tau_prior": [], "train_spectrum_prior": [],
        "val_total": [], "val_ep": [], "val_edp": [],
        "probe_cv_mean": [], "probe_rel_l2_mean": [], "probe_rel10_mean": [], "probe_rmse10_mean": [],
        "tradeoff": [], "main_w": [], "focus10_w": [], "lr": []
    }

    logger.log("Status updated.")

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()

        main_w = get_main_aux_weight(epoch)
        focus10_w = get_focus10_aux_weight(epoch)

        sum_total = 0.0
        sum_ep = 0.0
        sum_edp = 0.0
        sum_main_cons = 0.0
        sum_focus10 = 0.0
        sum_tau_prior = 0.0
        sum_spectrum_prior = 0.0

        for (
            feat_curve, omega_feat, T_target, omega_target, target_ep, target_edp,
            main_aux_feat_curve, main_aux_omega_feat, has_main_aux,
            focus10_feat_curve, focus10_omega_feat, has_focus10,
            sample_weight, is_hard_sample
        ) in train_loader:

            feat_curve = feat_curve.to(DEVICE, non_blocking=True)
            omega_feat = omega_feat.to(DEVICE, non_blocking=True)
            T_target = T_target.to(DEVICE, non_blocking=True)
            omega_target = omega_target.to(DEVICE, non_blocking=True)
            target_ep = target_ep.to(DEVICE, non_blocking=True)
            target_edp = target_edp.to(DEVICE, non_blocking=True)

            main_aux_feat_curve = main_aux_feat_curve.to(DEVICE, non_blocking=True)
            main_aux_omega_feat = main_aux_omega_feat.to(DEVICE, non_blocking=True)
            has_main_aux = has_main_aux.to(DEVICE, non_blocking=True)

            focus10_feat_curve = focus10_feat_curve.to(DEVICE, non_blocking=True)
            focus10_omega_feat = focus10_omega_feat.to(DEVICE, non_blocking=True)
            has_focus10 = has_focus10.to(DEVICE, non_blocking=True)
            sample_weight = sample_weight.to(DEVICE, non_blocking=True).squeeze(1)
            is_hard_sample = is_hard_sample.to(DEVICE, non_blocking=True).squeeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                pred_ep, pred_edp, params_main = model(feat_curve, omega_feat, T_target, omega_target)

                loss_ep = weighted_point_mse(pred_ep, target_ep, sample_weight)
                loss_edp = weighted_point_mse(pred_edp, target_edp, sample_weight)
                loss_fit = loss_ep + loss_edp

                loss_main_cons = torch.tensor(0.0, device=DEVICE)
                loss_focus10 = torch.tensor(0.0, device=DEVICE)
                loss_tau_prior = 0.002 * tau_smooth_prior(params_main)
                loss_spectrum_prior = SPECTRUM_SHAPE_PRIOR_WEIGHT * spectrum_shape_prior(params_main)

                loss = loss_fit + loss_tau_prior + loss_spectrum_prior

                # Public-release note.
                if main_w > 0:
                    use_mask = (has_main_aux.squeeze(1) > 0.5) & (torch.rand(len(has_main_aux), device=DEVICE) < MAIN_AUX_PROB)
                    if torch.any(use_mask):
                        aux_params = model.infer_params(main_aux_feat_curve[use_mask], main_aux_omega_feat[use_mask])
                        main_cons_weights = sample_weight[use_mask]
                        if HARD_MAIN_CONS_MULT != 1.0:
                            main_cons_weights = main_cons_weights * torch.where(
                                is_hard_sample[use_mask] > 0.5,
                                torch.full_like(main_cons_weights, HARD_MAIN_CONS_MULT),
                                torch.ones_like(main_cons_weights),
                            )

                        main_sub = {k: v[use_mask] if isinstance(v, torch.Tensor) and v.shape[0] == feat_curve.shape[0] else v
                                    for k, v in params_main.items()}

                        loss_main_cons = (
                            core_anchor_consistency_loss(main_sub, aux_params, main_cons_weights)
                            + 0.25 * spectrum_stat_consistency_loss(main_sub, aux_params, main_cons_weights)
                        )
                        if MAIN_PARAM_VEC_CONS_WEIGHT > 0:
                            loss_main_cons = (
                                loss_main_cons
                                + MAIN_PARAM_VEC_CONS_WEIGHT * relative_param_consistency_loss(main_sub, aux_params, main_cons_weights)
                            )
                        if MAIN_SPECTRUM_VEC_CONS_WEIGHT > 0:
                            loss_main_cons = (
                                loss_main_cons
                                + MAIN_SPECTRUM_VEC_CONS_WEIGHT * relative_spectrum_consistency_loss(main_sub, aux_params, main_cons_weights)
                            )
                        loss = loss + main_w * loss_main_cons

                # Public-release note.
                if focus10_w > 0:
                    use_mask10 = (has_focus10.squeeze(1) > 0.5)
                    if torch.any(use_mask10):
                        aux_params10 = model.infer_params(focus10_feat_curve[use_mask10], focus10_omega_feat[use_mask10])
                        focus10_cons_weights = sample_weight[use_mask10]
                        if HARD_FOCUS10_CONS_MULT != 1.0:
                            focus10_cons_weights = focus10_cons_weights * torch.where(
                                is_hard_sample[use_mask10] > 0.5,
                                torch.full_like(focus10_cons_weights, HARD_FOCUS10_CONS_MULT),
                                torch.ones_like(focus10_cons_weights),
                            )

                        main_sub10 = {k: v[use_mask10] if isinstance(v, torch.Tensor) and v.shape[0] == feat_curve.shape[0] else v
                                      for k, v in params_main.items()}

                        loss_focus10 = focus10_consistency_loss(main_sub10, aux_params10, focus10_cons_weights)
                        if FOCUS10_PARAM_VEC_CONS_WEIGHT > 0:
                            loss_focus10 = (
                                loss_focus10
                                + FOCUS10_PARAM_VEC_CONS_WEIGHT * relative_param_consistency_loss(main_sub10, aux_params10, focus10_cons_weights)
                            )
                        if FOCUS10_SPECTRUM_VEC_CONS_WEIGHT > 0:
                            loss_focus10 = (
                                loss_focus10
                                + FOCUS10_SPECTRUM_VEC_CONS_WEIGHT * relative_spectrum_consistency_loss(
                                    main_sub10, aux_params10, focus10_cons_weights
                                )
                            )
                        loss = loss + focus10_w * loss_focus10

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()

            sum_total += float(loss.detach().cpu())
            sum_ep += float(loss_ep.detach().cpu())
            sum_edp += float(loss_edp.detach().cpu())
            sum_main_cons += float(loss_main_cons.detach().cpu())
            sum_focus10 += float(loss_focus10.detach().cpu())
            sum_tau_prior += float(loss_tau_prior.detach().cpu())
            sum_spectrum_prior += float(loss_spectrum_prior.detach().cpu())

        avg_train_total = sum_total / max(len(train_loader), 1)
        avg_train_ep = sum_ep / max(len(train_loader), 1)
        avg_train_edp = sum_edp / max(len(train_loader), 1)
        avg_train_main_cons = sum_main_cons / max(len(train_loader), 1)
        avg_train_focus10 = sum_focus10 / max(len(train_loader), 1)
        avg_train_tau_prior = sum_tau_prior / max(len(train_loader), 1)
        avg_train_spectrum_prior = sum_spectrum_prior / max(len(train_loader), 1)

        # ---------------- val ----------------
        model.eval()
        v_total = 0.0
        v_ep = 0.0
        v_edp = 0.0

        with torch.no_grad():
            for (
                feat_curve, omega_feat, T_target, omega_target, target_ep, target_edp,
                _, _, _, _, _, _, _, _
            ) in val_loader:

                feat_curve = feat_curve.to(DEVICE, non_blocking=True)
                omega_feat = omega_feat.to(DEVICE, non_blocking=True)
                T_target = T_target.to(DEVICE, non_blocking=True)
                omega_target = omega_target.to(DEVICE, non_blocking=True)
                target_ep = target_ep.to(DEVICE, non_blocking=True)
                target_edp = target_edp.to(DEVICE, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    pred_ep, pred_edp, _ = model(feat_curve, omega_feat, T_target, omega_target)
                    loss_ep = mse(pred_ep, target_ep)
                    loss_edp = mse(pred_edp, target_edp)
                    loss = loss_ep + loss_edp

                v_total += float(loss.detach().cpu())
                v_ep += float(loss_ep.detach().cpu())
                v_edp += float(loss_edp.detach().cpu())

        avg_val_total = v_total / max(len(val_loader), 1)
        avg_val_ep = v_ep / max(len(val_loader), 1)
        avg_val_edp = v_edp / max(len(val_loader), 1)

        # ---------------- probe ----------------
        probe_cv_vals = []
        probe_rel_l2_vals = []
        probe_rel10_vals = []
        probe_rmse10_vals = []
        if probe_samples:
            for probe in probe_samples:
                info = evaluate_probe_consistency(model, probe, DEVICE, train_dataset.T_standard)
                probe_cv_vals.append(info["param_cv_mean"])
                probe_rel_l2_vals.append(info["param_rel_l2_mean"])
                probe_rel10_vals.append(info["param_rel_l2_10_mean"])
                if not np.isnan(info["rmse10_edp"]):
                    probe_rmse10_vals.append(info["rmse10_edp"])

        probe_cv_mean = float(np.mean(probe_cv_vals)) if probe_cv_vals else 0.0
        probe_rel_l2_mean = float(np.mean(probe_rel_l2_vals)) if probe_rel_l2_vals else 0.0
        probe_rel10_mean = float(np.mean(probe_rel10_vals)) if probe_rel10_vals else 0.0
        probe_rmse10_mean = float(np.mean(probe_rmse10_vals)) if probe_rmse10_vals else 0.0

        tradeoff = (
            TRADEOFF_W_VAL * avg_val_total
            + TRADEOFF_W_CV * probe_cv_mean
            + TRADEOFF_W_REL * probe_rel_l2_mean
            + TRADEOFF_W_REL10 * probe_rel10_mean
            + TRADEOFF_W_RMSE10 * probe_rmse10_mean
        )

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_total"].append(avg_train_total)
        history["train_ep"].append(avg_train_ep)
        history["train_edp"].append(avg_train_edp)
        history["train_main_cons"].append(avg_train_main_cons)
        history["train_focus10_cons"].append(avg_train_focus10)
        history["train_tau_prior"].append(avg_train_tau_prior)
        history["train_spectrum_prior"].append(avg_train_spectrum_prior)
        history["val_total"].append(avg_val_total)
        history["val_ep"].append(avg_val_ep)
        history["val_edp"].append(avg_val_edp)
        history["probe_cv_mean"].append(probe_cv_mean)
        history["probe_rel_l2_mean"].append(probe_rel_l2_mean)
        history["probe_rel10_mean"].append(probe_rel10_mean)
        history["probe_rmse10_mean"].append(probe_rmse10_mean)
        history["tradeoff"].append(tradeoff)
        history["main_w"].append(main_w)
        history["focus10_w"].append(focus10_w)
        history["lr"].append(cur_lr)

        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            best_fit_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_BEST_FIT)

        if (epoch + 1) >= BEST_CONS_START_EPOCH and tradeoff < best_tradeoff:
            best_tradeoff = tradeoff
            best_cons_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_BEST_CONS)

        if SAVE_EPOCH_CHECKPOINTS:
            EPOCH_CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), EPOCH_CKPT_DIR / f"epoch_{epoch + 1:03d}.pth")

        logger.log_json({
            "epoch": epoch + 1,
            "train_total": avg_train_total,
            "train_ep": avg_train_ep,
            "train_edp": avg_train_edp,
            "train_main_cons": avg_train_main_cons,
            "train_focus10_cons": avg_train_focus10,
            "train_tau_prior": avg_train_tau_prior,
            "train_spectrum_prior": avg_train_spectrum_prior,
            "val_total": avg_val_total,
            "val_ep": avg_val_ep,
            "val_edp": avg_val_edp,
            "probe_cv_mean": probe_cv_mean,
            "probe_rel_l2_mean": probe_rel_l2_mean,
            "probe_rel10_mean": probe_rel10_mean,
            "probe_rmse10_mean": probe_rmse10_mean,
            "tradeoff": tradeoff,
            "main_w": main_w,
            "focus10_w": focus10_w,
            "lr": cur_lr,
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.log(
                f"Epoch [{epoch + 1:03d}/{EPOCHS}] | "
                f"Time {time.time() - t0:.2f}s | "
                f"Train {avg_train_total:.4f} "
                f"(Ep {avg_train_ep:.4f}, Edp {avg_train_edp:.4f}, MainCons {avg_train_main_cons:.5f}, "
                f"F10 {avg_train_focus10:.5f}, TauPrior {avg_train_tau_prior:.5f}, SpecPrior {avg_train_spectrum_prior:.5f}) | "
                f"Val {avg_val_total:.4f} (Ep {avg_val_ep:.4f}, Edp {avg_val_edp:.4f}) | "
                f"CV {probe_cv_mean:.4f} | Rel {probe_rel_l2_mean:.4f} | Rel10 {probe_rel10_mean:.4f} | RMSE10 {probe_rmse10_mean:.4f} | "
                f"BestFit {best_val_loss:.4f}@{best_fit_epoch} | BestCons {best_tradeoff:.4f}@{best_cons_epoch}"
            )

        if probe_samples and ((epoch + 1) % PROBE_EVERY == 0 or epoch == 0):
            logger.log(f"🔎 Probe @ epoch {epoch + 1}")
            for probe in probe_samples:
                try:
                    info = evaluate_probe_consistency(model, probe, DEVICE, train_dataset.T_standard)
                    logger.log(
                        f"   - {info['sample_name']} | anchors={info['n_anchors']} | "
                        f"cv={info['param_cv_mean']:.4f} | rel_l2={info['param_rel_l2_mean']:.4f} | "
                        f"rel10={info['param_rel_l2_10_mean']:.4f} | rmse10={info['rmse10_edp']:.4f}"
                    )
                except Exception as e:
                    logger.log(f"   - probe {probe.sample_name} failed: {type(e).__name__}: {e}")

        last_improve_epoch = max(best_fit_epoch, best_cons_epoch)
        if last_improve_epoch > 0 and (epoch + 1) - last_improve_epoch >= PATIENCE:
            logger.log(f"🛑 Early stop at epoch {epoch + 1}")
            break

    # Public-release note.
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_total"], label="Train Total")
    plt.plot(history["val_total"], label="Val Total")
    plt.plot(history["val_ep"], "--", label="Val E'")
    plt.plot(history["val_edp"], "--", label="Val E''")
    plt.plot(history["probe_rel10_mean"], "-.", label="Probe Rel10")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=300)

    np.savez(HISTORY_NPZ_PATH, **{k: np.asarray(v, dtype=np.float32) for k, v in history.items()})

    logger.log(f"Best-fit model saved: {MODEL_SAVE_BEST_FIT}")
    logger.log(f"Best-consistency checkpoint saved: {MODEL_SAVE_BEST_CONS}")


if __name__ == "__main__":
    train_model()
