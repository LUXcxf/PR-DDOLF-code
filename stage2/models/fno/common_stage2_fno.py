import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from torch.utils.data import Dataset


def _resolve_env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    path = Path(raw.strip())
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / path).resolve()


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1.models.transformer.train_transformer_stage1 import (
    PINNPolymerTransformer,
    choose_closest_freq,
    infer_transformer_config_from_state_dict,
    load_npz_freqs,
    make_feat_curve,
)
STAGE2_DIR = Path(__file__).resolve().parents[2]
STAGE1_DIR = REPO_ROOT / "stage1"
DATA_DIR = STAGE1_DIR / "data" / "npz_converted"
STAGE1_SPLIT_JSON = _resolve_env_path("STAGE1_SPLIT_JSON_OVERRIDE", STAGE1_DIR / "data" / "splits" / "split_stage1_transformer.json")
STAGE2_SPLIT_JSON = _resolve_env_path("STAGE2_SPLIT_JSON_OVERRIDE", STAGE2_DIR / "data" / "splits" / "split_stage2_fno.json")
DEFAULT_STAGE1_MODEL = (
    STAGE1_DIR / "results" / "transformer" / "checkpoints" / "transformer_stage1_best_consistency_anchormix2edp_clean_v1.pth"
)

FREQ_GRID = [1.0, 2.0, 5.0, 10.0]
T_GRID_POINTS = 100
T_STANDARD = np.linspace(20.0, 180.0, T_GRID_POINTS, dtype=np.float32)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value is not None else default


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def ensure_stage2_split():
    STAGE2_SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    if STAGE2_SPLIT_JSON.exists():
        return
    if not STAGE1_SPLIT_JSON.exists():
        raise FileNotFoundError(f"missing stage1 split: {STAGE1_SPLIT_JSON}")
    payload = json.loads(STAGE1_SPLIT_JSON.read_text(encoding="utf-8"))
    STAGE2_SPLIT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_stage2_split() -> Dict[str, List[str]]:
    ensure_stage2_split()
    payload = json.loads(STAGE2_SPLIT_JSON.read_text(encoding="utf-8"))
    out = {}
    for key in ("train_files", "val_files", "test_files"):
        out[key] = [str(resolve_repo_path(p).resolve()) for p in payload.get(key, []) if resolve_repo_path(p).exists()]
    return out


def safe_log10_np(x: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(np.asarray(x, dtype=np.float32), a_min=1e-8, a_max=None))


def trimmed_valid_line_np(mask_1d: np.ndarray, trim_ratio: float) -> np.ndarray:
    mask_1d = np.asarray(mask_1d, dtype=np.float32)
    if trim_ratio <= 0.0:
        return mask_1d.copy()
    valid_idx = np.where(mask_1d > 0.5)[0]
    out = np.zeros_like(mask_1d, dtype=np.float32)
    if len(valid_idx) == 0:
        return out
    trim_n = int(np.ceil(len(valid_idx) * trim_ratio))
    max_trim = max((len(valid_idx) - 1) // 2, 0)
    trim_n = min(trim_n, max_trim)
    keep_idx = valid_idx[trim_n : len(valid_idx) - trim_n] if trim_n > 0 else valid_idx
    if len(keep_idx) == 0:
        keep_idx = valid_idx[len(valid_idx) // 2 : len(valid_idx) // 2 + 1]
    out[keep_idx] = 1.0
    return out


def build_trimmed_valid_mask(valid_mask: torch.Tensor, trim_ratio: float) -> torch.Tensor:
    if trim_ratio <= 0.0:
        return valid_mask.clone()
    out = torch.zeros_like(valid_mask)
    if valid_mask.dim() == 4:
        for b_idx in range(valid_mask.size(0)):
            for c_idx in range(valid_mask.size(1)):
                for f_idx in range(valid_mask.size(2)):
                    trimmed = trimmed_valid_line_np(valid_mask[b_idx, c_idx, f_idx].detach().cpu().numpy(), trim_ratio)
                    out[b_idx, c_idx, f_idx] = torch.from_numpy(trimmed).to(valid_mask.device, dtype=valid_mask.dtype)
        return out
    if valid_mask.dim() == 3:
        for c_idx in range(valid_mask.size(0)):
            for f_idx in range(valid_mask.size(1)):
                trimmed = trimmed_valid_line_np(valid_mask[c_idx, f_idx].detach().cpu().numpy(), trim_ratio)
                out[c_idx, f_idx] = torch.from_numpy(trimmed).to(valid_mask.device, dtype=valid_mask.dtype)
        return out
    raise ValueError(f"unsupported valid_mask dim: {valid_mask.dim()}")


def build_holdout_edge_mask(valid_mask: torch.Tensor, trim_ratio: float) -> torch.Tensor:
    if trim_ratio <= 0.0:
        return torch.zeros_like(valid_mask)
    trimmed = build_trimmed_valid_mask(valid_mask, trim_ratio)
    return torch.clamp(valid_mask - trimmed, min=0.0, max=1.0)


def interp_log_curve_with_mask(npz_data, value_key: str, temp_keys: List[str], t_standard: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    temps = None
    for temp_key in temp_keys:
        if temp_key in npz_data.files:
            temps = np.asarray(npz_data[temp_key], dtype=np.float32).reshape(-1)
            break
    if temps is None:
        raise KeyError(f"missing temperature array for {value_key}")

    values = np.asarray(npz_data[value_key], dtype=np.float32).reshape(-1)
    order = np.argsort(temps)
    temps = temps[order]
    values = values[order]
    temps, unique_idx = np.unique(temps, return_index=True)
    values = values[unique_idx]

    interp_func = interp1d(temps, values, kind="linear", bounds_error=False, fill_value="extrapolate")
    curve = safe_log10_np(interp_func(t_standard))
    mask = ((t_standard >= float(np.min(temps))) & (t_standard <= float(np.max(temps)))).astype(np.float32)
    return curve, mask


def normalized_param_vector_from_params(params: Dict[str, torch.Tensor]) -> torch.Tensor:
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


@dataclass
class Stage2Sample:
    sample_id: str
    anchor_freq: str
    target_freqs: List[str]
    material_vec: torch.Tensor
    grid_input: torch.Tensor
    base_grid: torch.Tensor
    target_grid: torch.Tensor
    residual_grid: torch.Tensor
    valid_mask: torch.Tensor


class Stage1Backbone:
    def __init__(self, model_path: Path, device: torch.device):
        if not model_path.exists():
            raise FileNotFoundError(f"missing stage1 backbone: {model_path}")
        self.device = device
        state = torch.load(model_path, map_location=device, weights_only=True)
        cfg = infer_transformer_config_from_state_dict(state)
        self.feature_mode = str(cfg["feature_mode"])
        self.model = PINNPolymerTransformer(
            seq_len=T_GRID_POINTS,
            num_maxwell=13,
            d_model=int(cfg["d_model"]),
            nhead=int(cfg["nhead"]),
            num_layers=int(cfg["num_layers"]),
            dim_feedforward=int(cfg["dim_feedforward"]),
            dropout=float(cfg["dropout"]),
            feature_dim=int(cfg["feature_dim"]),
            token_stem=str(cfg["token_stem"]),
        ).to(device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def build_sample(self, npz_path: str, t_standard: np.ndarray, target_freqs_hz: List[float]) -> Optional[Stage2Sample]:
        sample_id = Path(npz_path).stem
        npz = np.load(npz_path, allow_pickle=True)
        valid_freqs = load_npz_freqs(npz)
        if len(valid_freqs) < len(target_freqs_hz):
            return None

        mapped_freqs = []
        for hz in target_freqs_hz:
            chosen = choose_closest_freq(valid_freqs, hz)
            if chosen is None or chosen in mapped_freqs:
                return None
            mapped_freqs.append(chosen)

        anchor_freq = choose_closest_freq(valid_freqs, 1.0) or mapped_freqs[0]
        feat_curve = make_feat_curve(npz, anchor_freq, t_standard, feature_mode=self.feature_mode)
        if feat_curve is None:
            return None

        feat_curve = feat_curve.unsqueeze(0).to(self.device)
        anchor_hz = float(anchor_freq.replace("_", "."))
        omega_feat = torch.tensor([[math.log10(max(anchor_hz, 0.1))]], dtype=torch.float32, device=self.device)
        params = self.model.infer_params(feat_curve, omega_feat)

        param_vec = normalized_param_vector_from_params(params).detach().cpu().view(-1)
        global_feat = params["global_feat"].detach().cpu().view(-1)
        material_vec = torch.cat([param_vec, global_feat], dim=0).float()

        freq_vals = np.array([float(freq.replace("_", ".")) for freq in mapped_freqs], dtype=np.float32)
        h = len(mapped_freqs)
        w = len(t_standard)
        t_norm = ((t_standard - 20.0) / 160.0).astype(np.float32)
        omega_norm = np.log10(np.clip(freq_vals, a_min=0.1, a_max=None)).astype(np.float32)
        t_grid = np.tile(t_norm.reshape(1, -1), (h, 1))
        omega_grid = np.tile(omega_norm.reshape(-1, 1), (1, w))

        n_points = h * w
        t_target = torch.tensor(t_grid.reshape(-1, 1), dtype=torch.float32, device=self.device)
        omega_target = torch.tensor(omega_grid.reshape(-1, 1), dtype=torch.float32, device=self.device)
        params_batched = {}
        for key in ("C1", "C2", "E_e", "E_i", "tau_i"):
            value = params[key]
            repeat_dims = [n_points] + [1] * (value.dim() - 1)
            params_batched[key] = value.repeat(*repeat_dims)
        pred_ep, pred_edp = self.model.physics_decode(params_batched, t_target, omega_target)
        base_ep = pred_ep.detach().cpu().numpy().reshape(h, w)
        base_edp = pred_edp.detach().cpu().numpy().reshape(h, w)

        target_ep_rows = []
        target_edp_rows = []
        mask_ep_rows = []
        mask_edp_rows = []
        for freq in mapped_freqs:
            ep_curve, ep_mask = interp_log_curve_with_mask(npz, f"E_prime_val_{freq}Hz", [f"E_prime_temp_{freq}Hz"], t_standard)
            edp_curve, edp_mask = interp_log_curve_with_mask(
                    npz,
                    f"E_double_prime_val_{freq}Hz",
                    [f"E_double_prime_temp_{freq}Hz", f"E_prime_temp_{freq}Hz"],
                    t_standard,
                )
            target_ep_rows.append(ep_curve)
            target_edp_rows.append(edp_curve)
            mask_ep_rows.append(ep_mask)
            mask_edp_rows.append(edp_mask)
        target_ep = np.stack(target_ep_rows, axis=0)
        target_edp = np.stack(target_edp_rows, axis=0)
        valid_mask = np.stack([np.stack(mask_ep_rows, axis=0), np.stack(mask_edp_rows, axis=0)], axis=0).astype(np.float32)

        base_grid = np.stack([base_ep, base_edp], axis=0).astype(np.float32)
        target_grid = np.stack([target_ep, target_edp], axis=0).astype(np.float32)
        residual_grid = target_grid - base_grid
        grid_input = np.stack([t_grid, omega_grid, base_ep, base_edp], axis=0).astype(np.float32)

        return Stage2Sample(
            sample_id=sample_id,
            anchor_freq=anchor_freq,
            target_freqs=mapped_freqs,
            material_vec=material_vec,
            grid_input=torch.tensor(grid_input, dtype=torch.float32),
            base_grid=torch.tensor(base_grid, dtype=torch.float32),
            target_grid=torch.tensor(target_grid, dtype=torch.float32),
            residual_grid=torch.tensor(residual_grid, dtype=torch.float32),
            valid_mask=torch.tensor(valid_mask, dtype=torch.float32),
        )


class Stage2ResidualDataset(Dataset):
    def __init__(self, file_paths: List[str], stage1_model_path: Path, device: torch.device):
        self.samples: List[Stage2Sample] = []
        backbone = Stage1Backbone(stage1_model_path, device=device)
        for path in file_paths:
            sample = backbone.build_sample(path, T_STANDARD, FREQ_GRID)
            if sample is not None:
                self.samples.append(sample)
        if not self.samples:
            raise RuntimeError("stage2 dataset is empty")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "grid_input": sample.grid_input,
            "material_vec": sample.material_vec,
            "base_grid": sample.base_grid,
            "target_grid": sample.target_grid,
            "residual_grid": sample.residual_grid,
            "valid_mask": sample.valid_mask,
            "sample_id": sample.sample_id,
        }

    def material_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mat = torch.stack([sample.material_vec for sample in self.samples], dim=0)
        mean = mat.mean(dim=0)
        std = mat.std(dim=0)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return mean, std


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        scale = 1.0 / max(in_channels * out_channels, 1)
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, : self.modes1, : self.modes2], self.weight
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")


class FNOResidual2d(nn.Module):
    def __init__(self, grid_in_channels: int, material_dim: int, width: int = 48, modes1: int = 4, modes2: int = 20, depth: int = 4):
        super().__init__()
        self.material_encoder = nn.Sequential(
            nn.Linear(material_dim, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        self.input_proj = nn.Conv2d(grid_in_channels + width, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)])
        self.pointwise_layers = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)])
        self.output_head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, 2, kernel_size=1),
        )

    def forward(self, grid_input: torch.Tensor, material_vec: torch.Tensor) -> torch.Tensor:
        _, _, h, w = grid_input.shape
        material_context = self.material_encoder(material_vec).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        x = self.input_proj(torch.cat([grid_input, material_context], dim=1))
        for spectral, pointwise in zip(self.spectral_layers, self.pointwise_layers):
            x = F.gelu(spectral(x) + pointwise(x))
        return self.output_head(x)


def build_stage2_model(material_dim: int, width: int, modes1: int, modes2: int, depth: int) -> FNOResidual2d:
    return FNOResidual2d(grid_in_channels=4, material_dim=material_dim, width=width, modes1=modes1, modes2=modes2, depth=depth)


def compute_stage2_losses(
    pred_residual: torch.Tensor,
    base_grid: torch.Tensor,
    target_grid: torch.Tensor,
    target_residual: torch.Tensor,
    valid_mask: torch.Tensor,
    resid_l2_weight: float,
    smooth_weight: float,
    invalid_resid_weight: float,
) -> Dict[str, torch.Tensor]:
    refined = base_grid + pred_residual
    mask_sum = torch.clamp(valid_mask.sum(), min=1.0)
    loss_fit = torch.sum(((refined - target_grid) ** 2) * valid_mask) / mask_sum
    loss_residual = torch.sum(((pred_residual - target_residual) ** 2) * valid_mask) / mask_sum
    loss_bound = resid_l2_weight * torch.mean(pred_residual ** 2)
    invalid_mask = 1.0 - valid_mask
    invalid_sum = torch.clamp(invalid_mask.sum(), min=1.0)
    loss_invalid = invalid_resid_weight * torch.sum((pred_residual ** 2) * invalid_mask) / invalid_sum
    if pred_residual.size(-1) >= 2:
        loss_smooth = smooth_weight * torch.mean((pred_residual[:, :, :, 1:] - pred_residual[:, :, :, :-1]) ** 2)
    else:
        loss_smooth = pred_residual.new_tensor(0.0)
    return {
        "total": loss_fit + 0.35 * loss_residual + loss_bound + loss_invalid + loss_smooth,
        "fit": loss_fit,
        "residual": loss_residual,
        "bound": loss_bound,
        "invalid": loss_invalid,
        "smooth": loss_smooth,
    }


def evaluate_stage2_model(model: nn.Module, loader, material_mean: torch.Tensor, material_std: torch.Tensor, device: torch.device):
    model.eval()
    base_ep_rmse = []
    base_edp_rmse = []
    refined_ep_rmse = []
    refined_edp_rmse = []

    with torch.no_grad():
        for batch in loader:
            grid_input = batch["grid_input"].to(device)
            material_vec = batch["material_vec"].to(device)
            base_grid = batch["base_grid"].to(device)
            target_grid = batch["target_grid"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            pred_residual = model(grid_input, (material_vec - material_mean) / material_std)
            refined = base_grid + pred_residual

            channel_mask_sum = torch.clamp(valid_mask.sum(dim=(2, 3)), min=1.0)
            base_err = torch.sqrt(torch.sum(((base_grid - target_grid) ** 2) * valid_mask, dim=(2, 3)) / channel_mask_sum)
            refined_err = torch.sqrt(torch.sum(((refined - target_grid) ** 2) * valid_mask, dim=(2, 3)) / channel_mask_sum)
            base_ep_rmse.extend(base_err[:, 0].detach().cpu().tolist())
            base_edp_rmse.extend(base_err[:, 1].detach().cpu().tolist())
            refined_ep_rmse.extend(refined_err[:, 0].detach().cpu().tolist())
            refined_edp_rmse.extend(refined_err[:, 1].detach().cpu().tolist())

    base_ep_mean = float(np.mean(base_ep_rmse))
    base_edp_mean = float(np.mean(base_edp_rmse))
    refined_ep_mean = float(np.mean(refined_ep_rmse))
    refined_edp_mean = float(np.mean(refined_edp_rmse))
    return {
        "base_rmse_ep_mean": base_ep_mean,
        "base_rmse_edp_mean": base_edp_mean,
        "refined_rmse_ep_mean": refined_ep_mean,
        "refined_rmse_edp_mean": refined_edp_mean,
        "improve_ep_ratio": float((base_ep_mean - refined_ep_mean) / max(base_ep_mean, 1e-8)),
        "improve_edp_ratio": float((base_edp_mean - refined_edp_mean) / max(base_edp_mean, 1e-8)),
    }
