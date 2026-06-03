# -*- coding: utf-8 -*-
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT_FALLBACK = CURRENT_FILE.parents[3]
if str(REPO_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FALLBACK))

from stage1.models.transformer.train_transformer_stage1 import (
    DATA_DIR,
    REPO_ROOT,
    SEED,
    SPLIT_JSON,
    STAGE1_DIR,
    TARGET_BASE_FREQ_HZ,
    T_GRID_POINTS,
    FileLogger,
    PolymerDMADataset,
    choose_closest_freq,
    get_focus10_aux_weight,
    get_main_aux_weight,
    load_npz_freqs,
    make_feat_curve,
    safe_log10_np,
    weighted_mean,
    weighted_point_mse,
)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value is not None else default


def with_run_tag(path: Path, run_tag: str) -> Path:
    if not run_tag:
        return path
    return path.with_name(f"{path.stem}_{run_tag}{path.suffix}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def relative_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + eps))


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


class PureMLPStage1(nn.Module):
    def __init__(self, curve_dim: int, latent_dim: int = 96, hidden_dim: int = 192, dropout: float = 0.08):
        super().__init__()
        self.curve_dim = curve_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(curve_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.response_head = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def encode(self, feat_curve: torch.Tensor, omega_feat: torch.Tensor) -> torch.Tensor:
        x = feat_curve.float().reshape(feat_curve.shape[0], -1)
        omega_feat = omega_feat.float().reshape(feat_curve.shape[0], -1)
        return self.encoder(torch.cat([x, omega_feat], dim=1))

    def forward(self, feat_curve: torch.Tensor, omega_feat: torch.Tensor, t_target: torch.Tensor, omega_target: torch.Tensor):
        z = self.encode(feat_curve, omega_feat)
        head_in = torch.cat([z, t_target.float(), omega_target.float()], dim=1)
        out = self.response_head(head_in)
        pred_ep = out[:, 0:1]
        pred_edp = out[:, 1:2]
        return pred_ep, pred_edp, z


@torch.no_grad()
def evaluate_pointwise(model: PureMLPStage1, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    se_ep = []
    se_edp = []
    for batch in loader:
        (
            feat_curve,
            omega_feat,
            t_target,
            omega_target,
            target_ep,
            target_edp,
            _main_aux_feat_curve,
            _main_aux_omega_feat,
            _has_main_aux,
            _focus10_feat_curve,
            _focus10_omega_feat,
            _has_focus10,
            _sample_weight,
            _is_hard_sample,
        ) = batch

        feat_curve = feat_curve.to(device, non_blocking=True)
        omega_feat = omega_feat.to(device, non_blocking=True)
        t_target = t_target.to(device, non_blocking=True)
        omega_target = omega_target.to(device, non_blocking=True)
        target_ep = target_ep.to(device, non_blocking=True)
        target_edp = target_edp.to(device, non_blocking=True)

        pred_ep, pred_edp, _ = model(feat_curve, omega_feat, t_target, omega_target)
        se_ep.append(((pred_ep - target_ep) ** 2).detach().cpu())
        se_edp.append(((pred_edp - target_edp) ** 2).detach().cpu())

    all_se_ep = torch.cat(se_ep, dim=0).view(-1)
    all_se_edp = torch.cat(se_edp, dim=0).view(-1)
    return {
        "rmse_ep_mean": float(torch.sqrt(all_se_ep.mean()).item()),
        "rmse_edp_mean": float(torch.sqrt(all_se_edp.mean()).item()),
    }


@torch.no_grad()
def evaluate_latent_probe(model: PureMLPStage1, file_paths: List[str], device: torch.device, t_standard: np.ndarray) -> Dict[str, float]:
    rel_vals = []
    rel10_vals = []
    cos_vals = []
    rmse10_vals = []
    sample_count = 0

    for path in file_paths:
        npz = np.load(path, allow_pickle=True)
        valid_freqs = load_npz_freqs(npz)
        if len(valid_freqs) < 2:
            continue

        latents = []
        freq_names = []
        rows = []

        for anchor_freq in valid_freqs:
            feat_curve = make_feat_curve(npz, anchor_freq, t_standard)
            if feat_curve is None:
                continue

            feat_curve = feat_curve.unsqueeze(0).to(device)
            omega_feat = torch.tensor(
                [[np.log10(max(float(anchor_freq.replace("_", ".")), 0.1))]],
                dtype=torch.float32,
                device=device,
            )
            z = model.encode(feat_curve, omega_feat).detach().cpu().numpy().reshape(-1)
            latents.append(z)
            freq_names.append(anchor_freq)

            t_arr = np.asarray(npz[f"E_prime_temp_{anchor_freq}Hz"], dtype=np.float32)
            target_ep = safe_log10_np(np.asarray(npz[f"E_prime_val_{anchor_freq}Hz"], dtype=np.float32))
            target_edp = safe_log10_np(np.asarray(npz[f"E_double_prime_val_{anchor_freq}Hz"], dtype=np.float32))
            t_norm = torch.tensor(((t_arr - 20.0) / 160.0).reshape(-1, 1), dtype=torch.float32, device=device)
            omega_target = torch.full(
                (len(t_arr), 1),
                np.log10(max(float(anchor_freq.replace("_", ".")), 0.1)),
                dtype=torch.float32,
                device=device,
            )
            feat_rep = feat_curve.repeat(len(t_arr), *([1] * (feat_curve.dim() - 1)))
            omega_feat_rep = omega_feat.repeat(len(t_arr), 1)
            pred_ep, pred_edp, _ = model(feat_rep, omega_feat_rep, t_norm, omega_target)
            rows.append(
                {
                    "anchor_freq": anchor_freq,
                    "rmse_ep_self": float(torch.sqrt(torch.mean((pred_ep.squeeze(1) - torch.tensor(target_ep, device=device)) ** 2)).cpu()),
                    "rmse_edp_self": float(torch.sqrt(torch.mean((pred_edp.squeeze(1) - torch.tensor(target_edp, device=device)) ** 2)).cpu()),
                }
            )

        if len(latents) < 2:
            continue

        sample_count += 1
        mat = np.stack(latents, axis=0)
        for i in range(len(mat)):
            for j in range(i + 1, len(mat)):
                rel = relative_l2(mat[i], mat[j])
                cos = cosine_similarity(mat[i], mat[j])
                rel_vals.append(rel)
                cos_vals.append(cos)
                if freq_names[i] in {"10", "10_0"} or freq_names[j] in {"10", "10_0"}:
                    rel10_vals.append(rel)

        for row in rows:
            if row["anchor_freq"] in {"10", "10_0"}:
                rmse10_vals.append(row["rmse_edp_self"])
                break

    return {
        "n_samples_evaluated": sample_count,
        "latent_pairwise_rel_l2_mean": float(np.mean(rel_vals)) if rel_vals else float("nan"),
        "latent_cosine_mean": float(np.mean(cos_vals)) if cos_vals else float("nan"),
        "latent_rel10_mean": float(np.mean(rel10_vals)) if rel10_vals else float("nan"),
        "probe_rmse10_mean": float(np.mean(rmse10_vals)) if rmse10_vals else float("nan"),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    run_tag = env_str("STAGE1_RUN_TAG", "pure_mlp_v1")
    epochs = env_int("STAGE1_EPOCHS", 140)
    batch_train = env_int("STAGE1_BATCH_TRAIN", 512)
    batch_val = env_int("STAGE1_BATCH_VAL", 1024)
    lr = env_float("STAGE1_LR", 1e-3)
    weight_decay = env_float("STAGE1_WEIGHT_DECAY", 1e-4)
    patience = env_int("STAGE1_PATIENCE", 35)
    latent_dim = env_int("STAGE1_LATENT_DIM", 96)
    hidden_dim = env_int("STAGE1_HIDDEN_DIM", 192)
    dropout = env_float("STAGE1_DROPOUT", 0.08)

    results_dir = STAGE1_DIR / "results" / "baselines" / "pure_mlp"
    checkpoint_dir = results_dir / "checkpoints"
    log_dir = results_dir / "logs"
    history_dir = results_dir / "history"
    plot_dir = results_dir / "plots"
    eval_dir = results_dir / "evaluation"
    for folder in (checkpoint_dir, log_dir, history_dir, plot_dir, eval_dir):
        folder.mkdir(parents=True, exist_ok=True)

    model_best = with_run_tag(checkpoint_dir / "pure_mlp_stage1_best_fit.pth", run_tag)
    log_txt = with_run_tag(log_dir / "log_pure_mlp_stage1.txt", run_tag)
    log_jsonl = with_run_tag(log_dir / "log_pure_mlp_stage1.jsonl", run_tag)
    history_npz = with_run_tag(history_dir / "history_pure_mlp_stage1.npz", run_tag)
    loss_plot = with_run_tag(plot_dir / "loss_pure_mlp_stage1.png", run_tag)
    summary_json = eval_dir / f"{run_tag}_summary.json"

    logger = FileLogger(log_txt, log_jsonl)
    logger.log("========== Stage-1 Pure-MLP Baseline ==========")
    logger.log(f"device={device} seed={SEED}")
    logger.log(f"data_dir={DATA_DIR}")
    logger.log(f"split_json={SPLIT_JSON}")

    split = json.loads(Path(SPLIT_JSON).read_text(encoding="utf-8"))
    train_files = [str((REPO_ROOT / p).resolve()) for p in split["train_files"]]
    val_files = [str((REPO_ROOT / p).resolve()) for p in split["val_files"]]
    test_files = [str((REPO_ROOT / p).resolve()) for p in split["test_files"]]

    train_dataset = PolymerDMADataset(train_files, logger, T_grid_points=T_GRID_POINTS, target_base_freq_hz=TARGET_BASE_FREQ_HZ)
    val_dataset = PolymerDMADataset(val_files, logger, T_grid_points=T_GRID_POINTS, target_base_freq_hz=TARGET_BASE_FREQ_HZ)
    test_dataset = PolymerDMADataset(test_files, logger, T_grid_points=T_GRID_POINTS, target_base_freq_hz=TARGET_BASE_FREQ_HZ)

    train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_val, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_val, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    curve_dim = int(train_dataset[0][0].numel())
    model = PureMLPStage1(curve_dim=curve_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_total": [],
        "train_ep": [],
        "train_edp": [],
        "train_main_aux": [],
        "train_focus10_aux": [],
        "val_ep": [],
        "val_edp": [],
        "val_total": [],
        "lr": [],
    }

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        main_aux_w = get_main_aux_weight(epoch)
        focus10_w = get_focus10_aux_weight(epoch)

        sum_total = 0.0
        sum_ep = 0.0
        sum_edp = 0.0
        sum_main_aux = 0.0
        sum_focus10_aux = 0.0

        for batch in train_loader:
            (
                feat_curve,
                omega_feat,
                t_target,
                omega_target,
                target_ep,
                target_edp,
                main_aux_feat_curve,
                main_aux_omega_feat,
                has_main_aux,
                focus10_feat_curve,
                focus10_omega_feat,
                has_focus10,
                sample_weight,
                _is_hard_sample,
            ) = batch

            feat_curve = feat_curve.to(device, non_blocking=True)
            omega_feat = omega_feat.to(device, non_blocking=True)
            t_target = t_target.to(device, non_blocking=True)
            omega_target = omega_target.to(device, non_blocking=True)
            target_ep = target_ep.to(device, non_blocking=True)
            target_edp = target_edp.to(device, non_blocking=True)
            main_aux_feat_curve = main_aux_feat_curve.to(device, non_blocking=True)
            main_aux_omega_feat = main_aux_omega_feat.to(device, non_blocking=True)
            has_main_aux = has_main_aux.to(device, non_blocking=True).squeeze(1)
            focus10_feat_curve = focus10_feat_curve.to(device, non_blocking=True)
            focus10_omega_feat = focus10_omega_feat.to(device, non_blocking=True)
            has_focus10 = has_focus10.to(device, non_blocking=True).squeeze(1)
            sample_weight = sample_weight.to(device, non_blocking=True).squeeze(1)

            optimizer.zero_grad(set_to_none=True)
            pred_ep, pred_edp, z_main = model(feat_curve, omega_feat, t_target, omega_target)

            loss_ep = weighted_point_mse(pred_ep, target_ep, sample_weight)
            loss_edp = weighted_point_mse(pred_edp, target_edp, sample_weight)
            loss = loss_ep + loss_edp

            loss_main_aux = torch.tensor(0.0, device=device)
            loss_focus10 = torch.tensor(0.0, device=device)

            if main_aux_w > 0 and has_main_aux.any():
                use_mask = has_main_aux > 0.5
                z_aux = model.encode(main_aux_feat_curve[use_mask], main_aux_omega_feat[use_mask])
                per_sample = F.smooth_l1_loss(z_main[use_mask], z_aux, reduction="none").mean(dim=1)
                loss_main_aux = weighted_mean(per_sample, sample_weight[use_mask])
                loss = loss + main_aux_w * loss_main_aux

            if focus10_w > 0 and has_focus10.any():
                use_mask = has_focus10 > 0.5
                z_aux10 = model.encode(focus10_feat_curve[use_mask], focus10_omega_feat[use_mask])
                per_sample10 = F.smooth_l1_loss(z_main[use_mask], z_aux10, reduction="none").mean(dim=1)
                loss_focus10 = weighted_mean(per_sample10, sample_weight[use_mask])
                loss = loss + focus10_w * loss_focus10

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            sum_total += float(loss.detach().cpu())
            sum_ep += float(loss_ep.detach().cpu())
            sum_edp += float(loss_edp.detach().cpu())
            sum_main_aux += float(loss_main_aux.detach().cpu())
            sum_focus10_aux += float(loss_focus10.detach().cpu())

        scheduler.step()

        model.eval()
        val_ep_sum = 0.0
        val_edp_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                feat_curve, omega_feat, t_target, omega_target, target_ep, target_edp, *_ = batch
                feat_curve = feat_curve.to(device, non_blocking=True)
                omega_feat = omega_feat.to(device, non_blocking=True)
                t_target = t_target.to(device, non_blocking=True)
                omega_target = omega_target.to(device, non_blocking=True)
                target_ep = target_ep.to(device, non_blocking=True)
                target_edp = target_edp.to(device, non_blocking=True)
                pred_ep, pred_edp, _ = model(feat_curve, omega_feat, t_target, omega_target)
                val_ep_sum += float(F.mse_loss(pred_ep, target_ep).detach().cpu())
                val_edp_sum += float(F.mse_loss(pred_edp, target_edp).detach().cpu())

        avg_train_total = sum_total / max(len(train_loader), 1)
        avg_train_ep = sum_ep / max(len(train_loader), 1)
        avg_train_edp = sum_edp / max(len(train_loader), 1)
        avg_train_main_aux = sum_main_aux / max(len(train_loader), 1)
        avg_train_focus10 = sum_focus10_aux / max(len(train_loader), 1)
        avg_val_ep = val_ep_sum / max(len(val_loader), 1)
        avg_val_edp = val_edp_sum / max(len(val_loader), 1)
        avg_val_total = avg_val_ep + avg_val_edp

        history["train_total"].append(avg_train_total)
        history["train_ep"].append(avg_train_ep)
        history["train_edp"].append(avg_train_edp)
        history["train_main_aux"].append(avg_train_main_aux)
        history["train_focus10_aux"].append(avg_train_focus10)
        history["val_ep"].append(avg_val_ep)
        history["val_edp"].append(avg_val_edp)
        history["val_total"].append(avg_val_total)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if avg_val_total < best_val:
            best_val = avg_val_total
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "curve_dim": curve_dim,
                    "latent_dim": latent_dim,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "run_tag": run_tag,
                },
                model_best,
            )

        logger.log_json(
            {
                "epoch": epoch + 1,
                "train_total": avg_train_total,
                "train_ep": avg_train_ep,
                "train_edp": avg_train_edp,
                "train_main_aux": avg_train_main_aux,
                "train_focus10_aux": avg_train_focus10,
                "val_ep": avg_val_ep,
                "val_edp": avg_val_edp,
                "val_total": avg_val_total,
                "main_aux_w": main_aux_w,
                "focus10_w": focus10_w,
                "lr": optimizer.param_groups[0]["lr"],
                "best_val": best_val,
                "best_epoch": best_epoch,
            }
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.log(
                f"Epoch [{epoch + 1:03d}/{epochs}] | Time {time.time() - t0:.2f}s | "
                f"Train {avg_train_total:.4f} (Ep {avg_train_ep:.4f}, Edp {avg_train_edp:.4f}, "
                f"MainAux {avg_train_main_aux:.4f}, F10 {avg_train_focus10:.4f}) | "
                f"Val {avg_val_total:.4f} (Ep {avg_val_ep:.4f}, Edp {avg_val_edp:.4f}) | "
                f"Best {best_val:.4f}@{best_epoch}"
            )

        if best_epoch > 0 and (epoch + 1) - best_epoch >= patience:
            logger.log(f"Early stop at epoch {epoch + 1}")
            break

    np.savez(history_npz, **{k: np.asarray(v, dtype=np.float32) for k, v in history.items()})

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_total"], label="Train total")
    plt.plot(history["val_total"], label="Val total")
    plt.plot(history["train_main_aux"], label="Train main aux")
    plt.plot(history["train_focus10_aux"], label="Train focus10 aux")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(loss_plot, dpi=220)
    plt.close()

    checkpoint = torch.load(model_best, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    test_metrics = evaluate_pointwise(model, test_loader, device=device)
    probe_metrics = evaluate_latent_probe(model, test_files, device=device, t_standard=train_dataset.T_standard)
    summary = {
        "run_tag": run_tag,
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "input_curve": "1Hz E' main input + auxiliary frequency latent consistency",
        **test_metrics,
        **probe_metrics,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("========== Pure-MLP Test Summary ==========")
    logger.log(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
