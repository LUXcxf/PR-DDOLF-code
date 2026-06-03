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
    ProbeSample,
    REPO_ROOT,
    SEED,
    SPLIT_JSON,
    STAGE1_DIR,
    TARGET_BASE_FREQ_HZ,
    T_GRID_POINTS,
    FileLogger,
    ParameterHead,
    PolymerDMADataset,
    core_anchor_consistency_loss,
    evaluate_probe_consistency,
    get_focus10_aux_weight,
    get_main_aux_weight,
    load_npz_freqs,
    relative_param_consistency_loss,
    spectrum_shape_prior,
    tau_smooth_prior,
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


class SoftPINNStage1(nn.Module):
    def __init__(self, curve_dim: int, latent_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.08, num_maxwell: int = 13):
        super().__init__()
        self.curve_dim = curve_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_maxwell = num_maxwell
        self.Tr = 100.0

        self.encoder = nn.Sequential(
            nn.Linear(curve_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )
        self.param_head = ParameterHead(latent_dim + 1, num_maxwell=num_maxwell)
        self.response_head = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def encode(self, feat_curve: torch.Tensor, omega_feat: torch.Tensor) -> torch.Tensor:
        x = feat_curve.reshape(feat_curve.shape[0], -1).float()
        x = torch.cat([x, omega_feat.float()], dim=1)
        return self.encoder(x)

    def infer_params(self, feat_curve: torch.Tensor, omega_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(feat_curve, omega_feat)
        x = torch.cat([z, omega_feat.float()], dim=1)
        out_wlf, out_ee, out_ei, out_tau = self.param_head(x)

        C1 = 5.0 + 25.0 * torch.sigmoid(out_wlf[:, 0:1])
        C2 = 50.0 + 250.0 * torch.sigmoid(out_wlf[:, 1:2])
        E_e_log = 1.0 + 9.0 * torch.sigmoid(out_ee)
        E_e = 10.0 ** E_e_log
        E_i_log = 0.5 + 9.5 * torch.sigmoid(out_ei)
        E_i = 10.0 ** E_i_log
        tau_base_log = -15.0 + 10.0 * torch.sigmoid(out_tau[:, 0:1])
        tau_delta_log = 0.08 + 1.8 * torch.sigmoid(out_tau[:, 1:])
        tau_log = torch.cumsum(torch.cat([tau_base_log, tau_delta_log], dim=1), dim=1)
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
            "global_feat": z,
            "token_feat": z.unsqueeze(1),
        }

    def physics_decode(self, params: Dict[str, torch.Tensor], t_target: torch.Tensor, omega_target: torch.Tensor):
        T_real = t_target * 160.0 + 20.0
        omega_real = 2.0 * math.pi * (10.0 ** omega_target)

        log_aT = -params["C1"] * (T_real - self.Tr) / (params["C2"] + (T_real - self.Tr) + 1e-6)
        log_aT = torch.clamp(log_aT, min=-15.0, max=15.0)
        a_T = 10.0 ** log_aT
        omega_reduced = omega_real * a_T

        wt = torch.clamp(omega_reduced * params["tau_i"], max=1e15)
        wt2 = wt ** 2
        denom = 1.0 + wt2

        e_prime = params["E_e"] + torch.sum(params["E_i"] * (wt2 / denom), dim=1, keepdim=True)
        e_double_prime = torch.sum(params["E_i"] * (wt / denom), dim=1, keepdim=True)
        return torch.log10(e_prime + 1e-8), torch.log10(e_double_prime + 1e-8)

    def response_decode(self, z: torch.Tensor, t_target: torch.Tensor, omega_target: torch.Tensor):
        x = torch.cat([z, t_target.float(), omega_target.float()], dim=1)
        out = self.response_head(x)
        return out[:, 0:1], out[:, 1:2]

    def forward(self, feat_curve: torch.Tensor, omega_feat: torch.Tensor, t_target: torch.Tensor, omega_target: torch.Tensor):
        z = self.encode(feat_curve, omega_feat)
        params = self.infer_params(feat_curve, omega_feat)
        pred_ep, pred_edp = self.response_decode(z, t_target, omega_target)
        return pred_ep, pred_edp, params


@torch.no_grad()
def evaluate_pointwise(model: SoftPINNStage1, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    se_ep = []
    se_edp = []
    for batch in loader:
        feat_curve, omega_feat, t_target, omega_target, target_ep, target_edp, *_ = batch
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
def evaluate_param_summary(model: SoftPINNStage1, file_paths: List[str], device: torch.device, t_standard: np.ndarray) -> Dict[str, float]:
    rows = []
    for path in file_paths:
        try:
            npz = np.load(path, allow_pickle=True)
            freqs = load_npz_freqs(npz)
            if len(freqs) < 2:
                continue
            probe = ProbeSample(sample_name=Path(path).stem, file_path=path, all_freqs=freqs)
            rows.append(evaluate_probe_consistency(model, probe, device=device, T_standard=t_standard))
        except Exception:
            continue

    if not rows:
        return {
            "n_samples_evaluated": 0,
            "param_cv_mean": float("nan"),
            "pairwise_rel_l2_mean": float("nan"),
            "probe_rel10_mean": float("nan"),
            "probe_rmse10_mean": float("nan"),
        }

    rmse10_vals = [r["rmse10_edp"] for r in rows if not np.isnan(r["rmse10_edp"])]
    return {
        "n_samples_evaluated": len(rows),
        "param_cv_mean": float(np.mean([r["param_cv_mean"] for r in rows])),
        "pairwise_rel_l2_mean": float(np.mean([r["param_rel_l2_mean"] for r in rows])),
        "probe_rel10_mean": float(np.mean([r["param_rel_l2_10_mean"] for r in rows])),
        "probe_rmse10_mean": float(np.mean(rmse10_vals)) if rmse10_vals else float("nan"),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    run_tag = env_str("STAGE1_RUN_TAG", "soft_pinn_v1")
    epochs = env_int("STAGE1_EPOCHS", 180)
    batch_train = env_int("STAGE1_BATCH_TRAIN", 384)
    batch_val = env_int("STAGE1_BATCH_VAL", 768)
    lr = env_float("STAGE1_LR", 9e-4)
    weight_decay = env_float("STAGE1_WEIGHT_DECAY", 1e-4)
    patience = env_int("STAGE1_PATIENCE", 40)
    latent_dim = env_int("STAGE1_LATENT_DIM", 128)
    hidden_dim = env_int("STAGE1_HIDDEN_DIM", 256)
    dropout = env_float("STAGE1_DROPOUT", 0.08)
    phys_res_w = env_float("STAGE1_SOFT_PINN_PHYS_RES_W", 0.28)
    phys_fit_w = env_float("STAGE1_SOFT_PINN_PHYS_FIT_W", 0.20)
    main_param_w = env_float("STAGE1_SOFT_PINN_MAIN_PARAM_W", 0.10)
    focus10_param_w = env_float("STAGE1_SOFT_PINN_FOCUS10_PARAM_W", 0.08)
    tau_prior_w = env_float("STAGE1_SOFT_PINN_TAU_PRIOR_W", 0.0015)
    spectrum_prior_w = env_float("STAGE1_SOFT_PINN_SPECTRUM_PRIOR_W", 0.0030)

    results_dir = STAGE1_DIR / "results" / "baselines" / "soft_pinn"
    checkpoint_dir = results_dir / "checkpoints"
    log_dir = results_dir / "logs"
    history_dir = results_dir / "history"
    plot_dir = results_dir / "plots"
    eval_dir = results_dir / "evaluation"
    for folder in (checkpoint_dir, log_dir, history_dir, plot_dir, eval_dir):
        folder.mkdir(parents=True, exist_ok=True)

    model_best = with_run_tag(checkpoint_dir / "soft_pinn_stage1_best_fit.pth", run_tag)
    log_txt = with_run_tag(log_dir / "log_soft_pinn_stage1.txt", run_tag)
    log_jsonl = with_run_tag(log_dir / "log_soft_pinn_stage1.jsonl", run_tag)
    history_npz = with_run_tag(history_dir / "history_soft_pinn_stage1.npz", run_tag)
    loss_plot = with_run_tag(plot_dir / "loss_soft_pinn_stage1.png", run_tag)
    summary_json = eval_dir / f"{run_tag}_summary.json"

    logger = FileLogger(log_txt, log_jsonl)
    logger.log("========== Stage-1 Soft-PINN Baseline ==========")
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
    model = SoftPINNStage1(curve_dim=curve_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_total": [],
        "train_ep": [],
        "train_edp": [],
        "train_phys_res": [],
        "train_phys_fit": [],
        "train_main_aux": [],
        "train_focus10_aux": [],
        "train_tau_prior": [],
        "train_spectrum_prior": [],
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
        sum_phys_res = 0.0
        sum_phys_fit = 0.0
        sum_main_aux = 0.0
        sum_focus10_aux = 0.0
        sum_tau_prior = 0.0
        sum_spectrum_prior = 0.0

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

            pred_ep, pred_edp, params_main = model(feat_curve, omega_feat, t_target, omega_target)
            phys_ep, phys_edp = model.physics_decode(params_main, t_target, omega_target)

            loss_ep = weighted_point_mse(pred_ep, target_ep, sample_weight)
            loss_edp = weighted_point_mse(pred_edp, target_edp, sample_weight)
            loss_fit = loss_ep + loss_edp

            loss_phys_res = weighted_point_mse(pred_ep, phys_ep, sample_weight) + weighted_point_mse(pred_edp, phys_edp, sample_weight)
            loss_phys_fit = weighted_point_mse(phys_ep, target_ep, sample_weight) + weighted_point_mse(phys_edp, target_edp, sample_weight)
            loss_tau_prior = tau_prior_w * tau_smooth_prior(params_main)
            loss_spectrum_prior = spectrum_prior_w * spectrum_shape_prior(params_main)

            loss = loss_fit + phys_res_w * loss_phys_res + phys_fit_w * loss_phys_fit + loss_tau_prior + loss_spectrum_prior

            loss_main_aux = torch.tensor(0.0, device=device)
            loss_focus10 = torch.tensor(0.0, device=device)

            if main_aux_w > 0 and has_main_aux.any():
                use_mask = has_main_aux > 0.5
                aux_params = model.infer_params(main_aux_feat_curve[use_mask], main_aux_omega_feat[use_mask])
                main_sub = {k: v[use_mask] for k, v in params_main.items()}
                loss_main_aux = core_anchor_consistency_loss(main_sub, aux_params, sample_weight[use_mask])
                loss_main_aux = loss_main_aux + main_param_w * relative_param_consistency_loss(main_sub, aux_params, sample_weight[use_mask])
                loss = loss + main_aux_w * loss_main_aux

            if focus10_w > 0 and has_focus10.any():
                use_mask10 = has_focus10 > 0.5
                aux_params10 = model.infer_params(focus10_feat_curve[use_mask10], focus10_omega_feat[use_mask10])
                main_sub10 = {k: v[use_mask10] for k, v in params_main.items()}
                loss_focus10 = core_anchor_consistency_loss(main_sub10, aux_params10, sample_weight[use_mask10])
                loss_focus10 = loss_focus10 + focus10_param_w * relative_param_consistency_loss(main_sub10, aux_params10, sample_weight[use_mask10])
                loss = loss + focus10_w * loss_focus10

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            sum_total += float(loss.detach().cpu())
            sum_ep += float(loss_ep.detach().cpu())
            sum_edp += float(loss_edp.detach().cpu())
            sum_phys_res += float(loss_phys_res.detach().cpu())
            sum_phys_fit += float(loss_phys_fit.detach().cpu())
            sum_main_aux += float(loss_main_aux.detach().cpu())
            sum_focus10_aux += float(loss_focus10.detach().cpu())
            sum_tau_prior += float(loss_tau_prior.detach().cpu())
            sum_spectrum_prior += float(loss_spectrum_prior.detach().cpu())

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
        avg_train_phys_res = sum_phys_res / max(len(train_loader), 1)
        avg_train_phys_fit = sum_phys_fit / max(len(train_loader), 1)
        avg_train_main_aux = sum_main_aux / max(len(train_loader), 1)
        avg_train_focus10 = sum_focus10_aux / max(len(train_loader), 1)
        avg_train_tau_prior = sum_tau_prior / max(len(train_loader), 1)
        avg_train_spectrum_prior = sum_spectrum_prior / max(len(train_loader), 1)
        avg_val_ep = val_ep_sum / max(len(val_loader), 1)
        avg_val_edp = val_edp_sum / max(len(val_loader), 1)
        avg_val_total = avg_val_ep + avg_val_edp

        history["train_total"].append(avg_train_total)
        history["train_ep"].append(avg_train_ep)
        history["train_edp"].append(avg_train_edp)
        history["train_phys_res"].append(avg_train_phys_res)
        history["train_phys_fit"].append(avg_train_phys_fit)
        history["train_main_aux"].append(avg_train_main_aux)
        history["train_focus10_aux"].append(avg_train_focus10)
        history["train_tau_prior"].append(avg_train_tau_prior)
        history["train_spectrum_prior"].append(avg_train_spectrum_prior)
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
                "train_phys_res": avg_train_phys_res,
                "train_phys_fit": avg_train_phys_fit,
                "train_main_aux": avg_train_main_aux,
                "train_focus10_aux": avg_train_focus10,
                "train_tau_prior": avg_train_tau_prior,
                "train_spectrum_prior": avg_train_spectrum_prior,
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
                f"PhysRes {avg_train_phys_res:.4f}, PhysFit {avg_train_phys_fit:.4f}, "
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
    plt.plot(history["train_phys_res"], label="Train physics residual")
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
    summary_probe = evaluate_param_summary(model, test_files, device=device, t_standard=train_dataset.T_standard)

    summary = {
        "run_tag": run_tag,
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "input_curve": "1Hz E' main input + auxiliary frequency parameter consistency",
        "baseline_type": "Soft-PINN single-stage baseline",
        **test_metrics,
        **summary_probe,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("========== Soft-PINN Test Summary ==========")
    logger.log(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
