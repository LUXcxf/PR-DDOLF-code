import json
import os
from pathlib import Path
from typing import List, Tuple

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
OUTPUT_DIR = STAGE1_DIR / "results" / "transformer" / "analysis" / OUTPUT_TAG
ANCHOR_TARGET_HZ = env_float("STAGE1_ANCHOR_HZ", 1.0)
SELECTED_SAMPLE_IDS = [x.strip() for x in env_str("STAGE1_SAMPLE_IDS", "").split(",") if x.strip()]
PCA_SPLIT_NAME = env_str("STAGE1_PCA_SPLIT_NAME", "test").lower()
PCA_MAX_FILES = env_int("STAGE1_PCA_MAX_FILES", 31)
MAX_FREQS_PER_SAMPLE = env_int("STAGE1_MAX_FREQS_PER_SAMPLE", 4)
T_STANDARD = np.linspace(20.0, 180.0, T_GRID_POINTS, dtype=np.float32)


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_split_files(split_name: str) -> List[str]:
    if SPLIT_JSON.exists() and split_name in {"train", "val", "test"}:
        with open(SPLIT_JSON, "r", encoding="utf-8") as f:
            split = json.load(f)
        files = [resolve_repo_path(p) for p in split.get(f"{split_name}_files", [])]
        files = [str(p.resolve()) for p in files if p.exists()]
        if files:
            return files
    return sorted(str(p.resolve()) for p in DATA_DIR.glob("*.npz"))


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


def pca_reduce(x: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=0, keepdims=True)
    x_centered = x - mean
    _, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    components = vt[:n_components]
    proj = x_centered @ components.T
    var = (s ** 2) / max(x.shape[0] - 1, 1)
    var_ratio = var[:n_components] / max(np.sum(var), 1e-12)
    return proj.astype(np.float32), var_ratio.astype(np.float32)


def summarize_cluster_compactness(points: np.ndarray, labels: List[str]) -> dict:
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {"within_mean": np.nan, "between_mean": np.nan, "ratio": np.nan}

    within = []
    centroids = []
    for label in unique_labels:
        cluster = points[labels == label]
        centroid = np.mean(cluster, axis=0)
        centroids.append(centroid)
        within.extend(np.linalg.norm(cluster - centroid, axis=1).tolist())

    centroids = np.stack(centroids, axis=0)
    between = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            between.append(float(np.linalg.norm(centroids[i] - centroids[j])))

    within_mean = float(np.mean(within)) if within else np.nan
    between_mean = float(np.mean(between)) if between else np.nan
    ratio = within_mean / max(between_mean, 1e-12) if np.isfinite(within_mean) and np.isfinite(between_mean) else np.nan
    return {"within_mean": within_mean, "between_mean": between_mean, "ratio": float(ratio)}


def encode_with_attention(model: PINNPolymerTransformer, feat_curve: torch.Tensor):
    encoder = model.encoder
    x = feat_curve.float()
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    x = encoder.feature_norm(x)

    if encoder.token_stem == "conv":
        x = x.transpose(1, 2)
        x = encoder.input_proj(x)
        x = x.transpose(1, 2)
    else:
        x = encoder.input_proj(x)

    cls = encoder.cls_token.expand(x.size(0), -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = encoder.pos_encoder(x)

    attn_maps = []
    for layer in encoder.transformer.layers:
        if layer.norm_first:
            x_norm = layer.norm1(x)
            attn_out, attn_weights = layer.self_attn(
                x_norm,
                x_norm,
                x_norm,
                need_weights=True,
                average_attn_weights=False,
            )
            x = x + layer.dropout1(attn_out)

            x_norm2 = layer.norm2(x)
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_norm2))))
            x = x + layer.dropout2(ff)
        else:
            attn_out, attn_weights = layer.self_attn(
                x,
                x,
                x,
                need_weights=True,
                average_attn_weights=False,
            )
            x = layer.norm1(x + layer.dropout1(attn_out))
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(ff))

        attn_maps.append(attn_weights.detach().cpu().numpy())

    x = encoder.norm(x)
    global_feat = x[:, 0, :].detach().cpu().numpy()
    token_feat = x[:, 1:, :].detach().cpu().numpy()
    return global_feat, token_feat, attn_maps


def make_tick_positions(n_tokens: int, n_ticks: int = 8):
    positions = np.linspace(0, n_tokens - 1, min(n_ticks, n_tokens)).astype(int)
    labels = []
    for pos in positions:
        if pos == 0:
            labels.append("CLS")
        else:
            t_idx = min(pos - 1, len(T_STANDARD) - 1)
            labels.append(f"{T_STANDARD[t_idx]:.0f}")
    return positions, labels


def save_attention_heatmaps(sample_id: str, anchor_freq: str, attn_maps: List[np.ndarray]):
    sample_dir = OUTPUT_DIR / "attention" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, attn in enumerate(attn_maps, start=1):
        attn = attn[0]
        avg_attn = attn.mean(axis=0)
        positions, labels = make_tick_positions(avg_attn.shape[0])

        plt.figure(figsize=(7, 6))
        plt.imshow(avg_attn, aspect="auto", cmap="magma")
        plt.colorbar(label="attention weight")
        plt.xticks(positions, labels, rotation=45)
        plt.yticks(positions, labels)
        plt.title(f"Layer {layer_idx} average attention | sample {sample_id} | {anchor_freq}Hz")
        plt.xlabel("Key token")
        plt.ylabel("Query token")
        plt.tight_layout()
        plt.savefig(sample_dir / f"{sample_id}_layer{layer_idx}_avg_attention.png", dpi=240)
        plt.close()

        n_heads = attn.shape[0]
        n_cols = min(2, n_heads)
        n_rows = int(np.ceil(n_heads / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 3.5 * n_rows), squeeze=False)
        for head_idx in range(n_heads):
            ax = axes[head_idx // n_cols][head_idx % n_cols]
            ax.imshow(attn[head_idx], aspect="auto", cmap="magma")
            ax.set_title(f"Layer {layer_idx} Head {head_idx + 1}")
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45)
            ax.set_yticks(positions)
            ax.set_yticklabels(labels)
        for head_idx in range(n_heads, n_rows * n_cols):
            axes[head_idx // n_cols][head_idx % n_cols].axis("off")
        fig.suptitle(f"Per-head attention | sample {sample_id} | {anchor_freq}Hz", y=0.98)
        fig.tight_layout()
        fig.savefig(sample_dir / f"{sample_id}_layer{layer_idx}_head_attention_grid.png", dpi=220)
        plt.close(fig)


def save_cls_attention_curves(sample_id: str, anchor_freq: str, attn_maps: List[np.ndarray]):
    sample_dir = OUTPUT_DIR / "attention" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    temperatures = T_STANDARD

    plt.figure(figsize=(8, 5))
    for layer_idx, attn in enumerate(attn_maps, start=1):
        attn = attn[0]
        cls_to_tokens = attn[:, 0, 1:].mean(axis=0)
        plt.plot(temperatures, cls_to_tokens, linewidth=2, label=f"Layer {layer_idx}")
    plt.xlabel("Temperature (C)")
    plt.ylabel("CLS -> token attention")
    plt.title(f"CLS attention over temperature | sample {sample_id} | {anchor_freq}Hz")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(sample_dir / f"{sample_id}_cls_attention_by_layer.png", dpi=240)
    plt.close()

    layer_mean = np.stack([attn[0][:, 0, 1:].mean(axis=0) for attn in attn_maps], axis=0)
    plt.figure(figsize=(8, 5))
    plt.plot(temperatures, layer_mean.mean(axis=0), color="darkorange", linewidth=2.5)
    plt.fill_between(
        temperatures,
        layer_mean.mean(axis=0) - layer_mean.std(axis=0),
        layer_mean.mean(axis=0) + layer_mean.std(axis=0),
        color="orange",
        alpha=0.25,
    )
    plt.xlabel("Temperature (C)")
    plt.ylabel("CLS -> token attention")
    plt.title(f"Mean CLS attention with layer spread | sample {sample_id} | {anchor_freq}Hz")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(sample_dir / f"{sample_id}_cls_attention_mean.png", dpi=240)
    plt.close()


def analyze_sample(model: PINNPolymerTransformer, feature_mode: str, sample_path: str):
    sample_id = Path(sample_path).stem
    npz = np.load(sample_path, allow_pickle=True)
    valid_freqs = load_npz_freqs(npz)
    if not valid_freqs:
        print(f"skip {sample_id}: no valid frequencies")
        return

    anchor_freq = choose_closest_freq(valid_freqs, ANCHOR_TARGET_HZ) or valid_freqs[0]
    feat_curve, omega_feat = build_anchor_inputs(npz, anchor_freq, feature_mode)
    if feat_curve is None:
        print(f"skip {sample_id}: failed to build anchor feature")
        return

    with torch.no_grad():
        params = model.infer_params(feat_curve, omega_feat)
        global_feat, token_feat, attn_maps = encode_with_attention(model, feat_curve)

    save_attention_heatmaps(sample_id, anchor_freq, attn_maps)
    save_cls_attention_curves(sample_id, anchor_freq, attn_maps)

    sample_dir = OUTPUT_DIR / "attention" / sample_id
    np.savez(
        sample_dir / f"{sample_id}_analysis_arrays.npz",
        global_feat=global_feat,
        token_feat=token_feat,
        attention=np.stack([attn[0] for attn in attn_maps], axis=0),
        C1=params["C1"].detach().cpu().numpy(),
        C2=params["C2"].detach().cpu().numpy(),
        E_e_log=params["E_e_log"].detach().cpu().numpy(),
    )
    print(f"saved attention analysis for sample {sample_id} -> {sample_dir}")


def build_pca_dataset(model: PINNPolymerTransformer, feature_mode: str):
    files = load_split_files(PCA_SPLIT_NAME)[:PCA_MAX_FILES]
    rows = []
    points = []

    for sample_path in files:
        sample_id = Path(sample_path).stem
        npz = np.load(sample_path, allow_pickle=True)
        valid_freqs = load_npz_freqs(npz)[:MAX_FREQS_PER_SAMPLE]
        for anchor_freq in valid_freqs:
            feat_curve, omega_feat = build_anchor_inputs(npz, anchor_freq, feature_mode)
            if feat_curve is None:
                continue
            with torch.no_grad():
                global_feat, _, _ = encode_with_attention(model, feat_curve)
            points.append(global_feat[0])
            rows.append({"sample_id": sample_id, "anchor_freq": anchor_freq})

    return rows, np.asarray(points, dtype=np.float32)


def save_pca_plots(model: PINNPolymerTransformer, feature_mode: str):
    pca_dir = OUTPUT_DIR / "embedding"
    pca_dir.mkdir(parents=True, exist_ok=True)

    rows, points = build_pca_dataset(model, feature_mode)
    if len(rows) < 4:
        print("skip PCA: not enough embedding points")
        return

    proj, var_ratio = pca_reduce(points, n_components=2)
    sample_ids = [row["sample_id"] for row in rows]
    freqs = [row["anchor_freq"] for row in rows]

    unique_samples = sorted(set(sample_ids))
    cmap = plt.get_cmap("tab20", max(len(unique_samples), 3))
    color_map = {sample_id: cmap(i % cmap.N) for i, sample_id in enumerate(unique_samples)}
    marker_map = {"1": "o", "2": "s", "5": "^", "10": "D", "10_0": "D"}

    plt.figure(figsize=(9, 7))
    for i, row in enumerate(rows):
        plt.scatter(
            proj[i, 0],
            proj[i, 1],
            color=color_map[row["sample_id"]],
            marker=marker_map.get(row["anchor_freq"], "o"),
            s=52,
            alpha=0.85,
        )

    for sample_id in unique_samples:
        idx = [i for i, sid in enumerate(sample_ids) if sid == sample_id]
        if len(idx) >= 2:
            seq = proj[idx]
            plt.plot(seq[:, 0], seq[:, 1], color=color_map[sample_id], alpha=0.35, linewidth=1.2)

    plt.xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({var_ratio[1] * 100:.1f}%)")
    plt.title(f"Global feature PCA | split={PCA_SPLIT_NAME} | model={OUTPUT_TAG}")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(pca_dir / "global_feat_pca.png", dpi=240)
    plt.close()

    cluster_stats = summarize_cluster_compactness(proj, sample_ids)
    with open(pca_dir / "global_feat_pca_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_points": len(rows),
                "n_samples": len(unique_samples),
                "pc1_var_ratio": float(var_ratio[0]),
                "pc2_var_ratio": float(var_ratio[1]),
                **cluster_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(pca_dir / "global_feat_pca_points.csv", "w", encoding="utf-8") as f:
        f.write("sample_id,anchor_freq,pc1,pc2\n")
        for i, row in enumerate(rows):
            f.write(f"{row['sample_id']},{row['anchor_freq']},{proj[i, 0]},{proj[i, 1]}\n")

    print(f"saved embedding PCA -> {pca_dir}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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

    print(f"device = {DEVICE}")
    print(f"model = {MODEL_PATH}")
    print(f"output = {OUTPUT_DIR}")

    candidate_files = load_split_files("test")
    sample_map = {Path(path).stem: path for path in candidate_files}
    for sample_id in SELECTED_SAMPLE_IDS:
        path = sample_map.get(sample_id)
        if path is None:
            print(f"skip {sample_id}: not found in test split")
            continue
        analyze_sample(model, feature_mode, path)

    save_pca_plots(model, feature_mode)


if __name__ == "__main__":
    main()
