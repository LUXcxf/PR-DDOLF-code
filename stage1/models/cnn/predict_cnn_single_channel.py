import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE1_DIR = Path(__file__).resolve().parents[2]
SPLIT_JSON = STAGE1_DIR / "data" / "splits" / "split_stage1_cnn.json"
MODEL_PATH = STAGE1_DIR / "results" / "cnn" / "checkpoints" / "cnn_light10hz_bell_best_consistency.pth"
PLOT_DIR = STAGE1_DIR / "results" / "cnn" / "plots" / "predict_single_channel"


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
        out = self.relu(out)
        return out


class PINNPolymerModel(nn.Module):
    def __init__(self, curve_dim=100, num_maxwell=13):
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

    def forward(self, feat_curve, omega_feat, T_target, omega_target):
        feat_curve = feat_curve.float()
        omega_feat = omega_feat.float()
        T_target = T_target.float()
        omega_target = omega_target.float()

        with torch.amp.autocast('cuda', enabled=False):
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

            T_real = T_target * 160.0 + 20.0
            omega_real = 2.0 * np.pi * (10.0 ** omega_target)

            log_aT = -C1 * (T_real - self.Tr) / (C2 + (T_real - self.Tr))
            log_aT = torch.clamp(log_aT, min=-15.0, max=15.0)
            a_T = 10.0 ** log_aT

            omega_reduced = omega_real * a_T
            wt = omega_reduced * tau_i
            wt = torch.clamp(wt, max=1e15)
            wt2 = wt ** 2
            denom = 1.0 + wt2

            E_prime_linear = E_e + torch.sum(E_i * (wt2 / denom), dim=1, keepdim=True)
            E_double_prime_linear = torch.sum(E_i * (wt / denom), dim=1, keepdim=True)

            pred_ep = torch.log10(E_prime_linear + 1e-2)
            pred_edp = torch.log10(E_double_prime_linear + 1e-2)
            return pred_ep, pred_edp


def load_test_files(split_path=SPLIT_JSON):
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Cannot find {split_path}. Run the training script first to generate the split.")
    with open(split_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    return [str((REPO_ROOT / Path(p)).resolve()) for p in split_info.get("test_files", [])]


def build_feature_from_sample(data, t_standard):
    available_keys = data.files
    raw_freqs = [k.split('_')[-1].replace('Hz', '') for k in available_keys if k.startswith('E_prime_temp_')]
    freqs = sorted(list(set(raw_freqs)), key=lambda x: float(x.replace('_', '.')))
    if not freqs:
        raise ValueError("No frequency curves were found in this sample.")

    base_freq = freqs[0]
    base_t = data[f"E_prime_temp_{base_freq}Hz"]
    base_ep = data[f"E_prime_val_{base_freq}Hz"]
    interp_func = interp1d(base_t, base_ep, kind='linear', bounds_error=False, fill_value="extrapolate")
    base_ep_standard = np.log10(np.clip(interp_func(t_standard), a_min=1e-2, a_max=None))

    feat_curve = torch.tensor(base_ep_standard, dtype=torch.float32).unsqueeze(0)
    base_f_norm = np.log10(max(float(base_freq.replace('_', '.')), 0.1))
    omega_feat = torch.tensor([[base_f_norm]], dtype=torch.float32)
    return feat_curve, omega_feat, freqs


def get_prediction_temperature_grid(data, freq_str, default_points=200):
    ep_temp_key = f"E_prime_temp_{freq_str}Hz"
    edp_temp_key = f"E_double_prime_temp_{freq_str}Hz"

    temp_arrays = []
    if ep_temp_key in data.files:
        temp_arrays.append(np.asarray(data[ep_temp_key], dtype=float))
    if edp_temp_key in data.files:
        temp_arrays.append(np.asarray(data[edp_temp_key], dtype=float))

    if not temp_arrays:
        raise ValueError(f"Sample is missing {freq_str}Hz temperature data.")

    merged_t = np.unique(np.concatenate(temp_arrays))
    merged_t = merged_t[np.isfinite(merged_t)]
    if merged_t.size == 0:
        raise ValueError(f"Sample is missing {freq_str}Hz temperature data.")

    t_min = float(np.min(merged_t))
    t_max = float(np.max(merged_t))

    if merged_t.size >= 2 and t_max > t_min:
        return np.linspace(t_min, t_max, max(default_points, merged_t.size))
    return merged_t


def save_selected_sample_plots(model, sample_path, device, save_dir=PLOT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    sample_name = os.path.splitext(os.path.basename(sample_path))[0]
    print(f"\nStart test sample: [{sample_name}]")

    data = np.load(sample_path, allow_pickle=True)
    t_standard = np.linspace(20, 180, 100)
    feat_curve, omega_feat, freqs = build_feature_from_sample(data, t_standard)

    feat_curve = feat_curve.to(device)
    omega_feat = omega_feat.to(device)

    model.eval()
    with torch.no_grad():
        for f_str in freqs:
            try:
                freq_num = float(f_str.replace('_', '.'))
                t_plot = get_prediction_temperature_grid(data, f_str)
                t_target_tensor = torch.tensor((t_plot - 20) / 160.0, dtype=torch.float32).unsqueeze(1).to(device)
                omega_target_tensor = torch.tensor([[np.log10(max(freq_num, 0.1))]], dtype=torch.float32)
                omega_target_tensor = omega_target_tensor.repeat(len(t_plot), 1).to(device)

                feat_batch = feat_curve.repeat(len(t_plot), 1)
                omega_feat_batch = omega_feat.repeat(len(t_plot), 1)
                pred_ep_log, pred_edp_log = model(feat_batch, omega_feat_batch, t_target_tensor, omega_target_tensor)

                pred_ep_np = pred_ep_log.cpu().numpy().reshape(-1)
                pred_edp_np = pred_edp_log.cpu().numpy().reshape(-1)

                plt.figure(figsize=(10, 6))
                plt.plot(t_plot, pred_ep_np, color='red', linestyle='-', linewidth=2,
                         label=f"Pred E' ({freq_num}Hz)")

                if f"E_prime_temp_{f_str}Hz" in data.files and f"E_prime_val_{f_str}Hz" in data.files:
                    t_real_ep = np.asarray(data[f"E_prime_temp_{f_str}Hz"], dtype=float)
                    ep_real = np.log10(np.clip(data[f"E_prime_val_{f_str}Hz"], a_min=1e-2, a_max=None))
                    plt.scatter(t_real_ep, ep_real, color='darkred', marker='o', s=30,
                                label=f"True E' ({freq_num}Hz)")

                if f"E_double_prime_val_{f_str}Hz" in data.files and f"E_double_prime_temp_{f_str}Hz" in data.files:
                    t_real_edp = np.asarray(data[f"E_double_prime_temp_{f_str}Hz"], dtype=float)
                    edp_real = np.log10(np.clip(data[f"E_double_prime_val_{f_str}Hz"], a_min=1e-2, a_max=None))
                    plt.plot(t_plot, pred_edp_np, color='blue', linestyle='--', linewidth=2,
                             label=f"Pred E'' ({freq_num}Hz)")
                    plt.scatter(t_real_edp, edp_real, color='darkblue', marker='x', s=40,
                                label=f"True E'' ({freq_num}Hz)")

                plt.title(f"Physics Model Fit (1D-ResNet) - Sample: {sample_name} @ {freq_num}Hz")
                plt.xlabel("Temperature (°C)")
                plt.ylabel("Log10 Modulus (Pa)")
                plt.xlim(float(np.min(t_plot)), float(np.max(t_plot)))
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                save_path = os.path.join(save_dir, f"{sample_name}_{f_str}Hz.png")
                plt.savefig(save_path, dpi=300)
                plt.close()
                print(f"  [+] Saved figure: {save_path} | prediction temperature range: [{t_plot.min():.2f}, {t_plot.max():.2f}] degC")
            except Exception as e:
                print(f"  [-] Plotting {f_str}Hz failed and was skipped: {e}")


def main():
    test_files = load_test_files()
    if not test_files:
        print("Evaluation complete.")
        return

    sample_map = {os.path.splitext(os.path.basename(p))[0]: p for p in test_files}
    sample_ids = sorted(sample_map.keys())

    print("Evaluation complete.")
    for sid in sample_ids:
        print(sid)

    selected_id = input("\nEnter the sample ID to test and save curves: ").strip()
    if selected_id not in sample_map:
        print(f"Input sample ID does not exist: {selected_id}")
        return

    if not os.path.exists("polymer_pinn_best.pth"):
        print("Cannot find polymer_pinn_best.pth. Train the model first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINNPolymerModel().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    save_selected_sample_plots(model, sample_map[selected_id], device)


if __name__ == '__main__':
    main()
