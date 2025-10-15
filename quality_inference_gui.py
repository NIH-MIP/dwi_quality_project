# --- Dual-Model Image Quality Inference GUI (ADC/T2 & hiB/T2) ---
# - Select T2, ADC, hiB (DICOM folders or NIfTI files)
# - Preprocess to match training (Spacing 1x1x3, pad/crop to 512x512x64, normalize)
# - Run T2/ADC through the ADC-trained model (2-class)
# - Run T2/hiB through the hiB-trained model (2-class)
# - Show prediction + probabilities for each pair
#
# Notes:
# * Class mapping follows your eval code: class 0 = "Diagnostic", class 1 = "Non-Diagnostic".
# * Fill in ADC_MODEL_PATH and HIB_MODEL_PATH with your .pth checkpoints.

import os, sys
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import SimpleITK as sitk
import torch.nn as nn
import monai
from monai.transforms import (
    LoadImage, EnsureChannelFirst, NormalizeIntensity, Spacing,
    SpatialPad, CenterSpatialCrop, Compose
)
from monai.networks.nets import DenseNet121

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model architecture (matches your training code; out_channels=2)
# -----------------------------
class ChannelCrossAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, source, target):
        # source modulates target
        b, c, _, _, _ = source.shape
        squeeze = self.global_pool(source).view(b, c)
        attn = self.fc(squeeze).view(b, c, 1, 1, 1)
        return target * attn

class DualInputDenseNet121(nn.Module):
    def __init__(self, out_channels: int = 2):
        super().__init__()
        self.dwi_model = monai.networks.nets.DenseNet121(
            spatial_dims=3, in_channels=1, out_channels=1024
        )
        self.t2_model = monai.networks.nets.DenseNet121(
            spatial_dims=3, in_channels=1, out_channels=1024
        )
        self.cross_attention = ChannelCrossAttention(1024)
        self.conv_after_concat = nn.Sequential(
            nn.Conv3d(2048, 512, kernel_size=1, stride=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.dwi_convolution_only = nn.Sequential(
            nn.Conv3d(1024, 512, kernel_size=1, stride=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.comparison_block = nn.Sequential(
            nn.Conv3d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512 + 512 + 128, out_channels),
        )

    def forward(self, dwi, t2):
        dwi_features = self.dwi_model.features(dwi)
        t2_features = self.t2_model.features(t2)
        t2_attended = self.cross_attention(dwi_features, t2_features)
        combined_features = torch.cat((t2_attended, dwi_features), dim=1)
        conv_features = self.conv_after_concat(combined_features)
        conv_features_dwi = self.dwi_convolution_only(dwi_features)
        interaction_map = t2_features * dwi_features
        diff_features = self.comparison_block(interaction_map)
        combined_conv_features = torch.cat(
            (conv_features, conv_features_dwi, diff_features), dim=1
        )
        return self.classifier(combined_conv_features)

# -----------------------------
# GUI App
# -----------------------------
class InferenceApp:
    def __init__(self, root, adc_model_path: str, hib_model_path: str):
        self.root = root
        self.root.title("Image Quality Inference GUI (ADC/T2 & hiB/T2)")
        self.adc_model_path = adc_model_path
        self.hib_model_path = hib_model_path

        # Load both models
        self.model_adc = self._load_model(self.adc_model_path)
        self.model_hib = self._load_model(self.hib_model_path)

        # Input selection widgets
        self.image_type = tk.StringVar(value="DICOM")
        tk.Label(root, text="Select Input Format:").pack()
        tk.Radiobutton(root, text="DICOM", variable=self.image_type, value="DICOM").pack()
        tk.Radiobutton(root, text="NIfTI", variable=self.image_type, value="NIFTI").pack()

        tk.Button(root, text="Select Input Images", command=self.select_inputs).pack()
        self.result_label = tk.Label(root, text="", fg="blue", wraplength=700, justify="left")
        self.result_label.pack(pady=10)

    # -----------------------------
    # Model helpers
    # -----------------------------
    def _load_model(self, weights_path: str) -> nn.Module:
        model = DualInputDenseNet121(out_channels=2).to(device)
        if not weights_path or not os.path.exists(weights_path):
            # Leave uninitialized if path not set yet; user said they'll fill it later
            print(f"[WARN] Model weights not found or not set: {weights_path}\n"
                  f"       Load your .pth when ready.")
        else:
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state)
        model.eval()
        return model

    # -----------------------------
    # I/O selection
    # -----------------------------
    def select_inputs(self):
        input_type = self.image_type.get()
        if input_type == "DICOM":
            messagebox.showinfo("Select T2 DICOM", "Select folder with T2-weighted DICOM series.")
            t2_folder = filedialog.askdirectory(title="Select T2 DICOM Folder")

            messagebox.showinfo("Select ADC DICOM", "Select folder with ADC DICOM series.")
            adc_folder = filedialog.askdirectory(title="Select ADC DICOM Folder")

            messagebox.showinfo("Select HiB DICOM", "Select folder with hiB DICOM series.")
            hib_folder = filedialog.askdirectory(title="Select HiB DICOM Folder")

            if t2_folder and adc_folder and hib_folder:
                t2_nifti = self.convert_dicom_to_nifti(t2_folder, "T2.nii.gz")
                adc_nifti = self.convert_dicom_to_nifti(adc_folder, "ADC.nii.gz")
                hib_nifti = self.convert_dicom_to_nifti(hib_folder, "hiB.nii.gz")
                if t2_nifti and adc_nifti and hib_nifti:
                    self.run_inference(t2_nifti, adc_nifti, hib_nifti)
        else:
            messagebox.showinfo("Select T2 NIfTI", "Select the T2-weighted NIfTI.")
            t2_file = filedialog.askopenfilename(
                title="Select T2 NIfTI", filetypes=[("NIfTI files", "*.nii *.nii.gz")]
            )
            messagebox.showinfo("Select ADC NIfTI", "Select the ADC NIfTI.")
            adc_file = filedialog.askopenfilename(
                title="Select ADC NIfTI", filetypes=[("NIfTI files", "*.nii *.nii.gz")]
            )
            messagebox.showinfo("Select HiB NIfTI", "Select the hiB NIfTI.")
            hib_file = filedialog.askopenfilename(
                title="Select HiB NIfTI", filetypes=[("NIfTI files", "*.nii *.nii.gz")]
            )

            if t2_file and adc_file and hib_file:
                self.run_inference(t2_file, adc_file, hib_file)

    # -----------------------------
    # DICOM → NIfTI
    # -----------------------------
    def convert_dicom_to_nifti(self, dicom_folder, output_filename):
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(dicom_folder)
        if not series_IDs:
            messagebox.showerror("Error", f"No DICOM series found in {dicom_folder}")
            return None
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_folder, series_IDs[0])
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        output_path = os.path.join(dicom_folder, output_filename)
        sitk.WriteImage(image, output_path)
        return output_path

    # -----------------------------
    # Preprocessing (match training val/test transforms)
    # -----------------------------
    def _build_transforms(self):
        # Non-dict transforms for single image; mirror your valtest pipeline:
        # Normalize (nonzero, channel-wise), Spacing (1,1,3), pad/crop to 512x512x64
        return Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(channel_dim='no_channel'),
            NormalizeIntensity(nonzero=True, channel_wise=True),
            Spacing(pixdim=(1.0, 1.0, 3.0), mode="bilinear"),
            SpatialPad(spatial_size=(512, 512, 64)),
            CenterSpatialCrop(roi_size=(512, 512, 64)),
        ])

    def _preprocess_pair(self, path_dwi: str, path_t2: str):
        tfm = self._build_transforms()
        dwi = tfm(path_dwi)  # [C,D,H,W]
        t2 = tfm(path_t2)    # [C,D,H,W]
        # Batch dim + device
        return dwi.unsqueeze(0).to(device), t2.unsqueeze(0).to(device)

    # -----------------------------
    # Inference
    # -----------------------------
    def run_inference(self, t2_path, adc_path, hib_path):
        try:
            # Preprocess pairs
            adc_img, t2_for_adc = self._preprocess_pair(adc_path, t2_path)
            hib_img, t2_for_hib = self._preprocess_pair(hib_path, t2_path)

            quality_labels = ["Diagnostic", "Non-Diagnostic"]

            with torch.no_grad():
                # ADC/T2 via ADC model
                logits_adc = self.model_adc(adc_img, t2_for_adc)
                probs_adc = torch.softmax(logits_adc, dim=1).cpu().numpy().flatten()
                idx_adc = int(np.argmax(probs_adc))

                # hiB/T2 via hiB model
                logits_hib = self.model_hib(hib_img, t2_for_hib)
                probs_hib = torch.softmax(logits_hib, dim=1).cpu().numpy().flatten()
                idx_hib = int(np.argmax(probs_hib))

            # Format output
            txt = []
            txt.append("ADC/T2 prediction:")
            txt.append(f"  Class: {quality_labels[idx_adc]}")
            txt.append(f"  Probabilities [Diag, Non-Diag]: {np.round(probs_adc, 4).tolist()}")
            txt.append("")
            txt.append("hiB/T2 prediction:")
            txt.append(f"  Class: {quality_labels[idx_hib]}")
            txt.append(f"  Probabilities [Diag, Non-Diag]: {np.round(probs_hib, 4).tolist()}")

            self.result_label.config(text="\n".join(txt))

        except Exception as e:
            messagebox.showerror("Inference Error", str(e))

# -----------------------------
# Run GUI
# -----------------------------
if __name__ == "__main__":
    
        
    BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))

    # Use filenames only; we’ll place or embed the files later
    ADC_MODEL_PATH = os.path.join(BASE_DIR, "adc_model_epoch_99_2class.pth")
    HIB_MODEL_PATH = os.path.join(BASE_DIR, "hib_model_epoch_100_2class.pth")

    root = tk.Tk()
    app = InferenceApp(root, adc_model_path=ADC_MODEL_PATH, hib_model_path=HIB_MODEL_PATH)
    root.mainloop()
