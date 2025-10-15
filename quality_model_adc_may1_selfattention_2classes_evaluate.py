## IMPORT BLOCK
import os
import nibabel as nib
import pandas as pd
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import monai 
from monai.networks.nets import DenseNet121
from torch.utils.data import DataLoader
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, EnsureTyped, Spacingd, CenterSpatialCropd, Resized, Compose, NormalizeIntensityd, EnsureChannelFirst, NormalizeIntensity, Spacing, CenterSpatialCrop, Resize, SpatialPadd
)
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import csv
import setuptools.dist
import random
from datetime import datetime
import scipy.stats as st
## from torchcam.methods import GradCAM



###################################################################################################
## Modifiable fields that need to be addressed before running the code

##Image directory
image_directory = r"/home/steckohj/image_quality_project/patient_data_nii"
##Input spreadsheet with the following fields:
master_data_file = r"/home/steckohj/image_quality_project/data_input_v4.csv"
##True if you want saliency mappings
saliency_mapping = True
##Determine the number of batches you want saliency maps for
num_batches_saliency = 4
#batch sizes for testing
test_batch_size = 6
train_batch_size = 6
##Create new folder to save the outputs
now = datetime.now()
now_string = now.strftime("%Y-%m-%d_%H:%M:%S") + "_adc_model"
output_base_folder = r"/home/steckohj/image_quality_project"
output_folder_path = os.path.join(output_base_folder, now_string)
output_folder_path = os.path.join(output_folder_path, "adc_model")
output_directory = os.makedirs(output_folder_path, exist_ok=True)
##File name to save the model path
adc_model_save_path = os.path.join(output_folder_path, "adc_model.pth")
##File where the failed images will be saved
failed_images_file_adc = os.path.join(output_folder_path, "failed_images_adc.csv")


##File name to save ROC_CURVE
roc_curve_file = os.path.join(output_folder_path, "roc_curve.tiff")
##File name to save accuracy per epoch plot
accuracy_plot_file = os.path.join(output_folder_path, "accuracy_curve.tiff")
##Folder where Saliency maps should be saved
saliency_map_folder = r"/home/steckohj/image_quality_project/model_outputs_gpu_categories_skew_straight_2/"
##File where the test outputs should be exported
output_csv_file = os.path.join(output_folder_path, "test_output.csv")
##Number of workers for serving batch to the GPU
number_workers = 16
##Determines if a GPu is available to train/run the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################################################################################
## FUNCTION DEFINITIONS

def get_folder_path(folder_name, base_directory):
    folder_path = os.path.join(base_directory, folder_name)
    if os.path.isdir(folder_path):
        return folder_path
    else:
        return None
    
def get_image_path(row, image_type):
    if row['patient_folder']:
        return os.path.join(row['patient_folder'], image_type)
    else:
        return None
    
def get_t2_image_path(row, image_type):
    if row['patient_folder']:
        if os.path.isfile(os.path.join(row['patient_folder'], "T2.nii.gz")):
            return os.path.join(row['patient_folder'], "T2.nii.gz")
        elif os.path.isfile(os.path.join(row['patient_folder'], 't2.nii.gz')):
            return os.path.join(row['patient_folder'], "t2.nii.gz")
        else:
            return os.path.join(row['patient_folder'], image_type)
    else:
        print(f"There is no t2 image for {row}")
        return None

def pick_dwi(row):
    if row["adc_path"] and os.path.exists(row["adc_path"]):
        return row["adc_path"]
    else:
        print(f"Skipping {row['patient_id']} - high B image not found.")
        skipped_cases.append({
            "patient_id": row["patient_id"],
            "patient_folder": row["patient_folder"],
            "expected_adc_path": row["adc_path"],
            "reason": "Missing or invalid adc image"
        })
        return None

def convert_label(label):
    """
    Converts integer label to 2-class soft label:
    0 = Non-diagnostic     → [0.05, 0.95]
    1 = Acceptable         → [0.7,  0.3 ]
    2 = Diagnostic         → [0.95, 0.05]
    Applies label smoothing of 0.05.
    """
    if label == 0:
        base = [0.05, 0.95]
    elif label == 1:
        base = [0.7, 0.3]
    elif label == 2:
        base = [0.95, 0.05]
    else:
        base = [0.5, 0.5]

    smoothing = 0.05
    smoothed = [(1 - smoothing) * p + smoothing / 2 for p in base]
    total = sum(smoothed)
    return [p / total for p in smoothed]

###################################################################################################
## LOAD IN THE DATA FROM THE CSV FILE
skipped_cases=[]
master_data = pd.read_csv(master_data_file)
master_data["patient_folder"] = master_data["patient_id"].apply(lambda x: get_folder_path(x, image_directory))
master_data["adc_path"] = master_data.apply(lambda row: get_image_path(row, "ADC.nii.gz"), axis=1)
master_data["t2_path"] = master_data.apply(lambda row: get_t2_image_path(row, "t2.nii.gz"), axis=1)
master_data["highb_path"] = master_data.apply(lambda row: get_image_path(row, "hiB.nii.gz"), axis=1)
master_data["dwi_path"] = master_data.apply(pick_dwi, axis=1)
master_data = master_data.dropna(subset=["dwi_path"])
if skipped_cases:
    skipped_df = pd.DataFrame(skipped_cases)
    skipped_df.to_csv(failed_images_file_adc, index=False)
    print(f"Skipped {len(skipped_cases)} cases with missing adc images. Saved to {failed_images_file_adc}")
    
df_training = master_data[master_data['train_test'] == 0]
df_validation = master_data[master_data['train_test'] == 1]
df_testing = master_data[master_data['train_test'] == 2]


training_dict = []
for _, row in df_training.iterrows():
    label = convert_label(row["quality_label"])
    patient = row["patient_folder"]
    # If adc exists, add an entry
    if row["dwi_path"] is not None and os.path.isfile(row["dwi_path"]):
        training_dict.append({
            "dwi_path": row["dwi_path"],
            "t2_path": row["t2_path"],
            "label": label
        })
        print(f"DWI and T2 pair added to training dict for {patient}")

validation_dict = []
for _, row in df_validation.iterrows():
    label = convert_label(row["quality_label"])
    patient = row["patient_folder"]
    if row["dwi_path"] is not None and os.path.isfile(row["dwi_path"]):
        validation_dict.append({
            "dwi_path": row["dwi_path"],
            "t2_path": row["t2_path"],
            "label": label
        })
        print(f"high B and T2 pair added to validation dict for {patient}")
        
testing_dict = []
for _, row in df_testing.iterrows():
    label = convert_label(row["quality_label"])
    patient = row["patient_folder"]
    # If adc exists, add an entry
    if row["dwi_path"] is not None and os.path.isfile(row["dwi_path"]):
        testing_dict.append({
            "dwi_path": row["dwi_path"],
            "t2_path": row["t2_path"],
            "label": label
        })
        print(f"high B and T2 pair added to testing dict for {patient}")

        
print(f"Number of training samples: {len(training_dict)}")
print(f"Number of validation samples: {len(validation_dict)}")
print(f"Number of testing samples: {len(testing_dict)}")

###################################################################################################
## CUSTOM DATASET DEFINITION and CUSTOM DATALOADER FUNCTION

class DualInputDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.failed_images = []

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        dwi_path = self.data[index]["dwi_path"]
        t2_path = self.data[index]["t2_path"]
        label = torch.tensor(self.data[index]["label"], dtype=torch.float32)
        
        to_transform = {"dwi_path": dwi_path, "t2_path": t2_path}
        patient_name = os.path.basename(os.path.dirname(dwi_path))
        
        if not os.path.isfile(dwi_path):
            error_message = f"Missing DWI path: {dwi_path}"
            self.failed_images.append({"index": index, "dwi_path": dwi_path, "t2_path": t2_path, "error": error_message})
            print(error_message)
            return None
        if not os.path.isfile(t2_path):
            error_message = f"Missing T2 path: {t2_path}"
            self.failed_images.append({"index": index, "dwi_path": dwi_path, "t2_path": t2_path, "error": error_message})
            print(error_message)
            return None
        
        
        try:
            transformed = self.transform(to_transform)
            dwi_output = transformed["dwi_path"]
            t2_output = transformed["t2_path"]
            # Check image shapes
            if dwi_output.shape[1:] != t2_output.shape[1:]:
                error_message = f"Shape mismatch: DWI {dwi_output.shape}, T2 {t2_output.shape}"
                self.failed_images.append({"index": index, "dwi_path": dwi_path, "t2_path": t2_path, "error": error_message})
                print(error_message)
                return None
    
        except Exception as e:
            error_message = f"Transformation failed for {patient_name}: {str(e)}"
            self.failed_images.append({"index": index, "dwi_path": dwi_path, "t2_path": t2_path, "error": error_message})
            print(error_message)
            return None
        
        return {
            "dwi": dwi_output,
            "t2": t2_output,
            "label": label,
            "patient_id": patient_name
        }

    def save_failed_images(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        print(f"Created missing directory: {directory}")
        with open(file_path, 'w', newline='') as file:
            fieldnames = ['index', 'dwi_path', 't2_path', 'error']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.failed_images:
                writer.writerow(entry)
        print(f"Failed image pairs saved to {file_path}")

def collate_fn(batch):
    valid_batch = [b for b in batch if b is not None]
    if len(valid_batch) < len(batch):
        skipped_samples = len(batch) - len(valid_batch)
        print(f"Collate_fn: Skipped {skipped_samples} invalid samples in batch")
    
    if len(valid_batch) == 0:
        print("Collate_fn: Entire batch is invalid.")
        return None
    
    dwi_images = torch.stack([item['dwi'] for item in valid_batch])
    t2_images = torch.stack([item['t2'] for item in valid_batch])
    labels = torch.stack([item['label'] for item in valid_batch])
    patient_ids = [item['patient_id'] for item in valid_batch]
    
    return {
        "dwi": dwi_images,
        "t2": t2_images,
        "label": labels,
        "patient_id": patient_ids
    }

#########################################################################################################################################################
## TRANSFORM DEFINITIONS

spatial_size = (512, 512, 64)
roi_size = (320, 320, 16)
spacing = (.25, .25, -1)

train_transforms = Compose([
    LoadImaged(keys=["dwi_path", "t2_path"]),
    EnsureChannelFirstd(keys=["dwi_path", "t2_path"], channel_dim = "no_channel"),
    NormalizeIntensityd(keys=["dwi_path", "t2_path"], nonzero=True, channel_wise=True),
    Spacingd(keys=["dwi_path", "t2_path"], pixdim=(1.0, 1.0, 3.0), mode=("bilinear", "bilinear")),
    SpatialPadd(keys=["dwi_path", "t2_path"], spatial_size=(512, 512, 64)),
    CenterSpatialCropd(keys=["dwi_path", "t2_path"], roi_size=(512, 512, 64)),
    EnsureTyped(keys=["dwi_path", "t2_path"]),
])

valtest_transforms = Compose([
    LoadImaged(keys=["dwi_path", "t2_path"]),
    EnsureChannelFirstd(keys=["dwi_path", "t2_path"]),
    NormalizeIntensityd(keys=["dwi_path", "t2_path"], nonzero=True, channel_wise=True),
    Spacingd(keys=["dwi_path", "t2_path"], pixdim=(1.0, 1.0, 3.0), mode=("bilinear", "bilinear")),
    SpatialPadd(keys=["dwi_path", "t2_path"], spatial_size=(512, 512, 64)),
    CenterSpatialCropd(keys=["dwi_path", "t2_path"], roi_size=(512, 512, 64)),
    EnsureTyped(keys=["dwi_path", "t2_path"]),
])

########################################################################################################################################################
## DATA LOADERS

train_dataset = DualInputDataset(data=training_dict, transform=train_transforms)
val_dataset = DualInputDataset(data=validation_dict, transform=valtest_transforms)
test_dataset = DualInputDataset(data=testing_dict, transform=valtest_transforms)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=number_workers, collate_fn=collate_fn, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=number_workers, collate_fn=collate_fn, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=number_workers, collate_fn=collate_fn, drop_last=False)


train_dataset.save_failed_images(failed_images_file_adc)

##################################################################################################
## MODEL DEFINITION

dropout_probability = 0.5
num_prediction_classes = 2

class ChannelCrossAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelCrossAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, source, target):
        ##source modulates target
        b, c, _, _, _ = source.shape
        squeeze = self.global_pool(source).view(b, c)
        attn = self.fc(squeeze).view(b, c, 1, 1, 1)
        return target * attn

class DualInputDenseNet121(nn.Module):
    def __init__(self, out_channels=num_prediction_classes):
        super(DualInputDenseNet121, self).__init__()
        
        self.dwi_model = monai.networks.nets.DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1024
        )
        
        self.t2_model = monai.networks.nets.DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1024
        )

        ##Cross-attention module: DWI modulates T2
        self.cross_attention = ChannelCrossAttention(1024)
        
        self.conv_after_concat = nn.Sequential(
            nn.Conv3d(2048, 512, kernel_size=1, stride=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.dwi_convolution_only = nn.Sequential(
            nn.Conv3d(1024, 512, kernel_size=1, stride=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.comparison_block = nn.Sequential(
            nn.Conv3d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),  # Enables MC Dropout
            nn.Linear(512 + 512 + 128, out_channels)
        )

    def forward(self, dwi, t2):
        dwi_features = self.dwi_model.features(dwi)  # (B, 1024, D, H, W)
        t2_features = self.t2_model.features(t2)

        ##apply DWI-to-T2 cross attention
        t2_attended = self.cross_attention(dwi_features, t2_features)

        ##concatenate attended T2 with original DWI
        combined_features = torch.cat((t2_attended, dwi_features), dim=1)
        conv_features = self.conv_after_concat(combined_features)

        ##DWI-only features
        conv_features_dwi = self.dwi_convolution_only(dwi_features)

        ##feature interaction map (product of original features)
        interaction_map = t2_features * dwi_features
        diff_features = self.comparison_block(interaction_map)

        ##combine all
        combined_conv_features = torch.cat(
            (conv_features, conv_features_dwi, diff_features), dim=1
        )

        outputs = self.classifier(combined_conv_features)
        return outputs

##################################################################################################
## TRAINING FUNCTION

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    epoch_losses = []
    epoch_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for batch in train_loader:
            if batch is None:
                continue

            dwi_images = batch['dwi'].to(device)
            t2_images = batch['t2'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(dwi_images, t2_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * dwi_images.size(0)
            total_samples += dwi_images.size(0)

            true_labels = torch.argmax(labels, dim=1)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            running_corrects += (preds == true_labels).sum().item()

        train_loss = running_loss / total_samples
        train_acc = running_corrects / total_samples

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0

        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                dwi_images = batch['dwi'].to(device)
                t2_images = batch['t2'].to(device)
                labels = batch['label'].to(device)

                outputs = model(dwi_images, t2_images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * dwi_images.size(0)
                val_samples += dwi_images.size(0)

                true_labels = torch.argmax(labels, dim=1)
                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                val_corrects += (preds == true_labels).sum().item()

                all_val_labels.append(labels.cpu().numpy())
                all_val_preds.append(torch.softmax(outputs, dim=1).cpu().numpy())

        val_loss /= val_samples
        val_acc = val_corrects / val_samples

        # AUC Calculation
        all_val_labels = np.vstack(all_val_labels)
        all_val_preds = np.vstack(all_val_preds)
        val_hard_labels = np.argmax(all_val_labels, axis=1)

        try:
            auc_0 = roc_auc_score((val_hard_labels == 0).astype(int), all_val_preds[:, 0])
        except:
            auc_0 = float('nan')

        try:
            auc_1 = roc_auc_score((val_hard_labels == 1).astype(int), all_val_preds[:, 1])
        except:
            auc_1 = float('nan')

        epoch_losses.append(train_loss)
        epoch_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch}/{num_epochs - 1} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"AUC0 (Diag): {auc_0:.4f}, AUC1 (NonDiag): {auc_1:.4f}")

        torch.save(model.state_dict(), os.path.join(output_folder_path, f"adc_model_epoch_{epoch}_2class.pth"))

    return epoch_losses, epoch_accuracies, val_losses, val_accuracies
##################################################################################################
## EVALUATION FUNCTION

def enable_mc_dropout(model):
    """
    Force dropout layers to stay active during inference
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def evaluate_model(model, test_loader, output_csv_file, mc_passes=30):
    model.eval()
    enable_mc_dropout(model)

    all_labels = []
    all_probs = []
    all_patient_ids = []

    correct_total = 0
    n_total = 0

    for batch in test_loader:
        if batch is None:
            continue

        dwi_images = batch['dwi'].to(device)
        t2_images = batch['t2'].to(device)
        labels = batch['label'].to(device)
        patient_ids = batch['patient_id']

        mc_outputs = []
        with torch.no_grad():
            for _ in range(mc_passes):
                logits = model(dwi_images, t2_images)
                probs = torch.softmax(logits, dim=1)
                mc_outputs.append(probs.unsqueeze(0))
        mc_outputs = torch.cat(mc_outputs, dim=0)
        mean_probs = mc_outputs.mean(dim=0).cpu().numpy()
        std_probs = mc_outputs.std(dim=0).cpu().numpy()

        true_labels = torch.argmax(labels, dim=1).cpu().numpy()
        preds = np.argmax(mean_probs, axis=1)
        correct_total += (preds == true_labels).sum()
        n_total += len(true_labels)

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(mean_probs)
        all_patient_ids.extend(patient_ids)

    test_accuracy = correct_total / n_total if n_total > 0 else 0
    print(f"Test Accuracy: {test_accuracy:.4f}")

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    hard_labels = np.argmax(all_labels, axis=1)

    # ROC
    roc_auc = {}
    fpr = {}
    tpr = {}
    for i in range(2):
        binary_labels = (hard_labels == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(binary_labels, all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f'Class {i} ROC-AUC Score: {roc_auc[i]:.4f}')

    # Plot ROC
    plt.figure()
    for i in range(2):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title('ROC Curve for Diagnostic vs Non-Diagnostic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(roc_curve_file)
    print(f"ROC curve saved at {roc_curve_file}")

    # Save predictions
    all_results = []
    for i in range(len(all_patient_ids)):
        probs = all_probs[i]
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        _, ci_lower, ci_upper = get_confidence_interval(mc_outputs[:, i, pred_class].cpu().numpy())
        result = {
            'patient_id': all_patient_ids[i],
            'Diagnostic_prob': probs[0],
            'NonDiagnostic_prob': probs[1],
            'Label_Diagnostic': all_labels[i, 0],
            'Label_NonDiagnostic': all_labels[i, 1],
            'Predicted_Label': 'diagnostic' if pred_class == 0 else 'non-diagnostic',
            'Confidence_in_Prediction': confidence,
            'Confidence_Lower': ci_lower,
            'Confidence_Upper': ci_upper
        }
        all_results.append(result)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_csv_file, index=False)
    print(f"Test results saved to {output_csv_file}")
    
def evaluate_model_simplified(
    model,
    test_loader,
    epoch_weights_path: str,
    output_csv_path: str = "model_evaluation_results.csv",
    roc_curve_path: str = "roc_auc_curve.png",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Loads model weights, performs inference (no MC dropout), 
    saves results to a CSV, and plots ROC-AUC curves for both classes.

    Parameters:
    - model: PyTorch model
    - test_loader: test DataLoader
    - epoch_weights_path: .pth file with trained weights
    - output_csv_path: path to save prediction CSV
    - roc_curve_path: path to save ROC curve image
    - device: torch device
    """
    print(f"Loading weights from {epoch_weights_path}")
    model.load_state_dict(torch.load(epoch_weights_path, map_location=device))
    model.to(device)
    model.eval()

    results = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            dwi_images = batch["dwi"].to(device)
            t2_images = batch["t2"].to(device)
            labels = batch["label"].cpu().numpy()  # soft labels
            patient_ids = batch["patient_id"]

            outputs = model(dwi_images, t2_images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            for i in range(len(patient_ids)):
                result = {
                    "Patient ID": patient_ids[i],
                    "Label_Diagnostic": labels[i][0],       # soft label for class 0
                    "Label_NonDiagnostic": labels[i][1],    # soft label for class 1
                    "Class 0 Probability": float(probs[i][0]),
                    "Class 1 Probability": float(probs[i][1]),
                }
                results.append(result)

            all_labels.extend(labels)
            all_probs.extend(probs)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to {output_csv_path}")

    # Convert soft labels to hard labels for ROC-AUC
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    hard_labels = np.argmax(all_labels, axis=1)

    # Compute ROC-AUC per class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i, class_name in zip([0, 1], ["Diagnostic", "Non-Diagnostic"]):
        binary_true_labels = (hard_labels == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(binary_true_labels, all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f"{class_name} ROC-AUC: {roc_auc[i]:.4f}")

    # Plot ROC curves
    plt.figure()
    plt.plot(fpr[0], tpr[0], label=f"Diagnostic (AUC = {roc_auc[0]:.3f})")
    plt.plot(fpr[1], tpr[1], label=f"Non-Diagnostic (AUC = {roc_auc[1]:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Diagnostic vs Non-Diagnostic")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(roc_curve_path)
    plt.show()
    print(f"ROC curve saved to {roc_curve_path}")

######################################################################################################
## Plot Epoch loss function 

def plot_epoch_loss(train_losses, train_accuracies, val_losses, val_accuracies, save_path):
    """
    Plots training and validation loss/accuracy over epochs.

    Args:
        train_losses (list): Training loss per epoch.
        train_accuracies (list): Training accuracy per epoch.
        val_losses (list): Validation loss per epoch.
        val_accuracies (list): Validation accuracy per epoch.
        save_path (str): File path to save the plot (e.g., accuracy_curve.tiff).
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='red')
    plt.plot(epochs, val_losses, label='Val Loss', color='orange')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='green')
    plt.plot(epochs, val_accuracies, label='Val Accuracy', color='blue')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Epoch loss and accuracy plot saved to {save_path}")


#################################################################################################
##Generate saliency mapping function
"""
def generate_saliency_maps(model, test_loader, saliency_map_folder, num_batches=4):
    model.eval()
    cam_extractor = GradCAM(model, target_layer='conv_after_concat')  # Adjust target layer
    os.makedirs(saliency_map_folder, exist_ok=True)
    
    batch_count = 0
    for batch in test_loader:
        if batch is None or batch_count >= num_batches:
            continue
        
        dwi_images = batch['dwi'].to(device)
        t2_images = batch['t2'].to(device)
        patient_ids = batch['patient_id']
        
        for i in range(dwi_images.size(0)):
            dwi = dwi_images[i:i+1]
            t2 = t2_images[i:i+1]
            outputs = model(dwi, t2)
            pred_class = torch.argmax(outputs, dim=1).item()
            
            # Compute saliency map
            cam = cam_extractor(pred_class, outputs)
            cam = cam[0].cpu().numpy()  # [C, D, H, W]
            
            # Save or visualize
            np.save(os.path.join(saliency_map_folder, f"{patient_ids[i]}_cam.npy"), cam)
        
        batch_count += 1
    print(f"Saliency maps saved to {saliency_map_folder}")
"""
    
def get_confidence_interval(prob_array, confidence=0.95):
    if len(prob_array) < 2:
        return np.mean(prob_array), np.mean(prob_array), np.mean(prob_array)
    mean = np.mean(prob_array)
    sem = st.sem(prob_array)
    h = sem * st.t.ppf((1 + confidence) / 2., len(prob_array)-1)
    return mean, mean - h, mean + h
##################################################################################################
## INSTANTIATION AND TRAINING

model = DualInputDenseNet121(out_channels=num_prediction_classes).to(device)
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights  # Tensor of shape [2]

    def forward(self, logits, soft_labels):
        log_probs = torch.log_softmax(logits, dim=1) + 1e-10  # Numerical stability
        weighted_targets = self.class_weights.unsqueeze(0) * soft_labels  # [B, 2]
        loss = - (weighted_targets * log_probs).sum(dim=1)  # [B]
        # Normalize by the sum of effective weights per sample
        weight_sums = (self.class_weights.unsqueeze(0) * soft_labels).sum(dim=1)  # [B]
        loss = loss / (weight_sums + 1e-10)  # [B]
        return loss.mean()  # Average over batch

# Apply heavier penalty to class 0 (non-diagnostic)
criterion = WeightedCrossEntropyLoss(class_weights=torch.tensor([1.0, 3.0], device=device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

##epoch_losses, epoch_accuracies, val_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=125)
##torch.save(model.state_dict(), adc_model_save_path.replace(".pth", "_2class.pth"))
##print(f"Final model saved to {adc_model_save_path}")
# Load saved model checkpoint
checkpoint_path = r"/home/steckohj/image_quality_project/2025-05-01_13:29:55_adc_model/adc_model/adc_model_epoch_99_2class.pth"
model = DualInputDenseNet121(out_channels=num_prediction_classes).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Run evaluation
evaluate_model_simplified(
    model=model,
    test_loader=test_loader,
    epoch_weights_path=checkpoint_path,  # Already defined above
    output_csv_path=output_csv_file.replace(".csv", "_simplified.csv"),
    roc_curve_path=roc_curve_file.replace(".tiff", "_simplified.png"),
    device=device
)

##plot_epoch_loss(epoch_losses, epoch_accuracies, val_losses, val_accuracies, accuracy_plot_file)


