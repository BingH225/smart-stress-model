import os
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json
from dotenv import load_dotenv

from Model_testing import ClassifierECG, ds_wesad, extract_ds_from_dict, eq_dic, seed_worker, g

load_dotenv()

# Setup paths and environment
DIR_DATA = "Data_Processed"
DIR_NET_SAVING = "Models"
DIR_DATA_TEST = "Data_Processed"
DIR_RESULTS = "Results_SHAP"
SUBJECT_USED_FOR_TESTING = "S17"

if not os.path.exists(DIR_RESULTS):
    os.makedirs(DIR_RESULTS)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model setup
model = ClassifierECG(ngpu=1).to(device)

# Load best model weights (using existing model)
best_model_path = os.path.join(DIR_NET_SAVING, "net_1_0_epoch_26.pth")
try:
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded model from {best_model_path}")
except FileNotFoundError:
    print(f"Error: Could not find model at {best_model_path}. Please check the path and epoch number.")
    exit(1)
    
model.eval()

# Feature names exactly matching Data_preprocessing.py returns
feature_names = [
    'Mean Freq', 'Std Freq', 'TINN (T)', 'HRV Index', 
    'NN50', 'pNN50', 'Mean HRV', 'Std HRV', 
    'RMSSD', 'FFT Mean', 'FFT Std', 'Sum PSD'
]

# Load Data
print("Loading data...")
name_test = f'WESADECG_{SUBJECT_USED_FOR_TESTING}.json'
with open(os.path.join(DIR_DATA_TEST, name_test), 'r') as f:
    ds_test = ds_wesad(json.load(f))

dataloader_test = DataLoader(
    ds_test, batch_size=512, shuffle=True, 
    num_workers=0, worker_init_fn=seed_worker, generator=g, drop_last=False
)

# Fetch batch
batch = next(iter(dataloader_test))
features = batch[0].float().to(device)
labels = batch[1].float().to(device)

# 1. Background data for SHAP Explainer (e.g., 100 samples)
background_data = features[:100]

# 2. Test samples for SHAP explicitly (e.g., next 200 samples)
test_samples = features[100:300]
test_labels = labels[100:300]

print("Initializing SHAP DeepExplainer...")
# DeepExplainer is tailored for PyTorch neural networks
explainer = shap.DeepExplainer(model, background_data)

print("Calculating SHAP values...")
# Fetch SHAP values
# For PyTorch outputting a single value (sigmoid), shap_values will be a list with one item or an array
shap_values = explainer.shap_values(test_samples)

# Depending on shap version and output structure, shape might be [num_samples, num_features]
if isinstance(shap_values, list):
     shap_values_to_plot = shap_values[0]
else:
     shap_values_to_plot = shap_values
     
# Force to 2D for plotting
if len(shap_values_to_plot.shape) > 2:
    shap_values_to_plot = np.squeeze(shap_values_to_plot)

test_samples_np = test_samples.cpu().numpy()

print("Generating visualizations...")
# Set global style
plt.style.use('default')

# 1. Global Feature Importance (Summary Plot - Bar)
plt.figure()
shap.summary_plot(shap_values_to_plot, test_samples_np, feature_names=feature_names, plot_type="bar", show=False)
plt.title("SHAP Global Feature Importance")
plt.savefig(os.path.join(DIR_RESULTS, 'shap_summary_bar.png'), bbox_inches='tight', dpi=300)
plt.close()

# 2. Feature Influence Direction (Summary Plot - Beeswarm)
plt.figure()
shap.summary_plot(shap_values_to_plot, test_samples_np, feature_names=feature_names, show=False)
plt.title("SHAP Feature Influence Direction")
plt.savefig(os.path.join(DIR_RESULTS, 'shap_beeswarm_plot.png'), bbox_inches='tight', dpi=300)
plt.close()

# 3. Individual Case Studies
# Find one 'Relax' and one 'Stress' sample from our test_samples
stress_indices = torch.where(test_labels == 1)[0]
relax_indices = torch.where(test_labels == 0)[0]

# Convert for waterfall (needs shap.Explanation object in newer versions, or force_plot)
# We will use force_plot to save as HTML or static image (standard)

if len(relax_indices) > 0:
    idx_relax = relax_indices[0].item()
    print(f"Generating relax case study for index {idx_relax}...")
    
    # Waterfall plot
    plt.figure()
    # Create Explanation object required for waterfall
    exp_relax = shap.Explanation(values=shap_values_to_plot[idx_relax], 
                                 base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
                                 data=test_samples_np[idx_relax], 
                                 feature_names=feature_names)
    shap.waterfall_plot(exp_relax, show=False)
    plt.title("Local Explanation: Relaxed State")
    plt.savefig(os.path.join(DIR_RESULTS, 'shap_waterfall_relax.png'), bbox_inches='tight', dpi=300)
    plt.close()

if len(stress_indices) > 0:
    idx_stress = stress_indices[0].item()
    print(f"Generating stress case study for index {idx_stress}...")
    
    plt.figure()
    exp_stress = shap.Explanation(values=shap_values_to_plot[idx_stress], 
                                 base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
                                 data=test_samples_np[idx_stress], 
                                 feature_names=feature_names)
    shap.waterfall_plot(exp_stress, show=False)
    plt.title("Local Explanation: Stressed State")
    plt.savefig(os.path.join(DIR_RESULTS, 'shap_waterfall_stress.png'), bbox_inches='tight', dpi=300)
    plt.close()

print(f"SHAP analysis complete. All results saved in {DIR_RESULTS}.")

# Output mean absolute SHAP values for documentation
print("\n--- SHAP Feature Importances ---")
mean_abs_shap = np.abs(shap_values_to_plot).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1]
for i in sorted_idx:
    print(f"{feature_names[i]}: {mean_abs_shap[i]:.4f}")
