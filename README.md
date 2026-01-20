# Smart Stress Model: ECG-based Stress Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project focused on detecting human stress levels using electrocardiogram (ECG) data from the **WESAD** (Wearable Stress and Affect Detection) dataset. This repository implements both standard Deep Neural Networks (DNN) and Self-Attention augmented architectures.

---

## 🌟 Features
- **Raw Data Processing**: Automated extraction of ECG features (Heart Rate, HRV, TINN, etc.) from the WESAD dataset.
- **Advanced Architectures**:
  - **Standard DNN**: Multi-layer perceptron for robust feature classification.
  - **Attention-based DNN**: Integrated Self-Attention mechanism to weigh relevant ECG characteristics dynamically.
- **Cross-Validation**: Leave-One-Subject-Out (LOSO) and K-Fold cross-validation strategies.
- **Comprehensive Evaluation**: Automated generation of Confusion Matrices and metrics (Accuracy, Precision, Recall, F1).
- **GPU Optimized**: Full support for CUDA acceleration for training and inference.

---

## 📂 Project Structure
| File | Description |
| :--- | :--- |
| `Data_preprocessing.py` | Extracts features from raw WESAD `.pkl` files and saves them as JSON. |
| `Model-training.py` | Main training pipeline for the standard DNN model. |
| `Model-training-attention.py` | Training pipeline for the Attention-augmented model. |
| `Model_testing.py` | Evaluation script for standard DNN models. |
| `Model_testing-attention.py` | Evaluation script for Attention-based models. |
| `test_cuda.py` | Utility script to verify PyTorch/CUDA environment setup. |
| `.env.example` | Template for configuring local data and model directories. |

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+
- PyTorch (with CUDA support recommended)
- Dependencies: `pip install -r requirements.txt` (including `python-dotenv`, `heartpy`, `scipy`, `pandas`, `seaborn`, `tqdm`)

### 2. Configuration
Create a `.env` file in the root directory and configure your local paths:
```env
DIR_WESAD=D:/Path/To/WESAD/
DIR_DATA=D:/Path/To/Processed_Data/
DIR_NET_SAVING=D:/Path/To/Models/
DIR_RESULTS=D:/Path/To/Results/
```

### 3. Usage Pipeline
1. **Preprocess Data**:
   ```bash
   python Data_preprocessing.py
   ```
2. **Train Model**:
   ```bash
   python Model-training-attention.py
   ```
3. **Run Inference**:
   ```bash
   python Model_testing-attention.py
   ```

---

## 📊 Results
The models classify four emotional states:
- **0 - Neutral**
- **1 - Stress**
- **2 - Amusement**
- **3 - Relax**

Performance metrics and confusion matrices are automatically saved to the directory specified in `DIR_RESULTS`.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License
This project is licensed under the MIT License - see the `LICENSE` file for details.
