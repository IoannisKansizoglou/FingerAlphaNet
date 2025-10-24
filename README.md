# 🖐️ FingerAlphaNet

Deep Learning for **Sign Language Recognition** using two convolutional architectures with **CBAM (Convolutional Block Attention Module)** for attention-based interpretability.

This repository contains two main models:
- **FingerAlphaNet** — a classic convolutional neural network for Sign Language MNIST.
- **FingerAlphaNet.m** — a lightweight variant using **depthwise separable convolutions** (CBAMClassifierCompressed) for improved efficiency.

Both models incorporate **CBAM attention modules**, enabling the visualization of channel and spatial attention maps during inference.

---

## 🚀 Features

- **Classic and Separable Convolutions:** Trade-off between performance and computational efficiency.
- **CBAM Attention Mechanism:** Enhances focus on key spatial and channel features.
- **High Accuracy:** Trained on Sign Language MNIST (25 classes).
- **Modular Codebase:** Separate folders for datasets, models, utilities, and training scripts.
- **Visualization Tools:** Generate attention heatmaps and interpret model behavior.

---

## 📂 Project Structure

```markdown
FingerAlphaNet/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│ └── README.md
│
├── models/
│ ├── finger_alpha_net.py
│ ├── finger_alpha_net_m.py
│ ├── cbam_module.py
│ └── init.py
│
├── datasets/
│ ├── gesture_dataset.py
│ └── init.py
│
├── utils/
│ ├── metrics.py
│ ├── train_utils.py
│ ├── visualization.py
│ └── init.py
│
├── scripts/
│ ├── train_fingeralphanet.py
│ ├── train_fingeralphanet_m.py
│ ├── evaluate_model.py
│ └── visualize_attention.py
│
├── checkpoints/
  └── README.md
```

---

## 🧩 Models Overview

### **1️⃣ FingerAlphaNet**
A classic CNN model for sign language image classification.
- Convolutional layers with ReLU and BatchNorm  
- Max pooling for spatial downsampling  
- Fully connected layers for classification
- CBAM modules at multiple convolutional layers  

### **2️⃣ FingerAlphaNet.m (CBAMClassifierCompressed)**
An optimized variant using:
- **Depthwise Separable Convolutions**  
- **CBAM Attention Modules** at multiple levels  
- **Reduced parameters and memory footprint**  
- **Comparable accuracy to the classic version**

---

## 🧠 CBAM Attention Module

The **CBAM (Convolutional Block Attention Module)** combines **Channel** and **Spatial** attention:

| Channel Attention | Spatial Attention |
|--------------------|------------------|
| Learns *what* to focus on | Learns *where* to focus on |

Visualizations can be generated using:
```bash
python scripts/visualize_attention.py
```

## ⚙️ Installation
1. Clone the Repository
```bash
git clone https://github.com/yourusername/FingerAlphaNet.git
cd FingerAlphaNet
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```
## 📊 Dataset

The project uses the Sign Language MNIST dataset from Kaggle.
You can download it manually from the link below:
https://www.kaggle.com/datasets/datamunge/sign-language-mnist

## 👉 Sign Language MNIST on Kaggle

After downloading, place the CSV files in:

```text
data/
├── sign_mnist_train.csv
└── sign_mnist_test.csv
```

## 🏋️ Training
Classic CNN (FingerAlphaNet)
```bash
python scripts/train_fingeralphanet.py --epochs 20 --batch-size 128 --lr 1e-3
```

Separable CNN (FingerAlphaNet.m)
```bash
python scripts/train_fingeralphanet_m.py --epochs 20 --batch-size 128 --lr 1e-3
```

Checkpoints will be saved under /checkpoints.

## 📈 Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate_model.py --checkpoint checkpoints/FingerAlphaNetM_epoch20.pth.tar
```

Metrics such as accuracy, F1-score, and confusion matrix will be displayed.

## 🔍 Visualization

To visualize CBAM attention maps on sample test images:

```bash
python scripts/visualize_attention.py
```

## 📦 Requirements
```text
torch>=2.0
torchvision
pandas
numpy
matplotlib
seaborn
opencv-python
scikit-learn
tqdm
```

## 🧾 Citation

If you use this repository, please cite:

```bibtex
@misc{FingerAlphaNet2025,
  title={FingerAlphaNet: Deep Sign Language Recognition with CBAM Attention},
  author={Your Name},
  year={2025},
  howpublished={GitHub repository},
  url={https://github.com/yourusername/FingerAlphaNet}
}
```
## 🧰 License

This project is licensed under the MIT License — see the LICENSE file for details.

## 🌟 Acknowledgements

* Dataset: Sign Language MNIST (Kaggle)
* CBAM: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
* PyTorch Open Source Community
