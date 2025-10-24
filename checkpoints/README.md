# 🏁 Checkpoints

This folder contains pretrained models for **FingerAlphaNet** and **FingerAlphaNet.m**. You can use these checkpoints to:

- Resume training from a specific epoch  
- Evaluate the models on the test set  
- Generate CBAM attention visualizations  

---

## 📦 Available Checkpoints

| Model | Epoch | Description | Download Link |
|-------|-------|-------------|---------------|
| FingerAlphaNet | 20 | Classic CNN with CBAM | [Download](https://drive.google.com/your_classic_model_link) |
| FingerAlphaNet.m | 20 | Separable CNN with CBAM (compressed) | [Download](https://drive.google.com/your_separable_model_link) |

---

## ⚡ How to Use

1. Download the checkpoint(s) from the provided links.  
2. Place the `.pth.tar` files inside this folder:  

```text
checkpoints/
├── FingerAlphaNet_epoch20.pth.tar
└── FingerAlphaNetM_epoch20.pth.tar
```

3. Load the checkpoint in your scripts:

```python
import torch
from models.finger_alpha_net import CBAMClassifier
from models.finger_alpha_net_m import CBAMClassifierCompressed

# Classic CNN
model = CBAMClassifier()
checkpoint = torch.load("checkpoints/FingerAlphaNet_epoch20.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Separable CNN
model_m = CBAMClassifierCompressed()
checkpoint_m = torch.load("checkpoints/FingerAlphaNetM_epoch20.pth.tar")
model_m.load_state_dict(checkpoint_m['state_dict'])
model_m.eval()
```

## 📝 Notes

* Ensure the folder path is correct when loading checkpoints.
* These models were trained on Sign Language MNIST (23 classes).
* You can resume training by providing the checkpoint path to the training scripts.
