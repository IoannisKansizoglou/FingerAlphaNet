import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from datasets.gesture_dataset import GestureDataset
from models.finger_alpha_net_m import FingerAlphaNetM

def visualize_cbam_attention(model, images, attention_map, indices):
    fig, axes = plt.subplots(len(indices), 3, figsize=(10, 3 * len(indices)))
    for row, idx in enumerate(indices):
        img = images[idx].numpy().astype(np.float32) / 255.0
        img = cv2.resize(img, (25, 25))
        att_map = attention_map[idx]
        att_map = cv2.resize(att_map, (25, 25))
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())

        axes[row, 0].imshow(img, cmap="gray"); axes[row, 0].set_title("Original")
        axes[row, 1].imshow(1 - att_map, cmap="jet"); axes[row, 1].set_title("Attention")
        axes[row, 2].imshow(img, cmap="gray"); axes[row, 2].imshow(1 - att_map, cmap="jet", alpha=0.55)
        axes[row, 2].set_title("Overlay")

        for c in range(3): axes[row, c].axis("off")
    plt.tight_layout()
    plt.show()

def main():
    data_path = "data/SignLanguageMNIST/"
    dataset = GestureDataset(f"{data_path}sign_mnist_test/sign_mnist_test.csv")
    images, _ = next(iter(torch.utils.data.DataLoader(dataset, batch_size=9)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FingerAlphaNetM().to(device)
    checkpoint = torch.load("checkpoints/FingerAlphaNetM_epoch20.pth.tar", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        _ = model(images.to(device))
        attention_map = model.cbam3.sa_map.squeeze().cpu().numpy()

    visualize_cbam_attention(model, images.squeeze(), attention_map, list(range(9)))

if __name__ == "__main__":
    main()

