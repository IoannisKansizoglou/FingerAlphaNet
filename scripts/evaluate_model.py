import torch
from datasets.gesture_dataset import GestureDataset
from models.finger_alpha_net import FingerAlphaNet
from utils.train_utils import validate

def main(checkpoint_path):
    data_path = "data/SignLanguageMNIST/"
    val_data = GestureDataset(f"{data_path}sign_mnist_test/sign_mnist_test.csv")
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FingerAlphaNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    val_acc, val_f1 = validate(val_loader, model, device)
    print(f"Validation Accuracy: {val_acc:.4f}, F1-score: {val_f1:.4f}")

if __name__ == "__main__":
    main("checkpoints/FingerAlphaNet_epoch20.pth.tar")

