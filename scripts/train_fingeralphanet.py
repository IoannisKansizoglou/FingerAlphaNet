import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.gesture_dataset import GestureDataset
from models.finger_alpha_net import FingerAlphaNet
from utils.train_utils import validate, save_checkpoint

def main():
    data_path = "data/SignLanguageMNIST/"
    train_data = GestureDataset(f"{data_path}sign_mnist_train/sign_mnist_train.csv")
    val_data = GestureDataset(f"{data_path}sign_mnist_test/sign_mnist_test.csv")

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FingerAlphaNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    for epoch in range(1, 21):
        model.train()
        for images, labels in train_loader:
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_acc, val_f1 = validate(val_loader, model, device)
        scheduler.step(val_acc)
        print(f"Epoch {epoch:02d} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        save_checkpoint(model, optimizer, epoch, f"checkpoints/FingerAlphaNet_epoch{epoch}.pth.tar")

if __name__ == "__main__":
    main()

