import torch
from sklearn.metrics import f1_score

def validate(val_loader, model, device="cuda"):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for images, lbls in val_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(lbls.cpu().numpy())

    preds, labels = torch.tensor(preds), torch.tensor(labels)
    accuracy = (preds == labels).sum().item() / len(labels)
    f1 = f1_score(labels, preds, average="weighted")
    model.train()
    return accuracy, f1


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, path)

