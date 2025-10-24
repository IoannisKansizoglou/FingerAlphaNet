import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2

class GestureDataset(Dataset):
    """Custom dataset for Sign Language MNIST."""

    def __init__(self, csv_path, train=True, img_size=224):
        self.csv = pd.read_csv(csv_path)
        self.train = train
        self.img_size = img_size

        text = "pixel"
        self.images = torch.zeros((self.csv.shape[0], 1))
        for i in range(1, 785):
            temp = torch.FloatTensor(self.csv[f"{text}{i}"]).unsqueeze(1)
            self.images = torch.cat((self.images, temp), 1)
        self.images = self.images[:, 1:].view(-1, 28, 28)
        self.labels = self.csv["label"]

    def __getitem__(self, idx):
        img = self.images[idx].numpy()
        img = cv2.resize(img, (self.img_size, self.img_size))
        tensor_img = torch.FloatTensor(img).unsqueeze(0) / 255.
        if self.train:
            return tensor_img, self.labels[idx]
        return tensor_img

    def __len__(self):
        return self.images.shape[0]

