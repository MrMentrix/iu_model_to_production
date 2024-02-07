from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import tqdm
import os
import torch
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms
import numpy as np

config = json.load(open("config.json", "r")) # loading the config for naming uploaded images

base_dir = os.path.basename("training_data_small")
image_dir = os.path.join(base_dir, "images")
styles_path = os.path.join(base_dir, "styles.csv")

df = pd.read_csv(styles_path, on_bad_lines="skip")

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

class CNN(nn.Module):
    def __init__(self, num_classes=143):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(15*20*32, 1000)
        self.fc2 = nn.Linear(1000, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def get_data(model=config["training_model"]):
    data = []

    le = LabelEncoder()
    le.classes_ = np.load(os.path.join(f"models\{model}_labels.npy"), allow_pickle=True)

    for i, image in tqdm(enumerate(os.listdir(image_dir)), desc=f"Loading data ({len(os.listdir(image_dir))} images)"): # using tqdm for nicer iteration logging
        if i == 0:
            continue

        try:
            id = int(image.split(".")[0]) # get id from image name
            article_type = df.loc[df["id"] == id]["articleType"].item() # find articleType by id
            data.append((f"{os.path.join(image_dir, image)}", torch.tensor(le.transform([article_type])[0], dtype=torch.int64))) # append data with image path and articleType tensor
        except Exception as e:
            continue
    return data

def get_transformation():
    transform = transforms.Compose([
        transforms.Resize((80, 60)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform