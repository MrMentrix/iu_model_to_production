import torch
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sqlite3
from utils import CNN, get_transformation
import json

# Defining CustomDataset for Predictions which also includes the image ID in the SQLite database
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label, image_id = self.data[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label, image_id

def make_predictions():
    config = json.load(open("config.json", "r")) # loading the config for naming uploaded images

    MODEL_NAME = config["deployed_model"]
    base_dir = os.path.basename("training_data_small")
    model_dir = os.path.basename("models")

    # selecting GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    le = LabelEncoder()
    le.classes_ = np.load(os.path.join(f"models/{MODEL_NAME}_labels.npy"), allow_pickle=True)

    model = CNN()
    model.load_state_dict(torch.load(os.path.join(model_dir, f"{MODEL_NAME}.pth")))
    model.eval()
    model.to(device)

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    rows = c.execute("SELECT * FROM images WHERE status = 'in progress'").fetchall()
    data = []
    for row in rows:
        id, path, status, prediction, confidence = row
        data.append((path, torch.empty(1), id))

    transform = get_transformation()

    batch_size = 64
    dataset = CustomDataset(data, transform=transform)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for inputs, labels, image_ids in val_dataloader:
        with torch.no_grad():
            outputs = model(inputs)
            confidences, indices = torch.topk(F.softmax(outputs, dim=1), k=1)
            for index, confidence, image_id in zip(indices.tolist(), confidences.tolist(), image_ids.tolist()):
                label = le.inverse_transform(index)[0]
                c.execute("UPDATE images SET status = 'predicted', classification = (?), confidence = (?) WHERE id = (?)", (label, round(confidence[0]*100, 2), image_id))

    conn.commit()
    conn.close()

make_predictions()