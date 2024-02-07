from utils import CustomDataset, CNN, get_data, get_transformation
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

config = json.load(open("config.json", "r")) # loading the config for naming uploaded images
model_dir = os.path.basename("models")

MODEL_NAME = "model1"

torch.manual_seed(config["torch_seed"])

data = get_data(MODEL_NAME)

transform = get_transformation()

dataset = CustomDataset(data, transform=transform)

train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 64
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN()
model.load_state_dict(torch.load(os.path.join(model_dir, f"{MODEL_NAME}.pth")))
model.eval()
model.to(device)

total_number = 0
correct_number = 0

for inputs, labels in tqdm(val_dataloader, position=1, leave=False):
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    for pred, label in zip(preds.tolist(), labels.tolist()):
        total_number += 1
        correct_number += 1 if pred == label else 0

print(correct_number / total_number)