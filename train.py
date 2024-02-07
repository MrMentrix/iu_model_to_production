import torch
import numpy as np
import os
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import CustomDataset, CNN, get_data, get_transformation
import json

config = json.load(open("config.json", "r")) # loading the config for naming uploaded images

torch.manual_seed(config["torch_seed"])

# under this model name, the trained model will be stored in the /models directory
MODEL_NAME = config["training_model"]
TRAIN_TEST = config["train_test_ratio"]
EPOCHS = 5

# setting up paths
base_dir = os.path.basename("training_data_small")
image_dir = os.path.join(base_dir, "images")
model_dir = os.path.basename("models")
styles_path = os.path.join(base_dir, "styles.csv")

# loading the styles.csv file, skipping faulty data
df = pd.read_csv(styles_path, on_bad_lines="skip")

# setting up the label encoder
le = LabelEncoder()
encoded_labels = le.fit_transform(df["articleType"].unique())

# saving the label encoder 
np.save(f"{model_dir}/{MODEL_NAME}_labels.npy", le.classes_)

data = get_data(MODEL_NAME) # getting data using utils function

# generic transformation to ensure that all data in in the same format    
transform = get_transformation()

dataset = CustomDataset(data, transform=transform)

# define training and validation/"testing" data
train_size = int(TRAIN_TEST * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# prepare data for training
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# try selecting the GPU for training since it computes faster than a CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = CNN() # create model
model.to(device) # launch model on selected device

# set up loss and optimizer function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# start training process with predefined epochs
model.train()
for epoch in tqdm(range(EPOCHS), desc="outer", position=0): # tqdm for nicer iteration logging
    train_loss = 0.0
    train_correct = 0
    val_loss = 0.0
    val_correct = 0

    for inputs, labels in tqdm(train_dataloader, desc="inner", position=1, leave=False): # looping through the training dataset and training the model
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)
    train_loss /= len(train_dataset)
    train_acc = train_correct.double() / len(train_dataset)

# save the trained model for future use
torch.save(model.state_dict(), f"./models/{MODEL_NAME}.pth")