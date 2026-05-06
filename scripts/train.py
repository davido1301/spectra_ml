from data.processing import smiles_list_to_nparr_list
from torch_geometric.utils import from_smiles
import torch.nn as nn
import pandas as pd
import os
import sys
import numpy as np
from data.loading import df_to_list_target_int, load_spectra_csv
from data.dataset import SpectraDataset
import torch_geometric
import torch 
import torchvision
from datetime import datetime
from models.model import SpectraModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# CSV DATA 
test_data_path = "/home/davido/Projects/ML/spectra_ml/data/test_10000.csv"

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0 
    last_loss = 0 
    for i, data in enumerate(train_dataloader):
        inputs, labels = data 
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels) 
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item() 
        if i % 1000 == 999:
            last_loss = running_loss / 1000 
            print(" batch {} loss : {}".format(i+1, last_loss)) 
            tb_x = epoch_index * len(training_loader) +i +1 
            tb_writer.add_scalar("Loss/train", last_loss, tb_x) 
            running_loss = 0 
    return last_loss 



# Load Data into usable format for torch
df = load_spectra_csv(test_data_path)
smiles, spectras = df_to_list_target_int(df, "smile") 
data = SpectraDataset(smiles = smiles, spectras = spectras) 



train, test = torch.utils.data.random_split(data, [0.8,0.2])


# params model
model = SpectraModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 
loss_fn = torch.nn.MSELoss()

# data loader
train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test, batch_size=64, shuffle=True) 


# actual training loop 

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter("runs/spectra_{}".format(timestamp))
epoch_number = 0 
EPOCHS = 150 

best_vloss = 1_000_000
best_model_path = None
for epoch in range(EPOCHS):
    print("Epoch {}:".format(epoch_number+1))

    # grad tracking on pass over the data
    model.train(True) 
    avg_loss = train_one_epoch(epoch_number, writer) 
    running_vloss = 0.0 
    model.eval() 

    with torch.no_grad():
        for i, vdata in enumerate(test_dataloader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss 

    avg_vloss = running_vloss / (i+1)
    print("Loss Train {} valid train {}".format(avg_loss, avg_vloss))

    # logging 
    writer.add_scalar("Training", avg_loss)
    writer.add_scalar("Validation", avg_vloss)
    writer.add_scalar("epoch", epoch_number+1)
    writer.flush() 
    
    # track best performing model
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss 
        if best_model_path and os.path.exists(best_model_path):
            os.remove(best_model_path)
        best_model_path = "models/model_{}__{}.pt".format(timestamp, epoch_number)
        torch.save(model.state_dict(), best_model_path) 
    epoch_number +=1 
