import torch import from_smiles
import torch.nn as nn
import pandas as pd
import os
import sys
import numpy as np
from data.loading import df_to_list_target_int, load_spectra_csv
from data.dataset import SpectraDataset
import torch_geometric
# CSV DATA 
test_data_path = "/home/davido/Projects/ML/spectra_ml/data/test_10000.csv"


# Load Data into usable format for torch
df = load_spectra_csv(test_data_path)
smiles, spectras = df_to_list_target_int(df, "smile") 
data = SpectraDataset(smiles = smiles, spectras = spectras) 

print(data.__getitem__(0))


