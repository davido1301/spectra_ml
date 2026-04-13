import torch
import torch.nn as nn
import pandas as pd
import os
import sys
import numpy as np
from data.loading import df_to_list_target_int
from data.dataset import SpectraDataset
from data.loading import load_spectra_csv
print("cwd:", os.getcwd())
print("sys.path[0]:", sys.path[0])



test_data_path = "/home/davido/Projects/ML/spectra_ml/data/test_10000.csv"

# csv = pd.read_csv(test_data_path)
csv = load_spectra_csv(test_data_path)
# csv = csv.drop(columns=[col for col in csv.columns if col.startswith("Unnamed")])


# 3 to 703 is intensities 
x, y = df_to_list_target_int(csv, "smile")
data = SpectraDataset(smiles = x, spectras = y)
print(data.__getitem__(0))
