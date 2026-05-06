import torch
import torch.nn as nn
import pandas as pd
import os
import sys
import numpy as np
from data.loading import df_to_list_target_int
from data.dataset import SpectraDataset
from data.loading import load_spectra_csv
from torch_geometric.utils import from_smiles
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from data.processing import from_smiles_to_fingerprint, smiles_list_to_nparr_list

print("cwd:", os.getcwd())
print("sys.path[0]:", sys.path[0])



test_data_path = "/home/davido/Projects/ML/spectra_ml/data/test_10000.csv"

# csv = pd.read_csv(test_data_path)
csv = load_spectra_csv(test_data_path)
# csv = csv.drop(columns=[col for col in csv.columns if col.startswith("Unnamed")])


# 3 to 703 is intensities 
x, y = df_to_list_target_int(csv, "smile")
data = SpectraDataset(smiles = x, spectras = y)
# print(data.__getitem__(0))

print(x[0])
listi = smiles_list_to_nparr_list(x)
print(listi[0])
print(len(listi))



list2 = smiles_list_to_nparr_list(data.smiles)
print("list 2 ", list2[0])

train, test = torch.utils.data.random_split(data, [0.8, 0.2])

# print("here lol")
# rint(train[0])
# print(test[0])
