import torch
import torch.nn as nn
import pandas as pd


HEADER = ["edge_index", "pos", "number", "smile", "z", "quadrupole", "octapole", "npacharge", "dipole", "polar", "hyperpolar", "energy", "Hij", "Hi", "dedipole", "depolar", "tran_dipole", "tran_energy"]


test_data_path = "/home/davido/Projects/ML/spectra_ml/data/test_10000.csv"

csv = pd.read_csv(test_data_path) 
print(csv.head())
print(csv.info()) 
print("Hier iloc", csv.iloc[0])
print("Hier len iloc", len(csv.iloc[0]))
