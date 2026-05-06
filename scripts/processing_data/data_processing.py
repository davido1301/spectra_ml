import pandas as pd
from pathlib import Path

from torch import _wrapped_quantized_linear_prepacked



"""
Getting QM9S to a dataset for training+testing

UV: smile + UV spectrum 

"""
HEAD = ["edge_index", "pos", "number", "smile", "z", "quadrupole", "octapole", "npacharge", "dipole", "polar", "hyperpolar", "energy", "Hij", "Hi", "dedipole", "depolar", "tran_dipole", "tran_energy"]

datapath = Path("/home/davido/Projects/data/QMS9/csv/")
using_col = ["smile"]

# for csv in sorted(root.rglob("*csv")):
  #  df = pd.read_csv(csv, header=HEAD)
datapath1 = datapath/"000001.csv" 

uv_data = datapath/"uv_boraden.csv"
uv = pd.read_csv(uv_data)

# print(uv.head())
# print(uv.shape)


csv_dir = "qm9s_csv"
datapath2 = Path("/home/davido/Projects/data/QMS9/csv/qm9s_csv/")


smiles = []
for file in datapath2.rglob("*.csv"):
    df = pd.read_csv(file,  header=0, names=["smile"], usecols=[3])
    smiles.append(df["smile"][0])

print(len(smiles))
print(len(uv))

uv["smile"] = smiles 

uv.to_csv("/home/davido/Projects/data/QMS9/csv/qm9s_csv/smile_uv.csv")
print("Done!")
# No idea why but i cant load in the stuff LOL 
