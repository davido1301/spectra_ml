import numpy as np
import torch
from models.model import SpectraModel
from data.loading import load_spectra_csv, df_to_list_target_int
from data.dataset import SpectraDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 


MODEL_PATH = "models/model_20260506_112451__148.pt"
TESTING_DATA = "/home/davido/Projects/ML/spectra_ml/data/testset_5000.csv"


model = SpectraModel() 

model.load_state_dict(torch.load(MODEL_PATH))
df = load_spectra_csv(TESTING_DATA)
smiles, spectras = df_to_list_target_int(df, "smile") 
testset = SpectraDataset(smiles = smiles, spectras = spectras) 
testset_loader = DataLoader(testset, batch_size=1, shuffle=False)

model.eval() 
sp_prediction = [] 
with torch.no_grad():
    for x, y in testset_loader:
        outputs = model(x)
        sp_prediction.append(outputs)

df["predictions"] = sp_prediction

y_true = np.asarray(spectras[0])
y_pred = sp_prediction[0].squeeze()
print("Shape of true", y_true.shape) 
print("shape of pred", y_pred.shape)

x = np.arange(len(y_true))

plt.plot(x, y_true, label="True")
plt.plot(x, y_pred, label="Prediction")

plt.legend()
plt.savefig("comp_lessepochs.png", dpi=300)
