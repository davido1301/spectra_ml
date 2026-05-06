from torch.utils.data import Dataset
import torch 
from data.processing import from_smiles_to_fingerprint

class SpectraDataset(Dataset):
    def __init__(self, smiles, spectras):
        self.smiles = smiles
        self.spectras = spectras 

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        spectra = self.spectras[idx]
        smile = from_smiles_to_fingerprint(self.smiles[idx])
        smile = torch.tensor(smile, dtype=torch.float32)
        spectra = torch.tensor(spectra, dtype=torch.float32)
        return smile, spectra

