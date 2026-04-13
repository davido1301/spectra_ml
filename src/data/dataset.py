from torch.utils.data import Dataset
import torch 

class SpectraDataset(Dataset):
    def __init__(self, smiles, spectras):
        self.smiles = smiles
        self.spectras = spectras 

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        spectra = self.spectras[idx]
        return smile, spectra

