import torch 
import numpy as np 
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

def from_smiles_to_fingerprint(smile) -> np.array: 
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mol = Chem.MolFromSmiles(smile) 
    fp = fpgen.GetFingerprint(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype = np.int8)
    ConvertToNumpyArray(fp, arr)
    return arr

def smiles_list_to_nparr_list(smile_list: List[str]) -> List[np.ndarray]: 
    fp_list = []
    for smile in smile_list:
        fp_list.append(from_smiles_to_fingerprint(smile))
    return fp_list
