import pandas as pd 
from pathlib import Path
import numpy as np


def load_spectra_csv(path: str) ->pd.DataFrame:
    """
    Read in csv and clean unnamed columns
    """
    path = Path(path) 
    if not path.exists(): 
        raise FileNotFoundError(f"File not found at: {path}")
    df = pd.read_csv(path) 
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")])
    return df


def df_to_list_target_int(
        df: pd.Dataframe,
        feature_cols: list[str],
) -> Tuple[List[list], Optional[List]]:
    """
        Convert Dataframe to columns into row-wise Python lists

    returns: 
        X_list: list of features Here: SMILES for now
        y_list: Convert from iloc[x, 3-704] # Intensities
    """ 
    X_list = df[feature_cols].values.tolist() 
    y_list = None
    y_list = df.iloc[:, 3:701].to_numpy().tolist() 
    

    return X_list, y_list




