import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys

DATA_PATH = "D:\\fingerPrintsOfCompounds.txt"
CHACHE_DIR = "D:\\"

def to_MACCS(smile):
    mol = Chem.MolFromSmiles(smile)
    fp = MACCSkeys.GenMACCSKeys(mol)
    fp_str = fp.ToBitString()
    return fp_str

def set_data(data_path):

    dtype = {"compoundID": np.int32, "fingerprint": np.str}
    reactions = pd.read_table(data_path, sep=' ', dtype=dtype, usecols=range(2))
    for i in range(len(reactions)):

        reactions.loc[i,'fingerprint'] = to_MACCS(reactions.loc[i,'fingerprint'])

    return reactions

reactions = set_data(DATA_PATH)

print(reactions)


reactions.to_csv("D:\\compoundsWithFingerprint.txt", sep=" ",index=False, header=None)





