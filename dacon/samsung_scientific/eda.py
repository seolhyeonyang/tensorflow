import numpy as np
import pandas as pd
from tqdm import tqdm
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
DrawingOptions.bondLineWidth=1.8
from rdkit.Chem.rdmolops import SanitizeFlags

# https://github.com/jensengroup/xyz2mol
# from xyz2mol import xyz2mol
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

path = '/content/drive/MyDrive/samsung/'
train = pd.read_csv(path + 'train.csv')
dev = pd.read_csv(path + 'dev.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

from rdkit import Chem 

#Method transforms smiles strings to mol rdkit object
train['mol'] = train['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
dev['mol'] = dev['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
test['mol'] = test['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))

from rdkit.Chem import Draw
mols = train['mol'][:20]

Draw.MolsToGridImage(mols, molsPerRow=5, useSVG=True, legends=list(train['SMILES'][:20].values))

train['mol'] = train['mol'].apply(lambda x: Chem.AddHs(x))
train['num_of_atoms'] = train['mol'].apply(lambda x: x.GetNumAtoms())
train['num_of_heavy_atoms'] = train['mol'].apply(lambda x: x.GetNumHeavyAtoms())

dev['mol'] = dev['mol'].apply(lambda x: Chem.AddHs(x))
dev['num_of_atoms'] = dev['mol'].apply(lambda x: x.GetNumAtoms())
dev['num_of_heavy_atoms'] = dev['mol'].apply(lambda x: x.GetNumHeavyAtoms())

test['mol'] = test['mol'].apply(lambda x: Chem.AddHs(x))
test['num_of_atoms'] = test['mol'].apply(lambda x: x.GetNumAtoms())
test['num_of_heavy_atoms'] = test['mol'].apply(lambda x: x.GetNumHeavyAtoms())

train['ST1_GAP(eV)'] = train['S1_energy(eV)'] - train['T1_energy(eV)']

import seaborn as sns
sns.jointplot(train.num_of_atoms, train['ST1_GAP(eV)'])
plt.show()

c_patt = Chem.MolFromSmiles('C')

print(train['mol'][0].GetSubstructMatches(c_patt))

def number_of_atoms(atom_list, df):
    for i in atom_list:
        df['num_of_{}_atoms'.format(i)] = df['mol'].apply(lambda x: len(x.GetSubstructMatches(Chem.MolFromSmiles(i))))

number_of_atoms(['C','O', 'N', 'Cl'], train)

sns.pairplot(train[['num_of_atoms','num_of_C_atoms','num_of_N_atoms', 'num_of_O_atoms', 'ST1_GAP(eV)']], diag_kind='kde', kind='reg', markers='+')
plt.show()

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

#Leave only features columns
train_df = train.drop(columns=['SMILES', 'mol', 'ST1_GAP(eV)', 'uid', 'S1_energy(eV)', 'T1_energy(eV)'])
y = train['ST1_GAP(eV)'].values

print(train_df.columns)
X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=.1, random_state=1)

from sklearn.metrics import mean_absolute_error, mean_squared_error
def evaluation(model, X_test, y_test):
    prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    
    plt.figure(figsize=(15, 10))
    plt.plot(prediction[:300], "red", label="prediction", linewidth=1.0)
    plt.plot(y_test[:300], 'green', label="actual", linewidth=1.0)
    plt.legend()
    plt.ylabel('logP')
    plt.title("MAE {}, MSE {}".format(round(mae, 4), round(mse, 4)))
    plt.show()
    
    print('MAE score:', round(mae, 4))
    print('MSE score:', round(mse,4))

ridge = RidgeCV(cv=5)
ridge.fit(X_train, y_train)
#Evaluate results
evaluation(ridge, X_test, y_test)

from rdkit.Chem import Descriptors
train['tpsa'] = train['mol'].apply(lambda x: Descriptors.TPSA(x))
train['mol_w'] = train['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
train['num_valence_electrons'] = train['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
train['num_heteroatoms'] = train['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))

train_df = train.drop(columns=['SMILES', 'mol', 'ST1_GAP(eV)', 'uid', 'S1_energy(eV)', 'T1_energy(eV)'])
y = train['ST1_GAP(eV)'].values

print(train_df.columns)

#Perform a train-test split. We'll use 10% of the data to evaluate the model while training on 90%

X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=.1, random_state=1)

ridge = RidgeCV(cv=5)
ridge.fit(X_train, y_train)
#Evaluate results and plot predictions
evaluation(ridge, X_test, y_test)