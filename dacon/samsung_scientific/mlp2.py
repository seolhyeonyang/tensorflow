#fingerfrint name to use
ffpp = "pattern"

#read csv
import pandas as pd
train = pd.read_csv("/study2/dacon/samsung_scientific/_data/train.csv")
dev = pd.read_csv("/study2/dacon/samsung_scientific/_data/dev.csv")
test = pd.read_csv("/study2/dacon/samsung_scientific/_data/test.csv")

ss = pd.read_csv("/study2/dacon/samsung_scientific/_data/sample_submission.csv")

train = pd.concat([train,dev])

train['ST1_GAP(eV)'] = train['S1_energy(eV)'] - train['T1_energy(eV)']

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

import numpy as np
import pandas as pd

import math
train_fps = []#train fingerprints
train_y = [] #train y(label)

for index, row in train.iterrows() : 
    try : 
        mol = Chem.MolFromSmiles(row['SMILES'])
        if ffpp == 'maccs' :    
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif ffpp == 'morgan' : 
            fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
        elif ffpp == 'rdkit' : 
            fp = Chem.RDKFingerprint(mol)
        elif ffpp == 'pattern' : 
            fp = Chem.rdmolops.PatternFingerprint(mol)
        elif ffpp == 'layerd' : 
            fp = Chem.rdmolops.LayeredFingerprint(mol)

        train_fps.append(fp)
        train_y.append(row['ST1_GAP(eV)'])
    except : 
        pass

#fingerfrint object to ndarray
np_train_fps = []
for fp in train_fps:
    arr = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    np_train_fps.append(arr)

np_train_fps_array = np.array(np_train_fps)

print(np_train_fps_array.shape)
print(len(train_y))

pd.Series(np_train_fps_array[:,0]).value_counts()

import math
test_fps = []#test fingerprints
test_y = [] #test y(label)

for index, row in test.iterrows() : 
    try : 
        mol = Chem.MolFromSmiles(row['SMILES'])

        if ffpp == 'maccs' :    
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif ffpp == 'morgan' : 
            fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
        elif ffpp == 'rdkit' : 
            fp = Chem.RDKFingerprint(mol)
        elif ffpp == 'pattern' : 
            fp = Chem.rdmolops.PatternFingerprint(mol)
        elif ffpp == 'layerd' : 
            fp = Chem.rdmolops.LayeredFingerprint(mol)

        test_fps.append(fp)
        test_y.append(row['ST1_GAP(eV)'])
    except : 
        pass

np_test_fps = []
for fp in test_fps:
    arr = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    np_test_fps.append(arr)

np_test_fps_array = np.array(np_test_fps)

print(np_test_fps_array.shape)
print(len(test_y))

pd.Series(np_test_fps_array[:,0]).value_counts()

print(np_test_fps_array.shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def create_deep_learning_model():
    model = Sequential()
    model.add(Dense(2048, input_dim=2048, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=0.01))
    return model


# def create_deep_learning_model(drop=0.5, learning_rate=0.001, activation='relu', node = 512):
#     inputs = Input(shape=(2048), name='input')
#     x = Dense(node,  kernel_initializer='normal', name='hidden1')(inputs)
#     x= Dropout(drop)(x)
#     x = Dense(node/2, activation=activation, name='hidden2')(x)
#     x= Dropout(drop)(x)
#     x = Dense(node/4, activation=activation, name='hidden3')(x)
#     x= Dropout(drop)(x)
#     outputs = Dense(1,  kernel_initializer='normal',name='output')(x)
#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer=Adam(learning_rate),
#                 loss='mean_absolute_error')
#     return model


X, Y = np_train_fps_array , np.array(train_y)

#validation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

model2 = KerasClassifier(build_fn=create_deep_learning_model, verbose=1)#, epochs=2, validation_split=0.2)

estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=create_deep_learning_model, epochs=10)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=5)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("%.2f (%.2f) MAE" % (results.mean(), results.std()))

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, shuffle=True, random_state=9)

def create_hyperparameter():
    batches = [10, 20, 30, 40, 50]
    # optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.3, 0.4, 0.8]
    activation = ['selu', 'relu']
    node = [512, 1024, 2048]
    learning_rate = [0.1, 0.01, 0.001]
    return {'batch_size' : batches,
            'drop' : dropout,
            'activation' : activation,
            'node' : node,
            'learning_rate' : learning_rate}

hyperparamters = create_hyperparameter()

model = create_deep_learning_model()
es = EarlyStopping(monitor='loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
# model = RandomizedSearchCV(model2, hyperparamters ,cv=4)
model.fit(X, Y, epochs = 10000, callbacks=[es])
# model = XGBRegressor(n_estimators=2000, learning_rate=0.05, n_jobs=-1)
# model.fit(x_train, y_train, verbose=1, eval_metric=['rmse', 'mae'], eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=10)

# print(model.best_params_)
# print(model.best_estimator_)
# print(model.best_score_)

test_y = model.predict(np_test_fps_array)
ss['ST1_GAP(eV)'] = test_y


ss.to_csv("/study2/dacon/samsung_scientific/_save/mlp2.csv",index=False)