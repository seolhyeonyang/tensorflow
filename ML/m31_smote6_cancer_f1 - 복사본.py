from imblearn.over_sampling import SMOTE
from pandas.core.algorithms import value_counts
from sklearn.datasets import load_breast_cancer
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, f1_score


datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(pd.Series(y).value_counts())
# 1    357
# 0    212

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                                    shuffle=True, random_state=78, stratify=y)

print(pd.Series(y_train).value_counts())


model = XGBClassifier()

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print('model.score : ', score)

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred)
print('f1_score : ', f1)


########################################### smote 적용 ##########################################

print('################### smote 적용 ###################')

smote = SMOTE(random_state=78)

x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

# print(pd.Series(y_smote_train).value_counts())
# 0    53
# 1    53
# 2    53

# print(x_smote_train.shape, y_smote_train.shape)     # (111, 13) (111,)  -> (159, 13) (159,)

model2 = XGBClassifier()

model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score2 = model2.score(x_test, y_test)
print('model2.score : ', score2)

y_pred = model2.predict(x_test)
f1_2 = f1_score(y_test, y_pred)
print('f1_score : ', f1_2)


print('smote 전 : ', x_train.shape, y_train.shape)
print('smote 후 : ', x_smote_train.shape, y_smote_train.shape)
print('smote전 레이블 값 분포 :\n', pd.Series(y_train).value_counts())
print('smote후 레이블 값 분포 :\n', pd.Series(y_smote_train).value_counts())

print('smote 전 model.score : ', score)
print('smote 후 model2.score : ', score2)

print('smote 전 f1_score : ', f1)
print('smote 후 f1_score : ', f1_2)

'''
smote 전 :  (455, 30) (455,)
smote 후 :  (570, 30) (570,)
smote전 레이블 값 분포 :
1    285
0    170
dtype: int64
smote후 레이블 값 분포 :
0    285
1    285
dtype: int64
smote 전 model.score :  0.9824561403508771
smote 후 model2.score :  0.9824561403508771
smote 전 f1_score :  0.9861111111111112
smote 후 f1_score :  0.9861111111111112
'''
