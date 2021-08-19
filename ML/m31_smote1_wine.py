from imblearn.over_sampling import SMOTE
from pandas.core.algorithms import value_counts
#! 데이터 증폭 (제일 많은 데이터에 맞춰 증폭해 준다.) - F1 스코어에서 효과를 얻을 수 있다.
from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

datasets = load_wine()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)     # (178, 13) (178,)

# print(pd.Series(y).value_counts())
#. value_counts()은 np에서는 안됨 pd해줘야함
# 1    71
# 0    59
# 2    48

# print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x_new = x[:-30]
y_new = y[:-30]

# print(x_new.shape, y_new.shape)     # (148, 13) (148,)
# print(pd.Series(y_new).value_counts())
# 1    71
# 0    59
# 2    18

x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, train_size=0.75, 
                                                    shuffle=True, random_state=11, stratify=y_new)
#^ 라벨별 일정한 비율별로 나오지 않음
#! stratify 각 라벨을 train_size 비율별로 나눠서 나옴

# print(pd.Series(y_train).value_counts())
# 1    53
# 0    44
# 2    14

model = XGBClassifier()

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print('model.score : ', score)

'''
2를 30개 삭제한 XGB 디폴트 모델
model.score :  0.9459459459459459
'''

########################################### smote 적용 ##########################################

print('################### smote 적용 ###################')

smote = SMOTE(random_state=66)

x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)
#! train 데이터만 증폭, test는 평가하는 거라 증폭할 필요가 없다.

# print(pd.Series(y_smote_train).value_counts())
# 0    53
# 1    53
# 2    53
#^ 제일 많은 라벨일 '1'과 같은 개수로 증폭해 준다.(전부 53개로 만든다.)

# print(x_smote_train.shape, y_smote_train.shape)     # (111, 13) (111,)  -> (159, 13) (159,)

model2 = XGBClassifier()

model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score2 = model2.score(x_test, y_test)
print('model2.score : ', score2)


print('smote 전 : ', x_train.shape, y_train.shape)
print('smote 후 : ', x_smote_train.shape, y_smote_train.shape)
print('smote전 레이블 값 분포 :\n', pd.Series(y_train).value_counts())
print('smote후 레이블 값 분포 :\n', pd.Series(y_smote_train).value_counts())

print('smote 전 model.score : ', score)
print('smote 후 model2.score : ', score2)



'''
#? train_size=0.75
#? random_state=66
smote 전 :  (111, 13) (111,)
smote 후 :  (159, 13) (159,)
smote전 레이블 값 분포 :
1    53
0    44
2    14
smote후 레이블 값 분포 :
0    53
1    53
2    53
smote 전 model.score :  0.9459459459459459
smote 후 model2.score :  0.972972972972973

#? train_size=0.8
#? random_state=66
smote 전 :  (118, 13) (118,)
smote 후 :  (171, 13) (171,)
smote전 레이블 값 분포 :
1    57
0    47
2    14
smote후 레이블 값 분포 :
0    57
1    57
2    57
smote 전 model.score :  0.9333333333333333
smote 후 model2.score :  0.9666666666666667

#? train_size=0.8
#? random_state=78
smote 전 :  (118, 13) (118,)
smote 후 :  (171, 13) (171,)
smote전 레이블 값 분포 :
1    57
0    47
2    14
smote후 레이블 값 분포 :
0    57
1    57
2    57
smote 전 model.score :  1.0
smote 후 model2.score :  1.0

#? train_size=0.75
#? random_state=78
smote 전 :  (111, 13) (111,)
smote 후 :  (159, 13) (159,)
smote전 레이블 값 분포 :
1    53
0    44
2    14
smote후 레이블 값 분포 :
0    53
1    53
2    53
smote 전 model.score :  0.972972972972973
smote 후 model2.score :  1.0
'''