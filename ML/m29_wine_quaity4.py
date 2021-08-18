import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


datasets = pd.read_csv('/study2/_data/winequality-white.csv',
                        index_col=None, header=0, sep=';')

# print(datasets.head())
# print(datasets.shape)       # (4898, 12)
# print(datasets.describe())

# import matplotlib.pyplot as plt

# count_data = datasets.groupby('quality')['quality'].count()
# print(count_data)

# # count_data.plot()
# plt.bar(count_data.index, count_data)
# plt.show()

# exit()

datasets = datasets.values

# print(type(datasets))       # <class 'numpy.ndarray'>
# print(datasets.shape)       # (4898, 12)

x = datasets[:, :11]
y = datasets[:, 11]
# print(y.shape)       # (4898,)

#! 라벨 3, 4 -> 0 / 5, 6, 7 -> 1 / 8, 9 -> 2 수정
newlist = []
for i in list(y):
    if i <= 4 :
        newlist += [0]
    elif i <= 7 :
        newlist += [1]
    else:
        newlist += [2]
y = np.array(newlist)


# print(x.shape)       # (4898, 11)
# print(y.shape)       # (4898,)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=78)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
model = XGBClassifier()


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
score = model.score(x_test, y_test)

print('accuracy : ', score)
# accuracy :  0.6928571428571428

# 라벨 수정 후  accuracy :  0.9438775510204082