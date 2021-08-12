# m31로 만든 0.999 이상의 n_component = ? 를 사용하여
#  xgb 모델을 만든것 (디폴트)

# mnist dnn 보다 성능 좋게 만들어라
# dnn, cnn과 비교

# RandomSearch 로도 해볼것

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)

''' # ================ 기존 데이터
x_train = x_train.reshape(60000, 28 * 28)   
x_test = x_test.reshape(10000, 28 * 28)
# =========================== '''


# ================PCA 적용
x = np.append(x_train, x_test, axis=0)
# print(x.shape)      # (70000, 28, 28)

x = x.reshape(70000, 28 * 28)

pca = PCA(n_components=154)

x = pca.fit_transform(x)
# print(x)
# print(x.shape)     # (442, 7)

pca_EVR = pca.explained_variance_ratio_

# print(pca_EVR)
# print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)     # 누적합 구하는 것
# print(cumsum)

print(np.argmax(cumsum >=0.999)+1)     # 154

print(x.shape)      # (70000, 154)

x_train = x[:60000, :]
x_test = x[60000:, :]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# ===========================

# ================공통 적용
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ohe = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# ohe.fit(y_train)
# y_train = ohe.transform(y_train).toarray()
# y_test = ohe.transform(y_test).toarray()
# ===========================

''' # ================ DNN 모델
# 2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'val_acc', patience=30, mode='auto', verbose=1)

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=5000, batch_size=100, callbacks=[es], validation_split=0.1, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
# ====================================== '''

# ================ GridSearchCV
# 2. 모델구성
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {'n_estimators' : [100, 200, 300], 'learning_rate' : [0.1, 0.3, 0.001, 0.01], 'max_depth' : [4, 5, 6]},
    {'n_estimators' : [90, 100, 110], 'learning_rate' : [0.1, 0.001, 0.01],
    'max_depth' : [4, 5, 6], 'colsample_bytree' : [0.6, 0.9, 1]},
    {'n_estimators' : [90, 110], 'learning_rate' : [0.1, 0.001, 0.5], 'max_depth' : [4, 5, 6],
    'colsample_bytree' : [0.6, 0.9, 1], 'colsample_bylevel' : [0.6, 0.7, 0.9]},
]
n_jobs = -1

model = GridSearchCV(XGBClassifier(), parameters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 279 candidates, totalling 1395 fits

# model = XGBClassifier()

# 3. 훈련
start_time = time.time()

model.fit(x_train, y_train)
end_time = time.time() - start_time


# 4. 평가, 예측
print('최적의 매개변수 : ', model.best_estimator_)

print('best_score_ : ', model.best_score_)

print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))
print('걸린 시간 : ', end_time)

# =================================================



'''
DNN 모델
걸린 시간 :  128.92212629318237
loss :  0.25823506712913513
accuracy :  0.9801999926567078

XGB 모델(디폴트)
model.score :  0.9654
정답률 :  0.9654
걸린 시간 :  258.95328164100647

GridSearchCV_XGB 모델
'''
