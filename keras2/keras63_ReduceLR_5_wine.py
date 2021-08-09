import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import time


datasets = pd.read_csv('/study2/_data/winequality-white.csv', sep= ';',
                        index_col=None, header=0)

# print(datasets)
# print(datasets.shape)       # (4898, 12)
# print(datasets.info())
# print(datasets.describe())


# 1. 데이터
#^ 1. 판다스 -> 넘파이

datasets = datasets.to_numpy()
#datasets = datasets.values #도 가능


#^ 2. x와 y를 분리

# x = datasets.iloc[ : , :11]     # (4898, 11), df 데이터 나누기
# y = datasets.iloc[:, 11:]       # (4898, 1)

x = datasets[ : , :11]      
y = datasets[:, 11:]

# print(x.shape)       # (4898, 11)
# print(y.shape)      # (4898, 1)

#^ 3. y의 라벨을 확인 np.unique(y)

print(np.unique(y))     # [3. 4. 5. 6. 7. 8. 9.]

# 데이터 나누기

#y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=78)

ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()

#print(x_train.shape, x_test.shape)      # (3918, 11) (980, 11)

scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(y_train.shape, y_test.shape)        #(3918, 7) (980, 7)

# 2. 모델구성
model = Sequential()
model.add(Dense(2048, activation='relu', input_shape = (11,)))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(126, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))


# 3. 컴파일, 훈련
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
#                     filepath='/study/_save/ModelCheckPoint/keras48_5_MCP_wine.hdf5')

es = EarlyStopping(monitor= 'val_acc', patience=30, mode='auto', verbose=1)

reduce_lr = ReduceLROnPlateau(monitor= 'val_acc', patience=5, mode='auto', verbose=1, factor=0.5)


start_time = time.time()

hist = model.fit(x_train, y_train, epochs=5000, batch_size=10, callbacks=[es, reduce_lr], validation_split=0.2, verbose=2)

end_time = time.time() - start_time 


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

#y_predict = model.predict(x_test)

'''
걸린시간 :  134.3620195388794
loss :  1.0516475439071655
accuracy :  0.5581632852554321
'''