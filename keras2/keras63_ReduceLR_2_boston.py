import numpy as np
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import time


# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=78)

scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_dim = 13))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))


# 3. 컴파일, 훈련
optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer)

es = EarlyStopping(monitor= 'val_loss', patience=20, mode='min', verbose=1)

# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
#                     filepath='/study/_save/ModelCheckPoint/keras48_1_MCP_boston.hdf5')

reduce_lr = ReduceLROnPlateau(monitor= 'val_loss', patience=3, mode='auto', verbose=1, factor=0.5)

start_time = time.time()

model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=2, validation_split=0.2, callbacks=[es, reduce_lr])

end_time = time.time() - start_time

# model.save('/study/_save/ModelCheckPoint/keras48_1_save_model_boston.h5')


#model = load_model('/study/_save/ModelCheckPoint/keras48_1_save_model_boston.h5')

# model = load_model('/study/_save/ModelCheckPoint/keras48_1_MCP_boston.hdf5')


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('=' * 25)
print('걸린시간 : ', end_time)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)



'''
optimizer = Adam(lr=0.001)
걸린시간 :  39.355631828308105
loss :  8.885357856750488
r2 :  0.889191279033817
'''