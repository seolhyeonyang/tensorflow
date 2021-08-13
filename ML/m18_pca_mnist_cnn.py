import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Flatten, MaxPool2D
import time


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)



# PCA 적용
x = np.append(x_train, x_test, axis=0)
# print(x.shape)      # (70000, 28, 28)

x = x.reshape(70000, 28 * 28)

pca = PCA(n_components=625)

x = pca.fit_transform(x)
# print(x)
# print(x.shape)     # (442, 7)

pca_EVR = pca.explained_variance_ratio_

# print(pca_EVR)
# print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)     # 누적합 구하는 것
# print(cumsum)

# print(np.argmax(cumsum >=0.95)+1)     # 154

print(x.shape)      # (70000, 154)

x_train = x[:60000, :]
x_test = x[60000:, :]

print(x_train.shape)
print(x_test.shape)

# print(np.unique(y_train))       # [0 1 2 3 4 5 6 7 8 9]

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()

x_train = x_train.reshape(60000, 25, 25 ,1)   
x_test = x_test.reshape(10000, 25, 25 ,1)

# 2. 모델구성

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', input_shape=(x_train.shape[1],x_train.shape[2],1)))
model.add(Conv2D(5, (2,2), padding='same', activation='relu'))               
model.add(MaxPool2D())
model.add(Conv2D(5, (2,2), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
# model.add(MaxPool2D())                                                                
# model.add(Flatten())                                        
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.add(GlobalAveragePooling2D())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'val_acc', patience=30, mode='auto', verbose=1)

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=5000, batch_size=1000, callbacks=[es], validation_split=0.1, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


'''
기존 데이터
걸린 시간 :  9.789774894714355
loss :  0.11760503053665161
accuracy :  0.9761000275611877

PCA DNN
걸린 시간 :  43.74796986579895
loss :  0.14994606375694275
accuracy :  0.9599999785423279

PCA CNN(shape = 23,23,1)
걸린 시간 :  270.57553482055664
loss :  1.7736451625823975
accuracy :  0.3449999988079071

PCA CNN(shape = 25,25,1)
걸린 시간 :  178.63556718826294
loss :  1.8300591707229614
accuracy :  0.32600000500679016
'''