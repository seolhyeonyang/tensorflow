from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]
#! xor게이트 두개가 같으면 0, 다르면 1(둘 중 1개만 1일때 1)


# 2. 모델 
# model = LinearSVC()
# model = SVC()
model = Sequential()
model.add(Dense(32, input_dim=2, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_data, y_data, batch_size=1, epochs=100)

# 4. 평가, 예측
import tensorflow as tf

y_predict= model.predict(x_data)
y_predict = tf.round(y_predict)
print(x_data, '의 예측결과 : ', y_predict)

results = model.evaluate(x_data, y_data)
print('model.score  : ', results[1])

acc = accuracy_score(y_data, y_predict)
print('accuracy_score : ', acc)

'''
temp = tf.argmax(temp, axis=1)
model = LinearSVC()
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [0 1 1 1]
model.score  :  0.75
accuracy_score :  0.75

model = SVC()
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [0 1 1 0]
model.score  :  1.0
accuracy_score :  1.0
'''