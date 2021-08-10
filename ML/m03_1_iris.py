import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


#! 다중 분류 문제

datasets = load_iris()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)     # (150, 4) (150,)
# print(y)


# 1. 데이터

from tensorflow.keras.utils import to_categorical

#y = to_categorical(y)       # 원핫인코딩 한것이다.

# print(y[:5])            # [[1. 0. 0.], [1. 0. 0.], [1. 0. 0.], [1. 0. 0.], [1. 0. 0.]]
# print(y.shape)          # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=88)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#! LogisticRegression 분류 모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()

'''
#? train_size=0.8, random_state=71
model.score :  0.9333333333333333
accuracy_score :  0.9333333333333333

model.score :  0.9666666666666667
accuracy_score :  0.9666666666666667

model.score :  1.0
accuracy_score :  1.0

model.score :  0.9
accuracy_score :  0.9

model.score :  0.9333333333333333
accuracy_score :  0.9333333333333333

model.score :  0.9666666666666667
accuracy_score :  0.9666666666666667


#? train_size=0.9, random_state=88
model.score :  0.9333333333333333
accuracy_score :  0.9333333333333333

model.score :  1.0
accuracy_score :  1.0

model.score :  0.9333333333333333
accuracy_score :  0.9333333333333333

model.score :  0.8666666666666667
accuracy_score :  0.8666666666666667

model.score :  1.0
accuracy_score :  1.0

model.score :  1.0
accuracy_score :  1.0

#? train_size=0.7, random_state=88
model.score :  0.9333333333333333
accuracy_score :  0.9333333333333333

model.score :  0.9555555555555556
accuracy_score :  0.9555555555555556

model.score :  0.9333333333333333
accuracy_score :  0.9333333333333333

model.score :  0.9111111111111111
accuracy_score :  0.9111111111111111

model.score :  0.8888888888888888
accuracy_score :  0.8888888888888888

model.score :  0.8888888888888888
accuracy_score :  0.8888888888888888
'''

# model = Sequential()
# model.add(Dense(256, activation='relu', input_shape = (4,)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(3, activation='softmax'))       




# 3. 컴파일, 훈련
model.fit(x_train, y_train)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# es = EarlyStopping(monitor = 'loss', patience=20, mode='min', verbose=1)

# hist = model.fit(x_train, y_train, epochs=1000, batch_size=3, callbacks=[es])


# 4. 평가, 예측
results = model.score(x_test, y_test)
print('model.score : ', results)      #  0.9333333333333333

# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss[0])
# print('accuracy : ', loss[1])


from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)


y_predict2 = model.predict(x_test[:5])
print(y_predict2)
print(y_test[:5])



# loss :  0.0
# accuracy :  0.5

# 원핫인코딩 후
# loss :  0.07892175018787384
# accuracy :  0.9666666388511658
