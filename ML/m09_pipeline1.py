import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score


datasets = load_iris()

x = datasets.data
y = datasets.target


# 1. 데이터

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=88)


# 2. 모델구성
from sklearn.pipeline import make_pipeline, Pipeline

model = make_pipeline(MinMaxScaler(), SVC())
#! 엮을 수 있게 해주는 것(지금은 모델과 스케일링만 합친것)


# 3. 컴파일, 훈련
model.fit(x_train, y_train)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# es = EarlyStopping(monitor = 'loss', patience=20, mode='min', verbose=1)

# hist = model.fit(x_train, y_train, epochs=1000, batch_size=3, callbacks=[es])


# 4. 평가, 예측

print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))

'''
model.score :  0.9555555555555556
정답률 :  0.9555555555555556
'''