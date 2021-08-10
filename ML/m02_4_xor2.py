from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score


# 1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]
#! xor게이트 두개가 같으면 0, 다르면 1(둘 중 1개만 1일때 1)

# 2. 모델 
# model = LinearSVC()
model = SVC()

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
y_predict= model.predict(x_data)
print(x_data, '의 예측결과 : ', y_predict)

results = model.score(x_data, y_data)
print('model.score  : ', results)

acc = accuracy_score(y_data, y_predict)
print('accuracy_score : ', acc)

'''
model = LinearSVC()
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [0 1 1 1]
model.score  :  0.75
accuracy_score :  0.75

model = SVC()
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [0 1 1 0]
model.score  :  1.0
accuracy_score :  1.0
'''