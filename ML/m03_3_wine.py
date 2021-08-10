from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from sklearn.metrics import accuracy_score



datasets = pd.read_csv('/study2/_data/winequality-white.csv', sep= ';',
                        index_col=None, header=0)


# 1. 데이터

datasets = datasets.to_numpy()
#datasets = datasets.values #도 가능

# x = datasets.iloc[ : , :11]     # (4898, 11), df 데이터 나누기
# y = datasets.iloc[:, 11:]       # (4898, 1)

x = datasets[ : , :11]      
y = datasets[:, 11:]

# print(x.shape)       # (4898, 11)
# print(y.shape)      # (4898, 1)

# print(np.unique(y))     # [3. 4. 5. 6. 7. 8. 9.]

# 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=78)


scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(y_train.shape, y_test.shape)        #(3918, 7) (980, 7)


# 2. 모델구성
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()


# 3. 컴파일, 훈련

start_time = time.time()

model.fit(x_train, y_train)

end_time = time.time() - start_time



# 4. 평가, 예측
results = model.score(x_test, y_test)
print('걸린 시간 : ', end_time)
print('model.score : ', results)


y_predict = model.predict(x_test[:50])


acc = accuracy_score(y_test[:50], y_predict)
print('accuracy_score : ', acc)

'''
#? train_size=0.8, random_state=78
걸린 시간 :  0.06379842758178711
model.score :  0.5163265306122449
accuracy_score :  0.48

걸린 시간 :  0.7275471687316895
model.score :  0.5469387755102041
accuracy_score :  0.48

걸린 시간 :  0.00897359848022461
model.score :  0.5642857142857143
accuracy_score :  0.56

걸린 시간 :  0.15705466270446777
model.score :  0.5244897959183673
accuracy_score :  0.48

걸린 시간 :  0.022937297821044922
model.score :  0.6438775510204081
accuracy_score :  0.68

걸린 시간 :  0.5405521392822266
model.score :  0.6969387755102041
accuracy_score :  0.64
'''