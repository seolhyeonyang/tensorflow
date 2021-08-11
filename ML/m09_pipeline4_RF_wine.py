from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score


datasets = pd.read_csv('/study2/_data/winequality-white.csv', sep= ';',
                        index_col=None, header=0)


# 1. 데이터

datasets = datasets.to_numpy()


x = datasets[ : , :11]      
y = datasets[:, 11:]



# 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델구성
from sklearn.pipeline import make_pipeline, Pipeline

model = make_pipeline(MinMaxScaler(), RandomForestClassifier())

# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time


# 4. 평가, 예측

print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))


print('걸린 시간 : ', end_time)


'''
model.score :  0.713265306122449
정답률 :  0.713265306122449
걸린 시간 :  0.514948844909668
'''