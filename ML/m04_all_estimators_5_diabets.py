from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(442,10), (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from sklearn.utils import all_estimators


# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms)
print('모델의 총 개수 : ',len(allAlgorithms))

start_time = time.time()

for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()

        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)

        # acc = accuracy_score(y_test, y_predict)
        # print(name, '의 정답률 : ', acc)
        r2 = r2_score(y_test, y_predict)
        print(name, 'r2스코어 : ', r2)
    
    except:
        # continue
        print(name, '은 없는 모델')

end_time = time.time() - start_time
print('걸린 시간 : ', end_time)

