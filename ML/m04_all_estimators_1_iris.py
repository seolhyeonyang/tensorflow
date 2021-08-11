from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')
#! warning 무시


datasets = load_iris()

x = datasets.data
y = datasets.target


# 1. 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=88)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델구성

from sklearn.utils import all_estimators
#from sklearn.utils.testing import all_estimators
#^ estimators = 추정량, (testing이 사라짐)


allAlgorithms = all_estimators(type_filter='classifier')
#! classifier에 해당하는 모델을 모두 모은다.
# allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms)
print('모델의 총 개수 : ',len(allAlgorithms))

start_time = time.time()

for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()

        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        #! score가 없는 모델이 있다. predict는 100%  있다.

        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)

    except:
        # continue
        print(name, '은 없는 모델')
        #! 오류나는거 continue 무시하고 실행, 스케일 등등 에 따라 돌아가는 모델 다르다.

end_time = time.time() - start_time
print('걸린 시간 : ', end_time)    

'''
모델의 총 개수 :  41
AdaBoostClassifier 의 정답률 :  0.9333333333333333
BaggingClassifier 의 정답률 :  0.8888888888888888
BernoulliNB 의 정답률 :  0.26666666666666666
CalibratedClassifierCV 의 정답률 :  0.9111111111111111
CategoricalNB 의 정답률 :  0.3333333333333333
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :  0.7333333333333333
DecisionTreeClassifier 의 정답률 :  0.8888888888888888
DummyClassifier 의 정답률 :  0.24444444444444444
ExtraTreeClassifier 의 정답률 :  0.8666666666666667
ExtraTreesClassifier 의 정답률 :  0.9333333333333333
GaussianNB 의 정답률 :  0.9555555555555556
GaussianProcessClassifier 의 정답률 :  0.8666666666666667
GradientBoostingClassifier 의 정답률 :  0.8888888888888888
HistGradientBoostingClassifier 의 정답률 :  0.9111111111111111
KNeighborsClassifier 의 정답률 :  0.9333333333333333
LabelPropagation 의 정답률 :  0.8888888888888888
LabelSpreading 의 정답률 :  0.8888888888888888
LinearDiscriminantAnalysis 의 정답률 :  0.9777777777777777
LinearSVC 의 정답률 :  0.9333333333333333
LogisticRegression 의 정답률 :  0.9111111111111111
LogisticRegressionCV 의 정답률 :  0.9555555555555556
MLPClassifier 의 정답률 :  0.9333333333333333
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :  0.5777777777777777
NearestCentroid 의 정답률 :  0.8888888888888888
NuSVC 의 정답률 :  0.9111111111111111
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  0.7555555555555555
Perceptron 의 정답률 :  0.7777777777777778
QuadraticDiscriminantAnalysis 의 정답률 :  0.9777777777777777
RadiusNeighborsClassifier 의 정답률 :  0.4
RandomForestClassifier 의 정답률 :  0.8888888888888888
RidgeClassifier 의 정답률 :  0.8
RidgeClassifierCV 의 정답률 :  0.8222222222222222
SGDClassifier 의 정답률 :  0.9333333333333333
SVC 의 정답률 :  0.9555555555555556
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
걸린 시간 :  0.9127874374389648
'''