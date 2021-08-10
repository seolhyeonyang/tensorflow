from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import time


datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

# 1. 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=71)

scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



# 2. 모델구성
from sklearn.utils import all_estimators


allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms)
print('모델의 총 개수 : ',len(allAlgorithms))

start_time = time.time()

for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()

        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)

        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    
    except:
        # continue
        print(name, '은 없는 모델')

end_time = time.time() - start_time
print('걸린 시간 : ', end_time)  

'''
모델의 총 개수 :  41
AdaBoostClassifier 의 정답률 :  0.9824561403508771
BaggingClassifier 의 정답률 :  0.9649122807017544
BernoulliNB 의 정답률 :  0.6403508771929824
CalibratedClassifierCV 의 정답률 :  0.9824561403508771
CategoricalNB 은 없는 모델
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :  0.8859649122807017
DecisionTreeClassifier 의 정답률 :  0.9473684210526315
DummyClassifier 의 정답률 :  0.6491228070175439
ExtraTreeClassifier 의 정답률 :  0.9473684210526315
ExtraTreesClassifier 의 정답률 :  0.9824561403508771
GaussianNB 의 정답률 :  0.9649122807017544
GaussianProcessClassifier 의 정답률 :  0.9736842105263158
GradientBoostingClassifier 의 정답률 :  0.9824561403508771
HistGradientBoostingClassifier 의 정답률 :  0.9736842105263158
KNeighborsClassifier 의 정답률 :  0.9736842105263158
LabelPropagation 의 정답률 :  0.956140350877193
LabelSpreading 의 정답률 :  0.956140350877193
LinearDiscriminantAnalysis 의 정답률 :  0.9736842105263158
LinearSVC 의 정답률 :  0.9824561403508771
LogisticRegression 의 정답률 :  0.9736842105263158
LogisticRegressionCV 의 정답률 :  0.9824561403508771
MLPClassifier 의 정답률 :  0.9824561403508771
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :  0.8859649122807017
NearestCentroid 의 정답률 :  0.9649122807017544
NuSVC 의 정답률 :  0.9649122807017544
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  0.9912280701754386
Perceptron 의 정답률 :  0.9912280701754386
QuadraticDiscriminantAnalysis 의 정답률 :  0.9736842105263158
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 :  0.9824561403508771
RidgeClassifier 의 정답률 :  0.9649122807017544
RidgeClassifierCV 의 정답률 :  0.9736842105263158
SGDClassifier 의 정답률 :  0.9912280701754386
SVC 의 정답률 :  0.9736842105263158
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
걸린 시간 :  2.075375556945801
'''
