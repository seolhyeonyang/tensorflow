import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import time


datasets = pd.read_csv('/study2/_data/winequality-white.csv', sep= ';',
                        index_col=None, header=0)


# 1. 데이터

datasets = datasets.to_numpy()


x = datasets[ : , :11]      
y = datasets[:, 11:]


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=78)


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
AdaBoostClassifier 의 정답률 :  0.43673469387755104
BaggingClassifier 의 정답률 :  0.6571428571428571
BernoulliNB 의 정답률 :  0.4377551020408163
CalibratedClassifierCV 의 정답률 :  0.5275510204081633
CategoricalNB 은 없는 모델
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :  0.3826530612244898
DecisionTreeClassifier 의 정답률 :  0.6418367346938776
DummyClassifier 의 정답률 :  0.4357142857142857
ExtraTreeClassifier 의 정답률 :  0.6224489795918368
ExtraTreesClassifier 의 정답률 :  0.7020408163265306
GaussianNB 의 정답률 :  0.47244897959183674
GaussianProcessClassifier 의 정답률 :  0.5326530612244897
GradientBoostingClassifier 의 정답률 :  0.5846938775510204
HistGradientBoostingClassifier 의 정답률 :  0.6928571428571428
KNeighborsClassifier 의 정답률 :  0.5642857142857143
LabelPropagation 의 정답률 :  0.5255102040816326
LabelSpreading 의 정답률 :  0.5193877551020408
LinearDiscriminantAnalysis 의 정답률 :  0.5183673469387755
LinearSVC 의 정답률 :  0.5163265306122449
LogisticRegression 의 정답률 :  0.5244897959183673
LogisticRegressionCV 의 정답률 :  0.5214285714285715
MLPClassifier 의 정답률 :  0.5540816326530612
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :  0.4346938775510204
NearestCentroid 의 정답률 :  0.3
NuSVC 은 없는 모델
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  0.44387755102040816
Perceptron 의 정답률 :  0.32857142857142857
QuadraticDiscriminantAnalysis 의 정답률 :  0.5061224489795918
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 :  0.6979591836734694
RidgeClassifier 의 정답률 :  0.5183673469387755
RidgeClassifierCV 의 정답률 :  0.5173469387755102
SGDClassifier 의 정답률 :  0.49081632653061225
SVC 의 정답률 :  0.5469387755102041
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
걸린 시간 :  68.17498326301575
'''