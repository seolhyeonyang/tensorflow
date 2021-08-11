from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score


datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

# 1. 데이터


# 2. 모델구성
from sklearn.utils import all_estimators


allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms)
print('모델의 총 개수 : ',len(allAlgorithms))

start_time = time.time()

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()

        scores = cross_val_score(model, x, y, cv=kfold)
        print(name, scores, round(np.mean(scores),4))

    except:
        # continue
        print(name, '은 없는 모델')

end_time = time.time() - start_time
print('걸린 시간 : ', end_time)  

'''
모델의 총 개수 :  41
AdaBoostClassifier [0.94736842 0.99122807 0.94736842 0.96491228 0.97345133] 0.9649
BaggingClassifier [0.93859649 0.92982456 0.95614035 0.94736842 0.95575221] 0.9455
BernoulliNB [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 0.6274
CalibratedClassifierCV [0.89473684 0.93859649 0.89473684 0.92982456 0.97345133] 0.9263
CategoricalNB [nan nan nan nan nan] nan
ClassifierChain 은 없는 모델
ComplementNB [0.86842105 0.92982456 0.87719298 0.9122807  0.89380531] 0.8963
DecisionTreeClassifier [0.94736842 0.92105263 0.92105263 0.88596491 0.95575221] 0.9262
DummyClassifier [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 0.6274
ExtraTreeClassifier [0.92982456 0.88596491 0.92105263 0.90350877 0.9380531 ] 0.9157
ExtraTreesClassifier [0.96491228 0.98245614 0.96491228 0.94736842 0.99115044] 0.9702
GaussianNB [0.93859649 0.96491228 0.9122807  0.93859649 0.95575221] 0.942
GaussianProcessClassifier [0.87719298 0.89473684 0.89473684 0.94736842 0.94690265] 0.9122
GradientBoostingClassifier [0.95614035 0.97368421 0.95614035 0.93859649 0.98230088] 0.9614
HistGradientBoostingClassifier [0.97368421 0.98245614 0.96491228 0.96491228 0.98230088] 0.9737
KNeighborsClassifier [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221] 0.928
LabelPropagation [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 0.3902
LabelSpreading [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 0.3902
LinearDiscriminantAnalysis [0.94736842 0.98245614 0.94736842 0.95614035 0.97345133] 0.9614
LinearSVC [0.90350877 0.92105263 0.9122807  0.92105263 0.86725664] 0.905
LogisticRegression [0.93859649 0.95614035 0.88596491 0.94736842 0.96460177] 0.9385
LogisticRegressionCV [0.96491228 0.97368421 0.92105263 0.96491228 0.96460177] 0.9578
MLPClassifier [0.88596491 0.93859649 0.9122807  0.92982456 0.96460177] 0.9263
MultiOutputClassifier 은 없는 모델
MultinomialNB [0.85964912 0.92105263 0.87719298 0.9122807  0.89380531] 0.8928
NearestCentroid [0.86842105 0.89473684 0.85964912 0.9122807  0.91150442] 0.8893
NuSVC [0.85964912 0.9122807  0.83333333 0.87719298 0.88495575] 0.8735
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier [0.86842105 0.92105263 0.88596491 0.92105263 0.89380531] 0.8981
Perceptron [0.40350877 0.80701754 0.85964912 0.86842105 0.94690265] 0.7771
QuadraticDiscriminantAnalysis [0.93859649 0.95614035 0.93859649 0.98245614 0.94690265] 0.9525
RadiusNeighborsClassifier [nan nan nan nan nan] nan
RandomForestClassifier [0.95614035 0.96491228 0.96491228 0.94736842 0.98230088] 0.9631
RidgeClassifier [0.95614035 0.98245614 0.92105263 0.95614035 0.95575221] 0.9543
RidgeClassifierCV [0.94736842 0.97368421 0.93859649 0.95614035 0.96460177] 0.9561
SGDClassifier [0.84210526 0.86842105 0.86842105 0.92982456 0.90265487] 0.8823
SVC [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177] 0.921
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
걸린 시간 :  13.130767822265625
'''
