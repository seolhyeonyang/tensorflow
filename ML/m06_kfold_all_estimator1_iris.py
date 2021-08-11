from sklearn.datasets import load_iris
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score

datasets = load_iris()

x = datasets.data
y = datasets.target


# 1. 데이터


# 2. 모델구성

allAlgorithms = all_estimators(type_filter='classifier')
#! classifier에 해당하는 모델을 모두 모은다.
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
AdaBoostClassifier [0.63333333 0.93333333 1.         0.9        0.96666667] 0.8867
BaggingClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 0.96
BernoulliNB [0.3        0.33333333 0.3        0.23333333 0.3       ] 0.2933
CalibratedClassifierCV [0.9        0.83333333 1.         0.86666667 0.96666667] 0.9133
CategoricalNB [0.9        0.93333333 0.93333333 0.9        1.        ] 0.9333
ClassifierChain 은 없는 모델
ComplementNB [0.66666667 0.66666667 0.7        0.6        0.7       ] 0.6667
DecisionTreeClassifier [0.93333333 0.96666667 1.         0.9        0.93333333] 0.9467
DummyClassifier [0.3        0.33333333 0.3        0.23333333 0.3       ] 0.2933
ExtraTreeClassifier [0.83333333 0.93333333 0.93333333 0.93333333 0.96666667] 0.92
ExtraTreesClassifier [0.96666667 0.96666667 1.         0.86666667 0.96666667] 0.9533
GaussianNB [0.96666667 0.9        1.         0.9        0.96666667] 0.9467
GaussianProcessClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 0.96
GradientBoostingClassifier [0.93333333 0.96666667 1.         0.93333333 0.96666667] 0.96
HistGradientBoostingClassifier [0.86666667 0.96666667 1.         0.9        0.96666667] 0.94
KNeighborsClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 0.96
LabelPropagation [0.93333333 1.         1.         0.9        0.96666667] 0.96
LabelSpreading [0.93333333 1.         1.         0.9        0.96666667] 0.96
LinearDiscriminantAnalysis [1.  1.  1.  0.9 1. ] 0.98
LinearSVC [0.96666667 0.96666667 1.         0.9        1.        ] 0.9667
LogisticRegression [1.         0.96666667 1.         0.9        0.96666667] 0.9667
LogisticRegressionCV [1.         0.96666667 1.         0.9        1.        ] 0.9733
MLPClassifier [0.96666667 0.93333333 1.         0.93333333 1.        ] 0.9667
MultiOutputClassifier 은 없는 모델
MultinomialNB [0.96666667 0.93333333 1.         0.93333333 1.        ] 0.9667
NearestCentroid [0.93333333 0.9        0.96666667 0.9        0.96666667] 0.9333
NuSVC [0.96666667 0.96666667 1.         0.93333333 1.        ] 0.9733
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier [0.8        0.86666667 0.96666667 0.76666667 0.96666667] 0.8733
Perceptron [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ] 0.78
QuadraticDiscriminantAnalysis [1.         0.96666667 1.         0.93333333 1.        ] 0.98
RadiusNeighborsClassifier [0.96666667 0.9        0.96666667 0.93333333 1.        ] 0.9533
RandomForestClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 0.96
RidgeClassifier [0.86666667 0.8        0.93333333 0.7        0.9       ] 0.84
RidgeClassifierCV [0.86666667 0.8        0.93333333 0.7        0.9       ] 0.84
SGDClassifier [0.86666667 0.73333333 0.7        0.9        0.7       ] 0.78
SVC [0.96666667 0.96666667 1.         0.93333333 0.96666667] 0.9667
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
걸린 시간 :  5.910823106765747
'''