import pandas as pd
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.model_selection import KFold, cross_val_score
import numpy as np


datasets = pd.read_csv('/study2/_data/winequality-white.csv', sep= ';',
                        index_col=None, header=0)


# 1. 데이터

datasets = datasets.to_numpy()


x = datasets[ : , :11]      
y = datasets[:, 11:]


# 2. 모델구성


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
AdaBoostClassifier [0.41428571 0.45       0.42244898 0.36261491 0.43615935] 0.4171
BaggingClassifier [0.67653061 0.62346939 0.64897959 0.6578141  0.65168539] 0.6517
BernoulliNB [0.45816327 0.43367347 0.44285714 0.46271706 0.44637385] 0.4488
CalibratedClassifierCV [0.51326531 0.47142857 0.49693878 0.55158325 0.48416752] 0.5035
CategoricalNB [       nan        nan 0.50306122 0.51072523        nan] nan
ClassifierChain 은 없는 모델
ComplementNB [0.38163265 0.37653061 0.36632653 0.34320735 0.36159346] 0.3659
DecisionTreeClassifier [0.62653061 0.59795918 0.60204082 0.59959142 0.61899898] 0.609
DummyClassifier [0.45816327 0.43367347 0.44285714 0.46271706 0.44637385] 0.4488
ExtraTreeClassifier [0.64285714 0.59081633 0.62040816 0.60980592 0.61287028] 0.6154
ExtraTreesClassifier [0.71938776 0.67040816 0.6744898  0.70684372 0.68539326] 0.6913
GaussianNB [0.46530612 0.44591837 0.45510204 0.41266599 0.46373851] 0.4485
GaussianProcessClassifier [0.59693878 0.57244898 0.58163265 0.57405516 0.57099081] 0.5792
GradientBoostingClassifier [0.61122449 0.57244898 0.59489796 0.61082737 0.59141982] 0.5962
HistGradientBoostingClassifier [0.69183673 0.6622449  0.67653061 0.67109295 0.6639428 ] 0.6731
KNeighborsClassifier [0.48979592 0.48469388 0.4755102  0.46373851 0.45863126] 0.4745
LabelPropagation [0.59387755 0.57244898 0.57040816 0.5628192  0.56588355] 0.5731
LabelSpreading [0.59387755 0.57244898 0.57040816 0.56384065 0.56588355] 0.5733
LinearDiscriminantAnalysis [0.5255102  0.51326531 0.5244898  0.56384065 0.52706844] 0.5308
LinearSVC [0.32346939 0.25510204 0.19591837 0.52911134 0.33707865] 0.3281
LogisticRegression [0.47142857 0.45204082 0.44795918 0.48723187 0.46578141] 0.4649
LogisticRegressionCV [0.50204082 0.49591837 0.48979592 0.53421859 0.49233912] 0.5029
MLPClassifier [0.53877551 0.52244898 0.52142857 0.53524004 0.48314607] 0.5202
MultiOutputClassifier 은 없는 모델
MultinomialNB [0.41326531 0.39693878 0.3877551  0.38304392 0.40653728] 0.3975
NearestCentroid [0.12959184 0.10204082 0.10102041 0.11235955 0.09090909] 0.1072
NuSVC [nan nan nan nan nan] nan
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier [0.45816327 0.45510204 0.29081633 0.46169561 0.4453524 ] 0.4222
Perceptron [0.45816327 0.43367347 0.32244898 0.33094995 0.09499489] 0.328
QuadraticDiscriminantAnalysis [0.48367347 0.45102041 0.50306122 0.46782431 0.48008172] 0.4771
RadiusNeighborsClassifier [nan nan nan nan nan] nan
RandomForestClassifier [0.70102041 0.67142857 0.69387755 0.68845761 0.68845761] 0.6886
RidgeClassifier [0.53163265 0.5122449  0.52142857 0.54954035 0.51276813] 0.5255
RidgeClassifierCV [0.53163265 0.5122449  0.52142857 0.54954035 0.51276813] 0.5255
SGDClassifier [0.06734694 0.34795918 0.44285714 0.4494382  0.36261491] 0.334
SVC [0.4622449  0.4377551  0.44693878 0.46373851 0.4473953 ] 0.4516
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
걸린 시간 :  429.44256234169006
'''