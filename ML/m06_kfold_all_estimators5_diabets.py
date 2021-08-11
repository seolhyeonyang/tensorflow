from sklearn.datasets import load_diabetes
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target


#2. 모델구성

# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')
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

'''