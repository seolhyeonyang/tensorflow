from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')


datasets = pd.read_csv('/study2/_data/winequality-white.csv', sep= ';',
                        index_col=None, header=0)


# 1. 데이터

datasets = datasets.to_numpy()


x = datasets[ : , :11]      
y = datasets[:, 11:]



# 데이터 나누기
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


# 2. 모델구성
# model = LinearSVC()
# Acc : [0.2        0.30510204 0.3744898  0.30847804 0.47803882] 0.3332

# model = SVC()
# Acc : [0.4622449  0.4377551  0.44693878 0.46373851 0.4473953 ] 0.4516

# model = KNeighborsClassifier()
# Acc : [0.48979592 0.48469388 0.4755102  0.46373851 0.45863126] 0.4745

# model = LogisticRegression()
# Acc : [0.47142857 0.45204082 0.44795918 0.48723187 0.46578141] 0.4649

# model = DecisionTreeClassifier()
# Acc : [0.64387755 0.60306122 0.60408163 0.60367722 0.59754852] 0.6104

model = RandomForestClassifier()
# Acc : [0.71530612 0.6622449  0.68673469 0.69662921 0.69050051] 0.6903


# 3. 컴파일, 훈련
# 4. 평가, 예측

scores = cross_val_score(model, x, y, cv=kfold)

print('Acc :', scores, round(np.mean(scores),4))
