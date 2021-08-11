from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

datasets = pd.read_csv('/study2/_data/winequality-white.csv', sep= ';',
                        index_col=None, header=0)


# 1. 데이터

datasets = datasets.to_numpy()


x = datasets[ : , :11]      
y = datasets[:, 11:]



# 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# parmeters = [
#     {'randomforestclassifier__n_jobs' : [-1], 'randomforestclassifier__n_estimators' : [100, 200], 'randomforestclassifier__max_depth' : [6, 8, 10], 'randomforestclassifier__min_samples_leaf' : [5, 7, 10]},
#     {'randomforestclassifier__n_jobs' : [-1], 'randomforestclassifier__max_depth' : [6, 8, 10], 'randomforestclassifier__min_samples_leaf' : [3, 6, 9, 11], 'randomforestclassifier__min_samples_split' : [3, 4, 5]},
#     {'randomforestclassifier__n_jobs' : [-1], 'randomforestclassifier__min_samples_leaf' : [3, 5, 7], 'randomforestclassifier__min_samples_split' : [3, 4, 5]},
#     {'randomforestclassifier__n_jobs' : [-1], 'randomforestclassifier__min_samples_split' : [2, 3, 5, 10]}
# ]

parmeters = [
    {'rf__n_jobs' : [-1], 'rf__n_estimators' : [100, 200], 'rf__max_depth' : [6, 8, 10], 'rf__min_samples_leaf' : [5, 7, 10]},
    {'rf__n_jobs' : [-1], 'rf__max_depth' : [6, 8, 10], 'rf__min_samples_leaf' : [3, 6, 9, 11], 'rf__min_samples_split' : [3, 4, 5]},
    {'rf__n_jobs' : [-1], 'rf__min_samples_leaf' : [3, 5, 7], 'rf__min_samples_split' : [3, 4, 5]},
    {'rf__n_jobs' : [-1], 'rf__min_samples_split' : [2, 3, 5, 10]}
]

# 2. 모델 구성
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
pipe = Pipeline([('scaler', MinMaxScaler()), ('rf', RandomForestClassifier())])


model = RandomizedSearchCV(pipe, parmeters, cv=kfold, verbose=1)

# model = GridSearchCV(RandomForestClassifier(), parmeters, cv=kfold, verbose=1)


# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time


# 4. 평가, 예측

print('최적의 매개변수 : ', model.best_estimator_)

print('best_params_ : ', model.best_params_)

print('best_score_ : ', model.best_score_)

print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_predict))


print('걸린 시간 : ', end_time)


'''
최적의 매개변수 :  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestclassifier',
                RandomForestClassifier(min_samples_split=3, n_jobs=-1))])
best_params_ :  {'randomforestclassifier__n_jobs': -1, 'randomforestclassifier__min_samples_split': 3}
best_score_ :  0.6536522271743948
model.score :  0.7193877551020408
정답률 :  0.7193877551020408
걸린 시간 :  7.810338735580444
'''