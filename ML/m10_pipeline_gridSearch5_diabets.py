from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score
import time


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=88)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# parmeters = [
#     {'randomforestregressor__n_jobs' : [-1], 'randomforestregressor__n_estimators' : [100, 200], 'randomforestregressor__max_depth' : [6, 8, 10], 'randomforestregressor__min_samples_leaf' : [5, 7, 10]},
#     {'randomforestregressor__n_jobs' : [-1], 'randomforestregressor__max_depth' : [6, 8, 10], 'randomforestregressor__min_samples_leaf' : [3, 6, 9, 11], 'randomforestregressor__min_samples_split' : [3, 4, 5]},
#     {'randomforestregressor__n_jobs' : [-1], 'randomforestregressor__min_samples_leaf' : [3, 5, 7], 'randomforestregressor__min_samples_split' : [3, 4, 5]},
#     {'randomforestregressor__n_jobs' : [-1], 'randomforestregressor__min_samples_split' : [2, 3, 5, 10]}
# ]

parmeters = [
    {'rf__n_jobs' : [-1], 'rf__n_estimators' : [100, 200], 'rf__max_depth' : [6, 8, 10], 'rf__min_samples_leaf' : [5, 7, 10]},
    {'rf__n_jobs' : [-1], 'rf__max_depth' : [6, 8, 10], 'rf__min_samples_leaf' : [3, 6, 9, 11], 'rf__min_samples_split' : [3, 4, 5]},
    {'rf__n_jobs' : [-1], 'rf__min_samples_leaf' : [3, 5, 7], 'rf__min_samples_split' : [3, 4, 5]},
    {'rf__n_jobs' : [-1], 'rf__min_samples_split' : [2, 3, 5, 10]}
]

# 2. 모델 구성
# pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())
pipe = Pipeline([('scaler', MinMaxScaler()), ('rf', RandomForestRegressor())])


model = RandomizedSearchCV(pipe, parmeters, cv=kfold, verbose=1)

# model = GridSearchCV(RandomForestRegressor(), parmeters, cv=kfold, verbose=1)



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
print('r2스코어 : ', r2_score(y_test, y_predict))

print('걸린 시간 : ', end_time)


'''
최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('rf',
                RandomForestRegressor(max_depth=8, min_samples_leaf=9,
                                    min_samples_split=4, n_jobs=-1))])
best_params_ :  {'rf__n_jobs': -1, 'rf__min_samples_split': 4, 'rf__min_samples_leaf': 9, 'rf__max_depth': 8}
best_score_ :  0.37859155978213244
model.score :  0.5091106379433453
r2스코어 :  0.5091106379433454
걸린 시간 :  5.038547515869141
'''