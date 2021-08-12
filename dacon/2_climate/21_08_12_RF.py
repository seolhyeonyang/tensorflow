import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import os
import tqdm

from konlpy.tag import Okt

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

train=pd.read_csv('/study2/dacon/2_climate/_data/train.csv')
test=pd.read_csv('/study2/dacon/2_climate/_data/test.csv')
sample_submission=pd.read_csv('/study2/dacon/2_climate/_data/sample_submission.csv')

print(train.head(2))
print(sample_submission.head(6))

#데이터 구조 파악
print(train.shape)
print(test.shape)
print(sample_submission.shape)

#심각한 불균형 데이터임을 알 수 있습니다.
train.label.value_counts(sort=False)/len(train)


#해당 baseline 에서는 과제명 columns만 활용했습니다.
#다채로운 변수 활용법으로 성능을 높여주세요!
train=train[['과제명','label']]
test=test[['과제명']]

print(train.head(2))

#1. re.sub 한글 및 공백을 제외한 문자 제거
#2. okt 객체를 활용해 형태소 단위로 나눔
#3. remove_stopwords로 불용어 제거 
def preprocessing(text, okt, remove_stopwords=False, stop_words=[]):
    text=re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ]","", text)
    word_text=okt.morphs(text, stem=True)
    if remove_stopwords:
        word_review=[token for token in word_text if not token in stop_words]
    return word_review

stop_words=['은','는','이','가', '하','아','것','들','의','있','되','수','보','주','등','한']
okt=Okt()
clean_train_text=[]
clean_test_text=[]

#시간이 많이 걸립니다.
for text in tqdm.tqdm(train['과제명']):
    try:
        clean_train_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    except:
        clean_train_text.append([])

for text in tqdm.tqdm(test['과제명']):
    if type(text) == str:
        clean_test_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_test_text.append([])

print(len(clean_train_text))
print(len(clean_test_text))

from sklearn.feature_extraction.text import CountVectorizer

#tokenizer 인자에는 list를 받아서 그대로 내보내는 함수를 넣어줍니다. 또한 소문자화를 하지 않도록 설정해야 에러가 나지 않습니다.
vectorizer = CountVectorizer(tokenizer = lambda x: x, lowercase=False)
train_features=vectorizer.fit_transform(clean_train_text)
test_features=vectorizer.transform(clean_test_text)
#test데이터에 fit_transform을 할 경우 data leakage에 해당합니다

train_features

#훈련 데이터 셋과 검증 데이터 셋으로 분리
# TEST_SIZE=0.2
# RANDOM_SEED=42

TEST_SIZE=0.2
RANDOM_SEED=78

train_x, eval_x, train_y, eval_y=train_test_split(train_features, train['label'], test_size=TEST_SIZE, random_state=RANDOM_SEED)

#랜덤포레스트로 모델링
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parmeters = [
    {'n_jobs' : [-1], 'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10], 'min_samples_leaf' : [5, 7, 10]},
    {'n_jobs' : [-1], 'max_depth' : [6, 8, 10], 'min_samples_leaf' : [3, 6, 9, 11], 'min_samples_split' : [3, 4, 5]},
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7], 'min_samples_split' : [3, 4, 5]},
    {'n_jobs' : [-1], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1], 'n_estimators' : [100, 200], 'min_samples_split' : [2, 3, 5, 10]},

]

# forest=RandomizedSearchCV(RandomForestClassifier(), parmeters, cv=kfold, verbose=1)
forest=GridSearchCV(RandomForestClassifier(), parmeters, cv=kfold, verbose=1)


forest.fit(train_x, train_y)

#모델 검증
print('최적의 매개변수 : ', forest.best_estimator_)
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=-1)

print('best_params_ : ', forest.best_params_)
# best_params_ :  {'n_jobs': -1, 'min_samples_split': 3}

print('best_score_ : ', forest.best_score_)
# best_score_ :  0.9104436853770377

print(forest.score(eval_x, eval_y))
# 0.9203981526634348

'''
최적의 매개변수 :  RandomForestClassifier(n_estimators=200, n_jobs=-1)
best_params_ :  {'min_samples_split': 2, 'n_estimators': 200, 'n_jobs': -1}
best_score_ :  0.9108022733429737
0.9211726571240068
'''


#랜덤포레스트로 모델링
exit()
forest=RandomForestClassifier(n_jobs= -1, min_samples_split= 3)

forest.fit(train_x, train_y)

forest.score(eval_x, eval_y)

y_predict = forest.predict(test_features)

sample_submission['label']=forest.predict(test_features)

sample_submission.to_csv('/study2/dacon/2_climate/_save/rf_baseline3.csv', index=False)

print('f1 : ', f1_score(eval_y, y_predict, average='macro'))
