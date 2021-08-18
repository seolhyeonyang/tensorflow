# 지도 학습 / 비지도 학습
#! y값(라벨)의 유무

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


datasets = load_iris()

irisDF = pd.DataFrame(data= datasets.data, columns=datasets.feature_names)
print(irisDF)

kmean = KMeans(n_clusters=3, max_iter=300, random_state=66)
#! n_clusters=라벨개수, max_iter=epochs
#^ random_state에 따라 라벨이 달라 질 수 있음 
kmean.fit(irisDF)

results = kmean.labels_
print(results)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 2 2 2 1 2 2 2 2
#  2 2 1 1 2 2 2 2 1 2 1 2 1 2 2 1 1 2 2 2 2 2 1 2 2 2 2 1 2 2 2 1 2 2 2 1 2
#  2 1]
print(datasets.target)

irisDF['cluster'] = kmean.labels_       # 클러스터링해서 생성한 y값
irisDF['target'] = datasets.target      # 원래 y값

print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

iris_results = irisDF.groupby(['target', 'cluster'])['sepal length (cm)'].count()
print(iris_results)
'''
target  cluster
0       0          50
1       1          48
        2           2
2       1          14
        2          36
'''