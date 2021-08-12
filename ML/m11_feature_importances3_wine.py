from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터
datasets = load_wine()
x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, train_size=0.8, random_state=66)


# 2. 모델
model = DecisionTreeClassifier()
# model =RandomForestClassifier()


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc  : ', acc)

print(model.feature_importances_)


def plot_feature_importances_dataset(model):
    n_feature = datasets.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_,
            align = 'center')
    plt.yticks(np.arange(n_feature), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_feature)

plot_feature_importances_dataset(model)
plt.show()


'''
DecisionTreeClassifier()
acc  :  0.9444444444444444
[0.00489447 0.03045446 0.         0.         0.         0.
 0.1569445  0.         0.         0.04078249 0.07157599 0.33215293
 0.36319516]

RandomForestClassifier
acc  :  1.0
[0.13947981 0.0293382  0.0212199  0.02832531 0.02188401 0.07409211
 0.13289242 0.01241765 0.02911948 0.15144849 0.07080547 0.1349219
 0.15405525]
'''
