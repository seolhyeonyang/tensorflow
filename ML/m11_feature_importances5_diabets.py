from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터
datasets = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, train_size=0.8, random_state=66)


# 2. 모델
# model = DecisionTreeRegressor()
model =RandomForestRegressor()


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
DecisionTreeRegressor
acc  :  -0.1983616822489902
[0.06664682 0.00485975 0.22942387 0.12717207 0.04202624 0.03800121
 0.04890225 0.01221001 0.36206639 0.06869139]

RandomForestRegresso
acc  :  0.3809906006678777
[0.06055114 0.01098067 0.28311533 0.11058213 0.04046715 0.05173432
 0.05004007 0.02175209 0.29408978 0.07668732]
'''