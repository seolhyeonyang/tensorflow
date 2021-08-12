from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터
datasets = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, train_size=0.8, random_state=66)


# 2. 모델
# model = DecisionTreeClassifier()
# model =RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()


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
acc  :  0.9122807017543859
[0.00468454 0.05940707 0.         0.         0.         0.
 0.         0.03138642 0.         0.02248579 0.01233852 0.
 0.         0.         0.         0.         0.         0.00433754
 0.         0.01405362 0.00624605 0.01612033 0.         0.71474329
 0.         0.         0.         0.11419683 0.         0.        ]

RandomForestClassifier
acc  :  0.956140350877193
[0.05612627 0.01482628 0.03236206 0.07246309 0.0042524  0.00549285
 0.06071702 0.08733358 0.00283947 0.00299918 0.01647271 0.00740152
 0.00959126 0.03633305 0.00407895 0.00433538 0.00909528 0.00355973
 0.00390235 0.00495722 0.11342372 0.02192636 0.1208563  0.11782812
 0.01364436 0.01152733 0.01565277 0.13208115 0.00766758 0.00625267]
'''
