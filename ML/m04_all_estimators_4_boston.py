from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score



# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=78)

scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성
from sklearn.utils import all_estimators


# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms)
print('모델의 총 개수 : ',len(allAlgorithms))

start_time = time.time()

for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()

        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)

        # acc = accuracy_score(y_test, y_predict)
        # print(name, '의 정답률 : ', acc)
        r2 = r2_score(y_test, y_predict)
        print(name, 'r2스코어 : ', r2)
    
    except:
        # continue
        print(name, '은 없는 모델')

end_time = time.time() - start_time
print('걸린 시간 : ', end_time)

'''
모델의 총 개수 :  54
ARDRegression r2스코어 :  0.7489519488710621
AdaBoostRegressor r2스코어 :  0.8458743629970852
BaggingRegressor r2스코어 :  0.900447573738566
BayesianRidge r2스코어 :  0.744317600633944
CCA r2스코어 :  0.7112984219966143
DecisionTreeRegressor r2스코어 :  0.7971798115419315
DummyRegressor r2스코어 :  -0.004075176213653053
ElasticNet r2스코어 :  0.15090688291106036
ElasticNetCV r2스코어 :  0.7424421005454744
ExtraTreeRegressor r2스코어 :  0.7806435968361656
ExtraTreesRegressor r2스코어 :  0.9247182775894106
GammaRegressor r2스코어 :  0.1672362093903369
GaussianProcessRegressor r2스코어 :  -1.246869161005376
GradientBoostingRegressor r2스코어 :  0.9270621309281876
HistGradientBoostingRegressor r2스코어 :  0.9196362897790192
HuberRegressor r2스코어 :  0.7453275487042539
IsotonicRegression 은 없는 모델
KNeighborsRegressor r2스코어 :  0.7054961415199352
KernelRidge r2스코어 :  0.6964892442426712
Lars r2스코어 :  0.7452210925673006
LarsCV r2스코어 :  0.7494634848070956
Lasso r2스코어 :  0.24048956951414724
LassoCV r2스코어 :  0.7455714332711323
LassoLars r2스코어 :  -0.004075176213653053
LassoLarsCV r2스코어 :  0.7452210925673007
LassoLarsIC r2스코어 :  0.7452580930433159
LinearRegression r2스코어 :  0.7452210925673008
LinearSVR r2스코어 :  0.6443683836402193
MLPRegressor r2스코어 :  0.29562874781513926
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet 은 없는 모델
MultiTaskElasticNetCV 은 없는 모델
MultiTaskLasso 은 없는 모델
MultiTaskLassoCV 은 없는 모델
NuSVR r2스코어 :  0.5856134097385404
OrthogonalMatchingPursuit r2스코어 :  0.519229098284893
OrthogonalMatchingPursuitCV r2스코어 :  0.7124347701278164
PLSCanonical r2스코어 :  -1.9000915427606522
PLSRegression r2스코어 :  0.7376402639051556
PassiveAggressiveRegressor r2스코어 :  0.3070753547315208
PoissonRegressor r2스코어 :  0.6228614095841551
RANSACRegressor r2스코어 :  0.6685543684245524
RadiusNeighborsRegressor r2스코어 :  0.32030183398588197
RandomForestRegressor r2스코어 :  0.917430504534049
RegressorChain 은 없는 모델
Ridge r2스코어 :  0.7389866481429963
RidgeCV r2스코어 :  0.7448919748552758
SGDRegressor r2스코어 :  0.7354271526527096
SVR r2스코어 :  0.6246956126231183
StackingRegressor 은 없는 모델
TheilSenRegressor r2스코어 :  0.7381433276827485
TransformedTargetRegressor r2스코어 :  0.7452210925673008
TweedieRegressor r2스코어 :  0.1696253245054159
VotingRegressor 은 없는 모델
걸린 시간 :  1.6924042701721191
'''