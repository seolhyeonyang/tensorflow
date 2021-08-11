from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(442,10), (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
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
ARDRegression r2스코어 :  0.5894071989794667
AdaBoostRegressor r2스코어 :  0.5218787481801976
BaggingRegressor r2스코어 :  0.45750333631962337
BayesianRidge r2스코어 :  0.5954128350161362
CCA r2스코어 :  0.5852713803269576
DecisionTreeRegressor r2스코어 :  -0.17350205245771333
DummyRegressor r2스코어 :  -0.01545589029660177
ElasticNet r2스코어 :  0.1436197162111983
ElasticNetCV r2스코어 :  0.5838811172714526
ExtraTreeRegressor r2스코어 :  0.03863797520718859
ExtraTreesRegressor r2스코어 :  0.5541145181880283
GammaRegressor r2스코어 :  0.08710574754327771
GaussianProcessRegressor r2스코어 :  -16.860774095861917
GradientBoostingRegressor r2스코어 :  0.5462556708307605
HistGradientBoostingRegressor r2스코어 :  0.5392377257708727
HuberRegressor r2스코어 :  0.5785993482752368
IsotonicRegression 은 없는 모델
KNeighborsRegressor r2스코어 :  0.4904172186988308
KernelRidge r2스코어 :  0.5905597611712825
Lars r2스코어 :  0.585114126995973
LarsCV r2스코어 :  0.58962066606799
Lasso r2스코어 :  0.5816741737120826
LassoCV r2스코어 :  0.586470179320693
LassoLars r2스코어 :  0.4523872639391452
LassoLarsCV r2스코어 :  0.58962066606799
LassoLarsIC r2스코어 :  0.5962837270477235
LinearRegression r2스코어 :  0.5851141269959738
LinearSVR r2스코어 :  0.3510367525870707
MLPRegressor r2스코어 :  -0.5357068138050698
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet 은 없는 모델
MultiTaskElasticNetCV 은 없는 모델
MultiTaskLasso 은 없는 모델
MultiTaskLassoCV 은 없는 모델
NuSVR r2스코어 :  0.16780359645796983
OrthogonalMatchingPursuit r2스코어 :  0.32241768669099424
OrthogonalMatchingPursuitCV r2스코어 :  0.5812114430406672
PLSCanonical r2스코어 :  -1.6878518911601015
PLSRegression r2스코어 :  0.6072433305368464
PassiveAggressiveRegressor r2스코어 :  0.5889418634456447
PoissonRegressor r2스코어 :  0.580099523626783
RANSACRegressor r2스코어 :  0.191486579409467
RadiusNeighborsRegressor r2스코어 :  0.17842231317289414
RandomForestRegressor r2스코어 :  0.5302741094776809
RegressorChain 은 없는 모델
Ridge r2스코어 :  0.5940247828297051
RidgeCV r2스코어 :  0.593525110776487
SGDRegressor r2스코어 :  0.58200702376401
SVR r2스코어 :  0.18982394795986235
StackingRegressor 은 없는 모델
TheilSenRegressor r2스코어 :  0.5887187446294627
TransformedTargetRegressor r2스코어 :  0.5851141269959738
TweedieRegressor r2스코어 :  0.0830837969884558
VotingRegressor 은 없는 모델
걸린 시간 :  1.4777767658233643
'''