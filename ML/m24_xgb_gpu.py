from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time


# 1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
model = XGBRegressor(n_estimators=10000, learning_rate=0.01,
                    n_jobs=-1, tree_method='gpu_hist', gpu_id =0,
                    predictor='gpu_predictor')  # predictor='cpu_predictor'
#! 파라미터 오탈자는 알아서 인식하지 않는다!! 그러니 알아서 조심하자
#^ tree_method='gpu_hist' gpu로 돌리는 것
#^ gpu_id =0 gpu가 여러장일때 몇개를 사용할 것인지 지정

# 3. 훈련
start_time = time.time()
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'], 
        eval_set=[(x_train, y_train), (x_test, y_test)])

print('걸린 시간 : ',time.time() - start_time)

'''
i7-9700 / 2080

n_jobs=1
걸린 시간 :  9.402370691299438

n_jobs=2
걸린 시간 :  7.479304075241089

n_jobs=4
걸린 시간 :  6.8317577838897705

n_jobs=8
걸린 시간 :  7.098270654678345

n_jobs=-1
걸린 시간 :  7.424950361251831

tree_method='gpu_hist'
걸린 시간 :  37.94735145568848
'''