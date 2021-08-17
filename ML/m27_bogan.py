# [1, np.nan, np.nan, 8, 10]
#! 데이터가 적을때 결측치를 삭제하면 데이터 손실이 크다.

'''
#^ 결측치 처리
1. 행 삭제
2. 0 넣기 (특정값 넣기) ->  [1, 0, 0, 8, 10]
3. 앞에 값 넣기
4. 뒤에 값 넣기
5. 중위값 넣기
6. 보간 (linear 기준)
7. 모델링 - predict (결측치를 predict 한다.)
8. 부스트 계열은 결측치에서 대해 자유롭다.(결측치 처리를 안 해도 된다.)
'''

from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd


datastrs = ['8/13/2021', '8/14/2021', '8/15/2021', '8/16/2021', '8/17/2021']
dates = pd.to_datetime(datastrs)
print(dates)
print(type(dates))      # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
print('============================')


ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
print(ts)

ts_intp_linear = ts.interpolate()
#! 결측치 값 채우기, linear 방식으로 보간해 준다.
#^ 이상치가 있다면 이상치를 제거한 후 보간해 주기
print(ts_intp_linear)