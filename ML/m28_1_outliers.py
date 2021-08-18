import numpy as np


aaa = np.array([1, 2, -1000, 4, 5, 6, 7, 8, 90, 100, 500])

'''
#^ 이상치 처리
1. 삭제
2. Nan 처리후 -> 보간 (linear)
3. 결측치 처리 방법과 유사
4. scaler   ->  Rubsorscaler등등 하면 이상치에서 자유로움
5. 모델링(tree계열)
#! 이상치 판별 하는것이 중요


#! 평균값으로 하면 평균이 너무 커지는 경우가 발생 제대로 된 데이터를 얻을 수 없음

데이터 정렬후(오름차순)
중위수 -> 6
1사분위 -> 2
3사분위 -> 90

quartile -> 사분위수
iqr = 3사분위 - 1사분위
'''


def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    # percentile = 백분위수
    #! 오름차순으로 정렬했을때 0을 최소값, 100을 최대값으로 백분율로 나타낸 특정위치
    #^ 사분위수는 25, 50, 75를 기준점으로 1분위 ~ 4분위
    # np.percentile(data_out, [25, 50, 75]) data_out의 25, 50, 75 지점을 반환
    # quartile_1 -> 25지점
    # q2 -> 50지점
    # quartile_3 -> 75지점
    print('1사분위 : ', quartile_1)
    print('q2 : ', q2)
    print('3사분위 : ', quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 - (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc = outliers(aaa)

print('이상치의 위치 : ', outliers_loc)

# 시각화
import matplotlib.pyplot as plt

plt.boxplot(aaa)
plt.show()