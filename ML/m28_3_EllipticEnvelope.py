import numpy as np


aaa = np.array([[1, 2, 10000, 4, 5, 6, 7, 8, 90, 100, 5000],
                [1000, 2000, 3, 4000, 5000, 6000, 7000, 8, 9000, 10000, 1001]])

aaa = aaa.transpose()
# (2, 11)   ->  (11, 2)
print(aaa.shape)

from sklearn.covariance import EllipticEnvelope

outliers = EllipticEnvelope(contamination=.2)
#! outlier를 찾아 주는 함수
#^ contamination 범위는 0~0.5 (디폴트는 0.1)
outliers.fit(aaa)

results = outliers.predict(aaa)

print(results)
# [ 1  1 -1  1  1  1  1  1  1  1 -1]