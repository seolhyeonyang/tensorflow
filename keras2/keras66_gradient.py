import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 - 4*x +6

x = np.linspace(-1, 6, 100)
#^ -1 ~ 6 까지 100개의 데이터를 생성

y = f(x)

# 시각화
plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')

plt.show()