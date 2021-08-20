import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)
    #! 0보다 크면 그대로 , 0보다 작으면 0으로 한다.

x = np.arange(-5, 5, 0.1)

y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

'''
과제
elu, selu, reaky relu ...
68_3_2, 3, 4, 로 해서 만들기
'''