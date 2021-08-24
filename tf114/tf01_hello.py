import tensorflow as tf
print(tf.__version__)

# print('hello world')
#! 파이썬 문법


hello = tf.constant('Hello World')
#! constant = 상수 / 고정값

print(hello)
#. Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()
sess = tf.compat.v1.Session()
#! 둘다 가능 하지만 위쪽 코드는 워닝 뜸

print(sess.run(hello))
#. b'Hello World'
#! 출력하고 싶은것은 반드시 Session.run() 해야한다. (실행 되는 부분)
#^ 모든 실행은 Session안에서 해야된다.