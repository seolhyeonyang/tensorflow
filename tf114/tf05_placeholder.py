import tensorflow as tf


node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
#! placeholder는 안이 비어있는 자료형

adder_node = a + b

# print(sess.run(adder_node, feed_dict={a:3, b:4.5}))       # 7.5
#! feed_dict={} 넣어주면 연산 가능

print(sess.run(adder_node, feed_dict={a:[1, 3], b:[3, 4]})) # [4. 7.]
#! 다차원 연산 가능 / [] 넣으면 순서 대로 연산

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a: 4, b: 2}))     # 18.0
