import tensorflow as tf


# TODO +/-/*/%

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

node3_add =tf.add(node1, node2)
node4_sub = tf.subtract(node1, node2)
node5_mul = tf.multiply(node1, node2)
node6_div = tf.divide(node1, node2)

sess = tf.Session()
print('node1, node2 : ', sess.run([node1, node2]))
print('add : ', sess.run(node3_add))
print('sub : ', sess.run(node4_sub))
print('mul : ', sess.run(node5_mul))
print('div : ', sess.run(node6_div))

'''
node1, node2 :  [2.0, 3.0]
add :  5.0
sub :  -1.0
mul :  6.0
div :  0.6666667
'''