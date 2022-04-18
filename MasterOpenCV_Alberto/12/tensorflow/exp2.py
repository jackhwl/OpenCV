import tensorflow as tf

logs_path = "./logs"
x = 2
y = 3
add_op = tf.add(x, y, name='Add')
mul_op = tf.multiply(x, y, name='Multiply')
pow_op = tf.pow(add_op, mul_op, name='Power')
useless_op = tf.multiply(x, add_op, name='Useless')

with tf.Session() as sess:
    summary_writer = tf.compat.v1.summary.FileWriter(logs_path, sess.graph)
    pow_out, useless_out = sess.run([pow_op, useless_op])
    