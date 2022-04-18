import tensorflow as tf

logs_path = "./logs"
a = 2
b = 3
c = tf.add(a, b, name='Add')

with tf.compat.v1.Session() as sess:
    summary_writer = tf.compat.v1.summary.FileWriter(logs_path, sess.graph)
    print(sess.run(c))