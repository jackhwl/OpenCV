import tensorflow as tf

# create graph
a = tf.constant(2, name='A')
b = tf.constant(3, name='B')
c = tf.add(a, b, name='Sum')

# launch the graph in a session
with tf.Session() as sess:
    print(sess.run(c))