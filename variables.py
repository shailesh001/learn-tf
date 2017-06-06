import tensorflow as tf

c = tf.constant(15)
x = tf.Variable(c*5)

init = tf.global_variables_initializer()
print(x)

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(x))

