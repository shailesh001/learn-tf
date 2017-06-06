import tensorflow as tf

h = tf.constant("Hello")
w = tf.constant("World!")
hw = h + w
print(hw)
with tf.Session() as sess:
    ans = sess.run(hw)

print(ans)
