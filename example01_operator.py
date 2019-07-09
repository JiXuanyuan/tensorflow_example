import tensorflow as tf

# 定义常量
a1 = tf.constant(12)
a2 = tf.constant(5)

print(a1)
print(a2)

# 数学运算，重载操作符，简便但无法命名
b1 = a1 + a2
b2 = a1 - a2
b3 = a1 * a2
b4 = a1 / a2

print(b1)
print(b2)
print(b3)
print(b4)

# 数学运算
b5 = tf.add(a1, a2, name="add")
b6 = tf.subtract(a1, a2, name="sub")
b7 = tf.multiply(a1, a2, name="mul")
b8 = tf.divide(a1, a2, name="div")

print(b5)
print(b6)
print(b7)
print(b8)

with tf.Session() as sess:
    print(sess.run(a1))
    print(sess.run(a2))
    print(sess.run(b1))
    print(sess.run(b2))
    print(sess.run(b3))
    print(sess.run(b4))
    print(sess.run(b5))
    print(sess.run(b6))
    print(sess.run(b7))
    print(sess.run(b8))


