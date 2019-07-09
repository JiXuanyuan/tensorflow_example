import tensorflow as tf

# tensor 张量
# 1.tensor是所有数据的表示形式
# 2.tensor从功能角度，可理解为多为数组
#   0阶表示标量(scalar)，1阶表示向量(vector)，n阶表示n维数组
# 3.tensor保存三个属性：名字(name)、维度(shape)、类型(dtype)
# 4.tensor类型：变量Variable()
#   一个变量在使用之前，初始化过程需要被明确地调用
# 6.tensor其他类型、创建方式：
#   常量constant()、运算操作(operator)、占位符placeholder()
#   placeholder机制，用于提供输入数据，避免计算时增加节点


# ==============================================
# 1.常量 constant
a1 = tf.constant(4)
a2 = tf.constant(5)

b1 = a1 * a2

print(a1)
print(a2)
print(b1)

with tf.Session() as sess:
    print(sess.run(b1))


# ==============================================
# 2.变量 Variable
c1 = tf.Variable([12, 3])
c2 = tf.Variable([4, 5])

d1 = c1 + c2

print(c1)
print(c2)
print(d1)

with tf.Session() as sess:
    # 初始化变量
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(d1))


# ==============================================
# 3.placeholder机制
e1 = tf.placeholder(tf.float32, shape=[2, 1])
f1 = tf.constant([[3, 4]], shape=[1, 2], dtype=tf.float32)
# 矩阵相乘，2行1列乘以1行2列
g1 = tf.matmul(e1, f1)

print(e1)
print(f1)
print(g1)

with tf.Session() as sess:
    # 初始化变量
    init = tf.global_variables_initializer()
    sess.run(init)
    # placeholder类型在计算时，需要通过feed_dict指定取值
    # feed_dict是一个字典(map)
    print(sess.run(g1, feed_dict={e1: [[3], [4]]}))




