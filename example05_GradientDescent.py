import tensorflow as tf
import numpy as np

# 以一元二次函数为例，使用梯度下降方法，找到函数最小值
# 程序中展示了手动实现、借助框架实现两种代码
# 理论计算上，一元二次函数在 w = - x[1][0]/(2 * x[0][0]) 取得最小值


# ==============================================
# 1.手动实现梯度下降
w = np.array([0])
x = np.array([[1], [-2], [4]])

cost = x[0][0] * w ** 2 + x[1][0] * w + x[2][0]

# 关于cost函数对求导
# 使用梯度下降法，找到函数最小值
for i in range(1000):
    dw = 2 * x[0][0] * w + x[1][0]
    w = w - 0.01 * dw

print("w=", w)
print("理论值：", - x[1][0]/(2 * x[0][0]))


# ==============================================
# 2.借助框架实现梯度下降
# 输入系数
coefficients = np.array([[1], [-2], [4]])

w = tf.Variable([0], dtype=tf.float32)
x = tf.placeholder(tf.float32, shape=[3, 1])

# 定义成本函数
cost = x[0][0] * w ** 2 + x[1][0] * w + x[2][0]
# 定义优化方法
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train, feed_dict={x: coefficients})
    print("w=", sess.run(w))

print("理论值：", - coefficients[1][0]/(2 * coefficients[0][0]))
