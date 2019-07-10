import tensorflow as tf
from MNIST_data import input_data


# MNIST教程：
#   http://www.tensorfly.cn/tfdoc/tutorials/mnist_download.html
# 全部样本数据、部分代码可在上述网站找到


# ==============================================
# 1.加载MNIST数据
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# 2.定义样本、参数
#   样本特征值横向排列，矩阵相乘时为 x * w
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 3.定义计算图
#   构建Softmax回归模型
y_hat = tf.nn.softmax(tf.matmul(x, w) + b)
# 4.定义成本函数
#   loss = ∑(j=1,n)(- y[j] * log(y_hat[j]))
#   cost = 1/m * ∑(i=1,m)(loss[i])
#   程序中合并了损失函数与成本函数，去掉了样本数分数系数
cost = - tf.reduce_sum(y * tf.log(y_hat))
# 5.定义训练方法
#   使用梯度下降法
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# 6.定义评估方法
#   方法为在计算图上，构建一个预测值正确概率的计算节点，
#   在评估时使用测试集输入
result = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
#   预测结果为bool列表，先转换为float，再求平均
test = tf.reduce_mean(tf.cast(result, "float"))

# 7.全局初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 训练模型
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
    # 评估模型
    print(sess.run(test, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

