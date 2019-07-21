import tensorflow as tf
from datetime import datetime
from MNIST_data import input_data


# MNIST教程：
#   http://www.tensorfly.cn/tfdoc/tutorials/mnist_download.html
# 全部样本数据、部分代码可在上述网站找到


# ==============================================
STR_TIMESTAMP = "{0:%Y-%m-%dT%H:%M:%S}".format(datetime.now())
DIR_LOGS = "../logs/" + STR_TIMESTAMP

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 构建模型
y_hat = tf.nn.softmax(tf.matmul(x, w) + b)
cost = - tf.reduce_sum(y * tf.log(y_hat))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# 评估方法
result = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
test = tf.reduce_mean(tf.cast(result, "float"))

# 初始化变量
init = tf.global_variables_initializer()

# ==============================================
# 1.指定需要可视化的变量
# 保存输入图片x
#   格式：[batch_size, height, width, channels]
#   转换x的形状，-1表示数量任意，高28宽28，通道数1，为灰度图像
#   最多保存10张
tf.summary.image("x", tf.reshape(x, [-1, 28, 28, 1]), 10)
# 保存张量w
tf.summary.histogram("w", w)
# 保存标量cost
tf.summary.scalar("cost", cost)
# 2.整合所有可视化量
merge = tf.summary.merge_all()

# ==============================================
with tf.Session() as sess:
    sess.run(init)
    # 3.初始化写日志的writer，并保存当前计算图
    with tf.summary.FileWriter(DIR_LOGS, sess.graph) as writer:
        # 训练模型
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            # 4.获得这次运行的日志
            summary, _ = sess.run([merge, train], feed_dict={
                x: batch_xs, y: batch_ys})
            # 5.将日志写入文件
            writer.add_summary(summary, i)
        # 评估模型
        print(sess.run(test, feed_dict={
            x: mnist.test.images, y: mnist.test.labels}))

# 终端：
#   tensorboard --logdir=logs/
# 浏览器：
#   http://localhost:6006
