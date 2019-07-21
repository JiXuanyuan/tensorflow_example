import tensorflow as tf
from datetime import datetime
from MNIST_data import input_data


# MNIST教程：
#   http://www.tensorfly.cn/tfdoc/tutorials/mnist_download.html
# 全部样本数据、部分代码可在上述网站找到


# ==============================================
STR_TIMESTAMP = '{0:%Y-%m-%dT%H:%M:%S}'.format(datetime.now())
DIR_LOGS = '../logs/' + STR_TIMESTAMP

# 数据源
mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

# 输入层
with tf.name_scope('input_layer'):
    with tf.name_scope('x'):
        x = tf.placeholder(tf.float32, [None, 784])
        # 保存输入图片x
        tf.summary.image('x', tf.reshape(x, [-1, 28, 28, 1]), 10)
    with tf.name_scope('y'):
        y = tf.placeholder(tf.float32, [None, 10])

# 输出层
with tf.name_scope('output_layer'):
    with tf.name_scope('w'):
        w = tf.Variable(tf.zeros([784, 10]))
        # 保存张量w
        tf.summary.histogram('w', w)
    with tf.name_scope('b'):
        b = tf.Variable(tf.zeros([10]))
    with tf.name_scope('z'):
        z = tf.matmul(x, w) + b
    with tf.name_scope('softmax'):
        y_hat = tf.nn.softmax(z)

# 成本函数
with tf.name_scope('cost'):
    cost = - tf.reduce_sum(y * tf.log(y_hat))
    # 保存标量cost
    tf.summary.scalar('cost', cost)

# 训练方法
with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# 评估方法
with tf.name_scope('test'):
    result = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    test = tf.reduce_mean(tf.cast(result, tf.float32))

# 初始化变量
init = tf.global_variables_initializer()
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


