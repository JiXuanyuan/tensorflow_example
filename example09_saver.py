import tensorflow as tf
import os
from MNIST_data import input_data


# MNIST教程：
#   http://www.tensorfly.cn/tfdoc/tutorials/mnist_download.html
# 全部样本数据、部分代码可在上述网站找到


# ==============================================

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


def restore():
    print("restore")
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("../save/model.ckpt.meta")
        saver.restore(sess, "../save/model.ckpt")

        graph = tf.get_default_graph()
        x = graph.get_operation_by_name("x").outputs[0]
        y = graph.get_operation_by_name("y").outputs[0]
        test = graph.get_operation_by_name("test").outputs[0]

        print(sess.run(test, feed_dict={x: mnist.test.images, y: mnist.test.labels}))


if os.path.exists('../save/checkpoint'):
    restore()
    exit()

x = tf.placeholder("float", [None, 784], name="x")
y = tf.placeholder("float", [None, 10], name="y")
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_hat = tf.nn.softmax(tf.matmul(x, w) + b)
cost = - tf.reduce_sum(y * tf.log(y_hat))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

result = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
test = tf.reduce_mean(tf.cast(result, "float"), name="test")

init = tf.global_variables_initializer()

# 打印测试模型参数
print(x)
print(y)
print(test)

# 1.创建saver
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # 训练模型
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
    # 评估模型
    print(sess.run(test, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    # 2.保存模型
    print(saver.save(sess, "../save/model.ckpt"))


