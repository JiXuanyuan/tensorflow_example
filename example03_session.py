import tensorflow as tf

# session 会话
# 1.session用来组织数据和运算，管理程序运算时的所有资源
# 2.所有计算完成之后，需要关闭session，来帮助系统回收资源


# ==============================================
# 定义常量
a1 = tf.constant(12)
a2 = tf.constant(5)

print(a1)
print(a2)

# 数学运算
b5 = tf.add(a1, a2, name="add")
b6 = tf.subtract(a1, a2, name="sub")
b7 = tf.multiply(a1, a2, name="mul")
b8 = tf.divide(a1, a2, name="div")

print(b5)
print(b6)
print(b7)
print(b8)


# 1.创建会话，使用会话执行运算，with自动管理session资源
with tf.Session() as sess:
    print(sess.run(b5))
    print(sess.run(b6))
    print(sess.run(b7))
    print(sess.run(b8))


# ==============================================
# 2.创建会话，设定默认会话
sess = tf.Session()

with sess.as_default():
    print(b5.eval())
    print(b6.eval())
    print(b7.eval())
    print(b8.eval())

sess.close()


# ==============================================
# 3.在交互式环境中，比如使用IPython，最开始使用
# 未构建计算图之前，加载它自身作为默认构建的session
sess = tf.InteractiveSession()

# 构建计算图
c1 = tf.constant(3)
c2 = tf.constant(4)
d1 = c1 * c2

print(d1.eval())
sess.close()



