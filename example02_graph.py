import tensorflow as tf

# graph 计算图
# 1.graph用来构建计算流程，不组织数据和运算
# 2.graph由节点和弧组成，对应tensor和operator
# 3.tensorflow提供了一个默认图，创建的节点默认加到该图


# ==============================================
# 1.使用默认的计算图
# 构建计算图
a1 = tf.constant(12, dtype=tf.float32, name="input1")
a2 = tf.constant(5, dtype=tf.float32, name="input2")

print(a1)
print(a2)

b5 = tf.add(a1, a2, name="add")
b6 = tf.subtract(a1, a2, name="sub")
b7 = tf.multiply(a1, a2, name="mul")
b8 = tf.divide(a1, a2, name="div")

print(b5)
print(b6)
print(b7)
print(b8)

# 获取默认图，返回一个图的序列化的GraphDef表示
print(tf.get_default_graph().as_graph_def())


# ==============================================
# 2.创建一个新图
g = tf.Graph()

with g.as_default():
    # 构建计算图
    c1 = tf.constant(3)
    c2 = tf.constant(4)
    d1 = c1 * c2
    print(c1)
    print(c2)
    print(d1)

print(g.as_graph_def())
