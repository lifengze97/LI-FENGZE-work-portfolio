import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
sess = tf.InteractiveSession()

##########################################################################################################

#mnist = input_data.read_data_sets('MNIST_data',one_hot=True)                  # 对mnist数据集的读取
#sess = tf.InteractiveSession()                                                # 使用tf.InteractiveSession()来构建会话

##########################################################################################################
#

def build_graph():
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = slim.conv2d(x_image, 32, [3, 3], 1, padding='SAME')
    h_pool1 = slim.max_pool2d(h_conv1, [2, 2], [2, 2], padding='SAME')

    h_conv2 = slim.conv2d(h_pool1, 64, [3, 3], 1, padding='SAME')
    h_pool2 = slim.max_pool2d(h_conv2, [2, 2], [2, 2], padding='SAME')
# slim.conv2d---卷积---slim.max_pool2d---池化
# slim.fully_connected---全连接层---slim.fully_connected()前两个参数分别为网络输入、输出的神经元数量

    flatten = slim.flatten(h_pool2)
    h_fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn=tf.nn.relu)
    y_conv = slim.fully_connected(slim.dropout(h_fc1, keep_prob), 10, activation_fn=None)

    predict = tf.argmax(y_conv, 1)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return {
        'x': x,
        'y_': y_,
        'keep_prob': keep_prob,
        'accuracy': accuracy,
        'train_step': train_step,
        'y_conv': y_conv,
        'predict': predict,
    }

# 训练函数，graph = build_graph()表示建立神经网络图，sess.run(tf.global_variables_initializer())---初始化神经网络中的参数
# saver = tf.train.Saver()---用于保存模型
# batch = mnist.train.next_batch(30)---向神经网络中喂入数据，一次喂入的数量为30
# sess.run()---训练喂入的数据，
# 当i%500==0的时候，也就是i是500的整数倍的时候，输出一次神经网络的识别准确率
# 当i%2000==0的时候，也就是i是2000的整数倍的时候，将此时神经网络的参数进行保存，括号里面是保存的位置

def train():
    with tf.Session() as sess:
        graph = build_graph()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for i in range(20001):
            batch = mnist.train.next_batch(30)
            sess.run(graph['train_step'], feed_dict={graph['x']: batch[0], graph['y_']: batch[1], graph['keep_prob']: 0.5})
            if i % 50 == 0:
                train_accuracy = sess.run(graph['accuracy'], feed_dict={graph['x']: batch[0], graph['y_']: batch[1], graph['keep_prob']: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))

            if i % 2000 == 0:
                saver.save(sess, './data/model.ckpt')


train()