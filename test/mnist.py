from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import numpy as np
global_times = 0


def image_prepare(file_path):
    im = Image.open(file_path).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im.save('./data/sample.png')#保存路径
    #plt.imshow(im)
    #plt.show()

    tv = list(im.getdata())  # get pixel values 获取像素
    return tv


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


def predict_prepare():
    sess = tf.Session()
    graph = build_graph()
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint('./model/')
    if ckpt:
        saver.restore(sess, ckpt) #模型的恢复返回神经网络和回话
    return graph, sess


def recognition(file_path):
    global global_times #全局变量
    if global_times == 0: #需不需要加载模型  若==1 改变全局变量 下一次识别直接进入else
        global graph, sess #都是全局变量 一处变动都要变
        graph, sess = predict_prepare() #加载模型
        image = image_prepare(file_path)
        predict = sess.run(graph['predict'], feed_dict={graph['x']: [image], graph['keep_prob']: 1.0}) #将预处理后的图片喂入神经网络 得到预测值
        global_times = 1

        print('recognize result:')
        print(predict)
        return predict

    else:

        image = image_prepare(file_path)
        predict = sess.run(graph['predict'], feed_dict={graph['x']: [image], graph['keep_prob']: 1.0})

        print('recognize result:')
        print(predict)
        return predict



