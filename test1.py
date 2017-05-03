# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 10:27:29 2016

@author: L
"""

from gen_captcha import gen_captcha_text_and_image
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET

import numpy as np
import tensorflow as tf

from PIL import Image
# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 60
text, image = gen_captcha_text_and_image(IMAGE_WIDTH,IMAGE_HEIGHT,1)
print("验证码图像channel:", image.shape)  # (60, 160, 3)
MAX_CAPTCHA = len(text)
print("验证码文本最长字符数", MAX_CAPTCHA)   # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐
# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        from picPreHandle import Handle_Image1
        gray = Handle_Image1(Image.fromarray(img),IMAGE_WIDTH,IMAGE_HEIGHT)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return np.array(gray)
    else:
        return img

"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""

# 文本转向量
char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    def char2pos(c):
        if c =='_':
            k = 62
            return k
        k = ord(c)-48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map') 
        return k
#    print('word to vec',text)
    for i, c in enumerate(text): # enumerate(text)将字符串分离成位置和字符
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
#    print(vector.shape)
    return vector
# 向量转回文本
def vec2text(vec):
#    print('vec to word' ,vec)
    char_pos = vec.nonzero()[0] # 显示vec中非0值的位置
    text=[]
    for i, c in enumerate(char_pos): # enumerate(char_pos)将字符串分离成位置和字符
        char_at_pos = i #c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx <36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx-  36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)
# 生成一个训练batch
def get_next_batch(batch_size=64):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image(IMAGE_WIDTH,IMAGE_HEIGHT,1)
            if image.shape == (IMAGE_WIDTH,IMAGE_HEIGHT, 3):
                return text, image
#
#    for i in range(batch_size):
#        text, image = wrap_gen_captcha_text_and_image()
    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0
        batch_y[i,:] = text2vec(text)
#    print(len(batch_x),batch_y.shape)
    return batch_x, batch_y
# 定义网络超参数
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20

# 定义网络参数
n_input = IMAGE_HEIGHT*IMAGE_WIDTH # 输入的维度
n_classes = MAX_CAPTCHA*CHAR_SET_LEN # 标签的维度
dropout = 0.8 # Dropout 的概率

# 占位符输入
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# 卷积操作

def conv2d(name, l_input, w, b):
    """Build the AlexNet model. 
     
    Args: 
    name: 这个卷积器的名称
    l_input: Images Tensor 
    w : 权值
    b ：偏置
    
     
    Returns: 
    a Tensors of this op or op itself
    """  
    myconv2d = tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME')
    bias = tf.nn.bias_add(myconv2d,b)
    return tf.nn.relu(bias, name=name)

# 最大下采样操作
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

# 归一化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

# 定义整个网络 
def alex_net(_X, _weights, _biases, _dropout):
    # 向量转为矩阵
    _X = tf.reshape(_X, shape=[-1, IMAGE_HEIGHT,IMAGE_WIDTH, 1])###
#    print('x' , _X.get_shape())
    # 卷积层
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    # 下采样层
    pool1 = max_pool('pool1', conv1, k=2)
    # 归一化层
    norm1 = norm('norm1', pool1, lsize=4)
    # Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

#    print('x' , norm1.get_shape())
    # 卷积
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # 下采样
    pool2 = max_pool('pool2', conv2, k=2)
    # 归一化
    norm2 = norm('norm2', pool2, lsize=4)
    # Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)
#    print('x' , norm2.get_shape())

    # 卷积
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # 下采样
    pool3 = max_pool('pool3', conv3, k=2)
    # 归一化
    norm3 = norm('norm3', pool3, lsize=4)
    # Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)
#    print('norm3' , norm3.get_shape())

    # 全连接层，先把特征图转为向量
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') 
    # 全连接层
    dense2 = tf.reshape(dense1, [-1, _weights['wd2'].get_shape().as_list()[0]])  
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

#    print(dense2.get_shape())
#    print(_weights['out'].get_shape())
#    print(_biases['out'].get_shape())
    # 网络输出层
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out

# 存储所有的网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([8*8*256, 1024])),    #######最后这个全连接 层的参数非常重要 
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# 构建模型
pred = alex_net(x, weights, biases, keep_prob)

# 定义损失函数和学习步骤
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 测试网络
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有的共享变量
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# 开启一个训练
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    batch_xs, batch_ys = get_next_batch(64)
    while step * batch_size < training_iters:
        # 获取批数据
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # 计算精度
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # 计算损失值
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            # 如果准确率大于50%,保存模型,完成训练
            if acc > 0.5:
                saver.save(sess, "crack_capcha.model", global_step=step)
                break
        step += 1
    print("Optimization Finished!")
    # 计算测试精度
#    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))