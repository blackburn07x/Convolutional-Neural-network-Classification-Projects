#VGG network using Tensorflow 1.x
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import style

# Network
class VGG16net:

    def __init__(self, path=None):
        #self.input = np.reshape(imgs, [-1, imgs_shape[1], imgs_shape[2], imgs_shape[3]])
        #self.input = np.float32(self.input)
        # self.labels=labels
        #self.img_size = imgs_shape
        if path != None:
            self.weights_path = path
        self.weights = self.imagenet_weights(self.weights_path)

    def imagenet_weights(self, weight_file_path):
        parameters = np.load(weight_file_path)
        self.weights = {}
        keys = sorted(parameters.keys())
        for i in range(len(keys)):
            self.weights[keys[i]] = parameters[keys[i]]
        return self.weights

    def build_network(self):
        # Conv Layers
        outs = {}
        # Conv Layers
        input_ = tf.Variable(np.zeros((1, 500,500,3)), dtype = 'float32')
        outs['input'] = input_
        #conv1_1
        kernel1_1 = self.weights["conv1_1_W"]
        bias1_1 = self.weights["conv1_1_b"]
        self.conv1_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_, kernel1_1,
                                                              [1, 1, 1, 1], padding="SAME"), bias1_1))
        outs["conv1_1"] = self.conv1_1

        # conv1_2
        kernel1_2 = self.weights["conv1_2_W"]
        bias1_2 = self.weights["conv1_2_b"]
        self.conv1_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv1_1, kernel1_2,
                                                              [1, 1, 1, 1], padding="SAME"), bias1_2))
        outs["conv1_2"] = self.conv1_2
        # pool1

        self.pool1 = tf.nn.avg_pool2d(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding="SAME", name='pool1')
        outs["pool1"] = self.pool1

        # conv2_1_
        kernel2_1 = self.weights['conv2_1_W']
        bias2_1 = self.weights['conv2_1_b']
        self.conv2_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.pool1, kernel2_1,
                                                              [1, 1, 1, 1], padding="SAME"), bias2_1))
        outs["conv2_1"] = self.conv2_1
        # conv2_2
        kernel2_2 = self.weights['conv2_2_W']
        bias2_2 = self.weights['conv2_2_b']
        self.conv2_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv2_1, kernel2_2,
                                                              [1, 1, 1, 1], padding="SAME"), bias2_2))
        outs["conv2_2"] = self.conv2_2

            # pool2
        self.pool2 = tf.nn.avg_pool2d(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding="SAME", name='pool2')
        outs["pool2"] = self.pool2

            # CONV_3_1
        kernel3_1 = self.weights["conv3_1_W"]
        bias3_1 = self.weights["conv3_1_b"]
        self.conv3_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.pool2, kernel3_1,
                                                              [1, 1, 1, 1], padding="SAME"), bias3_1))
        outs["conv3_1"] = self.conv3_1
        # CONV_3_2
        kernel3_2 = self.weights["conv3_2_W"]
        bias3_2 = self.weights["conv3_2_b"]
        self.conv3_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv3_1, kernel3_2,
                                                              [1, 1, 1, 1], padding="SAME"), bias3_2))
        outs['conv3_2'] = self.conv3_2
        # CONV_3_3
        kernel3_3 = self.weights["conv3_3_W"]
        bias3_3 = self.weights["conv3_3_b"]
        self.conv3_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv3_2, kernel3_3,
                                                              [1, 1, 1, 1], padding="SAME"), bias3_3))
        outs['conv3_3'] = self.conv3_3
            # pool3
        self.pool3 = tf.nn.avg_pool2d(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding="SAME", name='pool3')
        outs["pool3"] = self.pool3

            # CONV_4_1
        kernel4_1 = self.weights["conv4_1_W"]
        bias4_1 = self.weights["conv4_1_b"]
        self.conv4_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.pool3, kernel4_1,
                                                              [1, 1, 1, 1], padding="SAME"), bias4_1))
        outs['conv4_1'] = self.conv4_1
        # CONV_4_2

        kernel4_2 = self.weights["conv4_2_W"]
        bias4_2 = self.weights["conv4_2_b"]
        self.conv4_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv4_1, kernel4_2,
                                                              [1, 1, 1, 1], padding="SAME"), bias4_2))
        outs['conv4_2'] = self.conv4_2
        # CONV_4_3
        kernel4_3 = self.weights["conv4_3_W"]
        bias4_3 = self.weights["conv4_3_b"]
        self.conv4_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv4_2, kernel4_3,
                                                              [1, 1, 1, 1], padding="SAME"), bias4_3))
        outs['conv4_3'] = self.conv4_3
            # pool4

        self.pool4 = tf.nn.avg_pool2d(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding="SAME", name='pool4')
        outs["pool4"] = self.pool4

            # CONV_5_1

        kernel5_1 = self.weights["conv5_1_W"]
        bias5_1 = self.weights["conv5_1_b"]
        self.conv5_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.pool4, kernel5_1,
                                                              [1, 1, 1, 1], padding="SAME"), bias5_1))
        outs['conv5_1'] = self.conv5_1
        # CONV_5_2

        kernel5_2 = self.weights["conv5_2_W"]
        bias5_2 = self.weights["conv5_2_b"]
        self.conv5_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv5_1, kernel5_2,
                                                              [1, 1, 1, 1], padding="SAME"), bias5_2))
        outs['conv5_2'] = self.conv5_2

        # CONV_5_3

        kernel5_3 = self.weights["conv5_3_W"]
        bias5_3 = self.weights["conv5_3_b"]
        self.conv5_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv5_2, kernel5_3,
                                                              [1, 1, 1, 1], padding="SAME"), bias5_3))
        outs['conv5_3'] = self.conv5_3


            # pool5
        self.pool5 = tf.nn.avg_pool2d(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding="SAME", name='pool5')
        outs["pool5"] = self.pool5

        return outs


    def plot_features(self,layer_name,subplots=False):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            layer = str(self) + "." + layer_name
            imgs = sess.run(self.conv3_1, feed_dict={X: self.input.reshape(1, 224, 224, 3)})
        fig, ax = plt.subplots(7, 7, figsize=(20, 20))
        if subplots:
            r=0
            style.use('dark_background')
            for i in range(ax.shape[0]):
                for j in range(ax.shape[1]):
                    fig.tight_layout(pad=3.0)
                    ax[i,j].imshow(imgs[0,:,:,r])
                    r+=1
            plt.show()
        else:
            plt.imshow(imgs[0, :, :, 33])
            plt.show()

