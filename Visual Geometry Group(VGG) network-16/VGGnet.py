#VGG network using Tensorflow 1.x
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Network
# Network
class VGG16net:

    def __init__(self, imgs, imgs_shape, path=None):
        self.input = np.reshape(imgs, [-1, imgs_shape[0], imgs_shape[1], imgs_shape[2]])
        self.input = np.float32(self.input)
        # self.labels=labels
        self.img_size = imgs_shape
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
        with tf.name_scope("conv_block_1") as scope:
            # conv1_1
            print("\nPassing Image through Conv_1_1")
            print("Filter_shape: ", self.weights["conv1_1_W"].shape)
            kernel1_1 = self.weights["conv1_1_W"]
            bias1_1 = self.weights["conv1_1_b"]
            self.conv1_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.input, kernel1_1,
                                                                  [1, 1, 1, 1], padding="SAME"), bias1_1))

            # conv1_2
            print("\nPassing Image through Conv_1_2")
            print("Filter_shape: ", self.weights["conv1_2_W"].shape)
            kernel1_2 = self.weights["conv1_2_W"]
            bias1_2 = self.weights["conv1_2_b"]
            self.conv1_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv1_1, kernel1_2,
                                                                  [1, 1, 1, 1], padding="SAME"), bias1_2))

        with tf.name_scope('pool_block_1') as scope:
            # pool1
            print("\nPOOL 1")
            self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                        padding="SAME", name='pool1')

        with tf.name_scope("conv_block_2") as scope:
            # conv2_1_
            print("\nPassing Image through Conv_2_1")
            print("Filter_shape: ", self.weights["conv2_1_W"].shape)
            kernel2_1 = self.weights['conv2_1_W']
            bias2_1 = self.weights['conv2_1_b']
            self.conv2_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.pool1, kernel2_1,
                                                                  [1, 1, 1, 1], padding="SAME"), bias2_1))

            # conv2_2
            print("\nPassing Image through Conv_2_2")
            print("Filter_shape: ", self.weights["conv2_2_W"].shape)
            kernel2_2 = self.weights['conv2_2_W']
            bias2_2 = self.weights['conv2_2_b']
            self.conv2_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv2_1, kernel2_2,
                                                                  [1, 1, 1, 1], padding="SAME"), bias2_2))

        with tf.name_scope('pool_block_2') as scope:
            # pool1
            print("\nPOOL 2")
            self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                        padding="SAME", name='pool2')

        with tf.name_scope('conv_block_3') as scope:
            # CONV_3_1
            print("\nPassing Image through Conv_3_1")
            print("Filter_shape: ", self.weights["conv3_1_W"].shape)
            kernel3_1 = self.weights["conv3_1_W"]
            bias3_1 = self.weights["conv3_1_b"]
            self.conv3_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.pool2, kernel3_1,
                                                                  [1, 1, 1, 1], padding="SAME"), bias3_1))

            # CONV_3_2
            print("\nPassing Image through Conv_3_2")
            print("Filter_shape: ", self.weights["conv3_2_W"].shape)
            kernel3_2 = self.weights["conv3_2_W"]
            bias3_2 = self.weights["conv3_2_b"]
            self.conv3_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv3_1, kernel3_2,
                                                                  [1, 1, 1, 1], padding="SAME"), bias3_2))

            # CONV_3_3
            print("\nPassing Image through Conv_3_3")
            print("Filter_shape: ", self.weights["conv3_3_W"].shape)
            kernel3_3 = self.weights["conv3_3_W"]
            bias3_3 = self.weights["conv3_3_b"]
            self.conv3_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv3_2, kernel3_3,
                                                                  [1, 1, 1, 1], padding="SAME"), bias3_3))

        with tf.name_scope("pool3") as scope:
            # pool1
            print("\nPOOL 3")
            self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                        padding="SAME", name='pool3')

        with tf.name_scope('conv_block_3') as scope:
            # CONV_4_1
            print("\nPassing Image through Conv_4_1")
            print("Filter_shape: ", self.weights["conv4_1_W"].shape)
            kernel4_1 = self.weights["conv4_1_W"]
            bias4_1 = self.weights["conv4_1_b"]
            self.conv4_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.pool3, kernel4_1,
                                                                  [1, 1, 1, 1], padding="SAME"), bias4_1))

            # CONV_3_2
            print("\nPassing Image through Conv_4_2")
            print("Filter_shape: ", self.weights["conv4_2_W"].shape)

            kernel4_2 = self.weights["conv4_2_W"]
            bias4_2 = self.weights["conv4_2_b"]
            self.conv4_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv4_1, kernel4_2,
                                                                  [1, 1, 1, 1], padding="SAME"), bias4_2))

            # CONV_3_3
            print("\nPassing Image through Conv_4_3")
            print("Filter_shape: ", self.weights["conv4_3_W"].shape)
            kernel4_3 = self.weights["conv4_3_W"]
            bias4_3 = self.weights["conv4_3_b"]
            self.conv4_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv4_2, kernel4_3,
                                                                  [1, 1, 1, 1], padding="SAME"), bias4_3))

        with tf.name_scope("pool4") as scope:
            # pool1
            print("\nPOOl 4")
            self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                        padding="SAME", name='pool4')

        with tf.name_scope('conv_block_4') as scope:
            # CONV_4_1
            print("\nPassing Image through Conv_5_1")
            print("Filter_shape: ", self.weights["conv5_1_W"].shape)

            kernel5_1 = self.weights["conv5_1_W"]
            bias5_1 = self.weights["conv5_1_b"]
            self.conv5_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.pool4, kernel5_1,
                                                                  [1, 1, 1, 1], padding="SAME"), bias5_1))

            # CONV_3_2
            print("Passing Image through Conv_5_2\n")
            print("Filter_shape: ", self.weights["conv5_2_W"].shape)

            kernel5_2 = self.weights["conv5_2_W"]
            bias5_2 = self.weights["conv5_2_b"]
            self.conv5_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv5_1, kernel5_2,
                                                                  [1, 1, 1, 1], padding="SAME"), bias5_2))

            # CONV_5_3
            print("Passing Image through Conv_5_3\n")
            print("Filter_shape: ", self.weights["conv5_3_W"].shape)
            kernel5_3 = self.weights["conv5_3_W"]
            bias5_3 = self.weights["conv5_3_b"]
            self.conv5_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv5_2, kernel5_3,
                                                                  [1, 1, 1, 1], padding="SAME"), bias5_3))
        # Visualising Image
        """with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            img = sess.run(self.conv3_2, feed_dict={X: self.input.reshape(1, 224, 224, 3)})
        plt.imshow(img[0,:,:,33])
        plt.show()"""

        with tf.name_scope("pool5") as scope:
            # pool1
            print("\nPOOL 5")
            self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                        padding="SAME", name='pool5')

        with tf.name_scope("dense_block_1") as scope:
            print("Passing Image through Dense Block 1\n")
            flatten = tf.contrib.layers.flatten(self.pool5)
            # flatten = np.reshape(self.pool5,(-1,self.pool5.shape[1] * self.pool5.shape[2] * self.pool5.shape[3]))
            fc1w = self.weights["fc6_W"]
            fc1b = self.weights["fc6_b"]
            self.fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flatten, fc1w), fc1b))

        with tf.name_scope('dense_block_2') as scope:
            print("Passing Image through Dense Block 1\n")
            fc2w = self.weights['fc7_W']
            fc2b = self.weights['fc7_b']
            self.fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b))

        with tf.name_scope('dense_block_3') as scope:
            print("Passing Image through Dense Block 1\n")
            fc3w = self.weights["fc8_W"]
            fc3b = self.weights['fc8_b']
            self.fc3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b))
        print("\n IMAGE PASS SUCCESSFUL,FUCK YEAH!")


# PASSING DUMMY IMAGE FOR TESTING
parameters = np.load('vgg16_weights.npz')
X = tf.placeholder(tf.float32, [None, 224, 224, 3])
img = cv2.imread("cat.jpg")
img = np.array(img)
img = cv2.resize(img, (224, 224))
img = img / 255.0
# network pass
model = VGG16net(img, img.shape, path='vgg16_weights.npz')
model = model.build_network()

