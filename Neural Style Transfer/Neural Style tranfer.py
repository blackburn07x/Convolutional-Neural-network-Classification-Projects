#neural Style Transfer using Tensorflow 1.x and VGG-16
import cv2
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from VGGnst import *

#utils
height = 500
width = 500
channels = 3

content_image = "download.jpg"
style_image = 'neb.jpg'
weights_path = '/content/drive/MyDrive/vgg16_weights.npz'
means = np.array([123.68,116.779,103.939]).reshape((1,1,1,3))

#Helper functions
def noisy_image(height,width,channels,img):
    image = np.random.uniform(-20,20,(1,height,width,channels)).astype('float32')
    generated_image = image * 0.6 + img * (1- 0.6)
    return generated_image

def preprocess_image(img,height,width):
    img = imageio.imread(img)
    img = img - means
    img = np.reshape(img,[-1,height,width,channels])
    return img

#get the output from VGG network
def get_VGG_output(weights_path):
    net = VGG16net(path = weights_path)
    model = net.build_network()
    return model

def content_loss(features_c,features_g):
    _,h,w,c = features_g.get_shape().as_list()
    content_l=(1/(4 * h * w * c)) * tf.reduce_sum(tf.pow(tf.transpose(features_c) - tf.transpose(features_g),2))
    return content_l


def style_loss(model,sess):
    style_l = 0
    l_activations = ["conv1_1","conv2_1","conv3_1",
                    "conv4_1","conv5_1"]
    for layer in l_activations:
        out = model[layer]
        #style image
        style_s = sess.run(out)
        style_s = tf.transpose(tf.reshape(style_s, [style_s.shape[1] * style_s.shape[2], style_s.shape[3]]))
        #generated image
        style_g = out
        _,h,w,c = style_g.get_shape().as_list()
        style_g = tf.transpose(tf.reshape(style_g,[style_g.shape[1]*style_g.shape[2],style_g.shape[3]]))
        #computing correlation or Gram matrix
        style_sg = tf.matmul(style_s,tf.transpose(style_s))
        style_gg = tf.matmul(style_g,tf.transpose(style_g))

        layer_loss = (1 / (4 * (c) ** 2 * (h * w) ** 2)) * tf.reduce_sum(tf.pow((style_sg - style_gg), 2))
        style_l += (0.2 * layer_loss)
    return style_l

def total_cost(content_l,style_l):
    alpha = 10
    beta=10
    J = alpha * content_l + beta * style_l
    return J


#processing content_image
content_image = preprocess_image(content_image,height,width)
#processing style_image
style_image = preprocess_image(style_image,height,width)
# generated_image
generated_image = noisy_image(height,width,channels,content_image)

def save_image(path, image): # this function is taken from Coursera Deep Learning Course
    # Un-normalize the image so that it looks good
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

    image = image + MEANS

    # Clip and Save the image
    image = np.clip(image[0], 0, 255).astype('uint8')
    imageio.imsave(path, image)

def train(generated_image, content_image, style_image, epochs=1000):
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = get_VGG_output(weights_path)
        image_input = tf.placeholder(tf.float32,[None,height,width,channels])
        sess.run(model['input'].assign(content_image))
        features_a = sess.run(model["conv3_2"],feed_dict={image_input:content_image})
        features_g = model["conv3_2"]
        content_l = content_loss(features_a,features_g)
        #style loss
        sess.run(model['input'].assign(style_image))
        style_l  = style_loss(model,sess)
        #total loss
        J = total_cost(content_l,style_l)
        train_step = tf.train.AdamOptimizer(2.0).minimize(J)
        sess.run(tf.global_variables_initializer())
        #generated image as input
        sess.run(model['input'].assign(generated_image))
        for i in range(epochs):
            sess.run(train_step)
            generated_image = sess.run(model['input'])
            if i % 20 == 0:
                Jt, Jc, Js = J, content_l, style_l
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(Jt.eval()))
                print("content cost = " + str(Jc.eval()))
                print("style cost = " + str(Js.eval()))
                save_image("output/" + str(i) + ".png", generated_image)

        save_image('output/generated_image.jpg', generated_image)
    return generated_image

gi =train(generated_image,content_image,style_image)