import tensorflow as tf
import numpy as np
#for test
import cv2

#ops
class ops:
    def __init__(self):
        self.params={}
    def weights_(self, filter_size,in_filter,filters):
        initializer = tf.contrib.layers.xavier_initializer()
        w = tf.Variable(initializer((filter_size,filter_size,in_filter,filters)))
        b = tf.Variable(initializer([filters]))
        return w,b

    def conv_(self, x_tensor: object, filter_size: object, filters: object, stride: object, padding: object) -> object:
        self.in_filter = x_tensor.shape.as_list()[-1]
        #get weights
        W,B = self.weights_(filter_size,self.in_filter,filters)
        self.params["W_" ] = W
        self.params["b_" ] = B
        #Convolve
        conv =  tf.nn.conv2d(x_tensor, W, strides=stride, padding= padding)
        conv = tf.nn.bias_add(conv,B)
        return conv

    def batch_norm_(self, x_tensor):
        bn = tf.layers.batch_normalization(x_tensor)
        self.params["bn_"] =bn
        return bn

    def max_pool_(self,x_tensor,k_size,stride):
        mp = tf.nn.max_pool(x_tensor,ksize=[1,k_size,k_size,1],strides = [1,stride,stride,1],
                            padding="SAME")
        self.params["mp_"] = mp
        return mp

    def average_pool_(self,x_tensor,k_size,stride):
        ap = tf.nn.avg_pool(x_tensor, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                            padding="SAME")
        self.params["ap_" ] = ap
        return ap

    def flatten_(self,x_tensor):
        return tf.layers.flatten(x_tensor)

    def weights_fc(self, in_neurons, out_neurons):
        initializer = tf.contrib.layers.xavier_initializer()
        w = tf.Variable(initializer((in_neurons,out_neurons)))
        b = tf.Variable(initializer([out_neurons]))
        return w,b

    def fully_connected_(self,x_tensor,out_dims):
        in_dims = x_tensor.shape.as_list()[-1]
        fw,fb = self.weights_fc(in_dims,out_dims)
        self.params["fc_w_"] = fw
        self.params["fc_b_" ] = fb

        #fc
        fc = tf.nn.bias_add(tf.matmul(x_tensor,fw),fb)
        self.params["fc_"] = fc
        return fc

    def params_(self):
        return self.params

class blocks:

    def identity_block(self,x_tensor, filter_size, in_channels,out_channels):
        #shortcut
        x_shortcut =x_tensor
        f1,f2 = filter_size
        #block 1
        op = ops()
        conv = op.conv_(x_tensor, f1, in_channels,1,"VALID")
        bn = op.batch_norm_(conv)
        act = tf.nn.relu(bn)

        #block 2
        conv1 = op.conv_(act,f2,in_channels,1,"SAME")
        bn1 = op.batch_norm_(conv1)
        act1 = tf.nn.relu(bn1)

        #block 3
        conv2 = op.conv_(act1,f1,out_channels,1,"VALID")
        bn2 = op.batch_norm_(conv2)

        #Add shortucut
        x_final = tf.keras.layers.Add()([bn2,x_shortcut])
        x_final = tf.nn.relu(x_final)
        return x_final


    def conv_block(self,x_tensor,filter_size, in_channels,out_channels):
        # shortcut
        x_shortcut = x_tensor
        f1, f2 = filter_size
        # block 1
        op = ops()
        conv = op.conv_(x_tensor, f1, in_channels,2, "VALID")
        bn = op.batch_norm_(conv)
        act = tf.nn.relu(bn)

        # block 2
        conv1 = op.conv_(act, f2, in_channels, 1, "SAME")
        bn1 = op.batch_norm_(conv1)
        act1 = tf.nn.relu(bn1)

        # block 3
        conv2 = op.conv_(act1, f1, out_channels, 1, "VALID")
        bn2 = op.batch_norm_(conv2)
        # Add shortucut
        conv_short = op.conv_(x_shortcut,f1,out_channels,2,"VALID")
        short_norm = op.batch_norm_(conv_short)
        x_final = tf.keras.layers.Add()([bn2, short_norm])
        x_final = tf.nn.relu(x_final)
        return x_final

