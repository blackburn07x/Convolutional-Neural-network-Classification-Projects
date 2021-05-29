#libraries
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from layers import ops,blocks

#define residual block for resnet50
class Resnet50:
    def __init__(self,x_tensor):
        self.input = x_tensor

    def build_network_(self,x_tensor):
        #Stage 0
        op = ops()
        block = blocks()

        #stage0
        conv = op.conv_(self.input,7,64,2,"SAME")
        bn  =op.batch_norm_(conv)
        act = tf.nn.relu(bn)
        max = op.max_pool_(act,2,2)

        #Stage 1
        conv = block.conv_block(max,[1,3],64,256)
        conv = block.identity_block(conv,[1,3],64,256)
        conv = block.identity_block(conv,[1,3],64,256)

        #Stage 2
        conv = block.conv_block(conv,[1,3],128,512)
        conv = block.identity_block(conv,[1,3],128,512)
        conv = block.identity_block(conv, [1, 3], 128, 512)
        conv = block.identity_block(conv, [1, 3], 128, 512)

        #Stage3
        conv = block.conv_block(conv,[1,3],256,1024)
        conv = block.identity_block(conv,[1,3],256,1024)
        conv = block.identity_block(conv,[1,3],256,1024)
        conv = block.identity_block(conv,[1,3],256,1024)
        conv = block.identity_block(conv,[1,3],256,1024)
        conv = block.identity_block(conv,[1,3],256,1024)

        #Stage4
        conv = block.conv_block(conv,[1,3],512,2048)
        conv = block.identity_block(conv,[1,3],512,2048)
        conv = block.identity_block(conv,[1,3],512,2048)

        #stage6
        conv = op.average_pool_(conv,2,2)
        conv = op.flatten_(conv)
        fc1  = op.fully_connected_(conv,1000)
        fc1 = tf.nn.relu(fc1)
        fc2 = op.fully_connected_(fc1,10)
        out = tf.nn.softmax(fc2)

        return fc2,out


