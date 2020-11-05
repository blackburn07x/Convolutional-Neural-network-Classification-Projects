# -*- coding: utf-8 -*-
"""Chest X-Ray Images (Pneumonia).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VxR2ok3P0ooveFtGlFLQXYqOy_DOK88j
"""

import os
os.environ["KAGGLE_CONFIG_DIR"] = '/content/drive/My Drive/Datasets'

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My Drive/Datasets

!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

!unzip \*.zip && rm *.zip

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


def read_image(img):
    img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    img=np.array(img)
    img=cv2.resize(img,(150,150))
    img = img/255.0
    return img

train_data=[]
import os
for dirname, n, filenames in os.walk('/content/drive/My Drive/Datasets/chest_xray/chest_xray/train/PNEUMONIA'):
    for file in filenames:
        if file == ".DS_Store":
            continue
        train_data.append([read_image(os.path.join("/content/drive/My Drive/Datasets/chest_xray/chest_xray/train/PNEUMONIA",file)),1])
print("ONE")  
for dirname, n, filenames in os.walk('/content/drive/My Drive/Datasets/chest_xray/chest_xray/train/NORMAL'):
    for file in filenames:
        if file == ".DS_Store":
            continue
        train_data.append([read_image(os.path.join('/content/drive/My Drive/Datasets/chest_xray/chest_xray/train/NORMAL',file)),0])

test_data=[]
for dirname, n, filenames in os.walk('/content/drive/My Drive/Datasets/chest_xray/chest_xray/test/PNEUMONIA'):
    for file in filenames:
        test_data.append([read_image(os.path.join("/content/drive/My Drive/Datasets/chest_xray/chest_xray/test/PNEUMONIA",file)),1])

for dirname, n, filenames in os.walk('/content/drive/My Drive/Datasets/chest_xray/chest_xray/test/NORMAL'):
    for file in filenames:
        if file == ".DS_Store":
            continue
        test_data.append([read_image(os.path.join("/content/drive/My Drive/Datasets/chest_xray/chest_xray/test/NORMAL",file)),0])

val_data=[]
for dirname,n,filenames in os.walk("/content/drive/My Drive/Datasets/chest_xray/val/PNEUMONIA"):
  for file in filenames:
        val_data.append([read_image(os.path.join("/content/drive/My Drive/Datasets/chest_xray/val/PNEUMONIA",file)),1])

for dirname, n, filenames in os.walk('/content/drive/My Drive/Datasets/chest_xray/val/NORMAL'):
    for file in filenames:
        if file == ".DS_Store":
            continue
        val_data.append([read_image(os.path.join("/content/drive/My Drive/Datasets/chest_xray/val/NORMAL",file)),0])

print("TOTAL TRAIN IMAGES: ",len(train_data))
print("TOTAL TEST IMAGES: ",len(test_data))
print("TOTAL VAL IMAGES: ",len(val_data))



def generate_data(data):
    imgs= []
    labels=[]
    for i in (data):
        imgs.append(i[0])
        labels.append(i[1])
    return imgs,labels

train_images,train_labels = generate_data(train_data)
test_images,test_labels=generate_data(test_data)
val_images,val_labels=generate_data(val_data)



train_images = np.array(train_images)
train_images = train_images.reshape(-1,150,150,1)
train_labels=np.array(train_labels)

test_images = np.array(test_images)
test_images = test_images.reshape(-1,150,150,1)
test_labels=np.array(test_labels)

val_images = np.array(val_images)
val_images = val_images.reshape(-1,150,150,1)
val_labels=np.array(val_labels)



train_images = np.float32(train_images)
train_labels=np.float32(train_labels)

test_images = np.float32(test_images)
test_labels = np.float32(test_labels)

val_images = np.float32(val_images)
val_labels = np.float32(val_labels)



from tensorflow.python.keras.utils import to_categorical
train_labels=to_categorical(train_labels)
test_labels = to_categorical(test_labels)
val_labels = to_categorical(val_labels)



print("TRAIN IMAGES SHAPE: ",train_images.shape)
print("TEST IMAGES SHAPE: ",test_images.shape)
print("VAL IMAGES SHAPE: ",val_images.shape)

r=0
from matplotlib import style
style.use('dark_background')
fig,ax = plt.subplots(5,5,figsize=(20,20))
for i in range(ax.shape[0]):
    for u in range(ax.shape[1]):
        r=np.random.randint(1,500)
        ax[i,u].imshow(np.squeeze(test_images[r]),cmap='gray')
        if int(np.argmax(test_labels[r])) ==0:
            title = "NORMAL"
        elif int(np.argmax(test_labels[r])) ==1:
            title="PNEUMONIA"
        ax[i,u].set_title(title)

def conv2d(X,weights,bias,padding="SAME",strides = 1):
    z = tf.nn.conv2d(X,weights,strides=[1,strides,strides,1],padding="SAME")
    z = tf.nn.bias_add(z,bias)
    activation = tf.nn.relu(z)
    return activation

def weights_initializer(nclasses):
    weights = {"W1" : tf.get_variable("W1",shape =(3,3,1,32),
                                     initializer = tf.contrib.layers.xavier_initializer()),
              "W2":tf.get_variable("W2",shape = (3,3,32,64),
                                  initializer = tf.contrib.layers.xavier_initializer()),
              "W3":tf.get_variable("W3",shape = (3,3,64,128),
                                  initializer = tf.contrib.layers.xavier_initializer()),
               #"W4":tf.get_variable("W4",shape=(3,3,128,128),
                                   #initializer = tf.contrib.layers.xavier_initializer()),
              "D1":tf.get_variable("D1",shape = (19*19*128,128),
                                  initializer = tf.contrib.layers.xavier_initializer()),
              "D2":tf.get_variable("D2",shape = (128,512),
                                  initializer = tf.contrib.layers.xavier_initializer()),
              "D3":tf.get_variable("D3",shape = (512,nclasses),
                                  initializer = tf.contrib.layers.xavier_initializer())}
    bias = {"B1" :tf.get_variable("B1",shape=(32),initializer = tf.zeros_initializer()),
           "B2":tf.get_variable("B2",shape = (64),initializer = tf.zeros_initializer()),
           "B3":tf.get_variable("B3",shape = (128),initializer = tf.zeros_initializer()),
           #"B4":tf.get_variable("B4",shape = (128),initializer = tf.zeros_initializer()),
           'D1': tf.get_variable('B5', shape=(128), initializer=tf.zeros_initializer()),
           'D2': tf.get_variable('B6', shape=(512), initializer=tf.zeros_initializer()),
           'D3': tf.get_variable('B7', shape=(nclasses), initializer=tf.zeros_initializer()),

}
    return weights,bias

def conv_net(inp_image,weights,bias):
    with tf.name_scope("Network"):
        #Conv_layer
        conv1 = conv2d(inp_image,weights["W1"],bias["B1"])
        pool1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1],strides = [1,2,2,1],padding="SAME")
        conv2 = conv2d(pool1,weights["W2"],bias["B2"])
        pool2 = tf.nn.max_pool(conv2,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")
        conv3 = conv2d(pool2,weights["W3"],bias["B3"])
        pool3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        #conv4 = conv2d(pool3,weights["W4"],bias["B4"])
       # pool4 = tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        
        #Flatten 
        flatten = tf.contrib.layers.flatten(pool3)

        #Dense
        Z1 = tf.add(tf.matmul(flatten,weights["D1"]),bias["D1"])
        A1 = tf.nn.relu(Z1)
        A1 = tf.nn.dropout(A1, keep_prob=1-0.25)
        
        Z2 = tf.add(tf.matmul(A1,weights["D2"]),bias["D2"])
        A2 = tf.nn.relu(Z2)
        A2 = tf.nn.dropout(A2,keep_prob=1-0.25)
        
        Z3 = tf.add(tf.matmul(A2,weights["D3"]),bias["D3"])

    return Z3

def placeholders(img_size,nclasses,nchannels):
    X_train = tf.placeholder(tf.float32,shape=(None,img_size,img_size,nchannels))
    y_train = tf.placeholder(tf.float32,shape=(None,nclasses))
    return X_train,y_train

def logits_and_loss(X_train,y_train,weights,bias):
    normal = 0
    pneumonia= 0
    for i in train_labels:
        if np.argmax(i) ==0:
            normal+=1
        elif np.argmax(i)==1:
            pneumonia+=1
    weights_for_normal = (1 / normal)*(len(train_images))/2.0
    weights_for_pneumonia = (1 / pneumonia)*(len(train_images))/2.0
    class_weight = tf.constant([weights_for_normal, weights_for_pneumonia])
    logits = conv_net(X_train,weights,bias)
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = logits,
                                                                    labels = y_train,pos_weight = class_weight))
    return loss,logits

from tqdm import tqdm
import math
from tensorflow.python.framework import ops
ops.reset_default_graph()
def train(train_images,train_labels,test_images,test_labels,
         learning_rate = 0.001,epochs=5,batch_size=64):
    nclasses=2
    img_size = 150
    nchannels = train_images.shape[-1]
    #initialize weights 
    weights,bias= weights_initializer(nclasses)
    #Placeholder
    X_train,y_train = placeholders(img_size,nclasses,nchannels)
    loss,_ = logits_and_loss(X_train,y_train,weights,bias)
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    tf.summary.scalar("Training_Loss_:",loss)
    A2 =  conv_net(X_train,weights,bias)
    test_predictions = tf.nn.softmax(A2)
    correct_prediction = tf.equal(tf.argmax(y_train, 1), tf.argmax(test_predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("Validation_accuracy_:",accuracy)

    merge_summary=tf.summary.merge_all()
    init=tf.global_variables_initializer()

    with tf.Session() as sess:  
        sess.run(init)
        summary_writer = tf.summary.FileWriter('graphs/',graph = tf.get_default_graph())
        for i in range(epochs):
            minibatch_cost = 0
            minibatches=random_mini_batches(train_images, train_labels,mini_batch_size = 128)
            for minibatch in minibatches:
                batch_images,batch_labels =minibatch 
                feed_dict = {X_train:batch_images,y_train:batch_labels}
                _,temp_cost,summary = sess.run([optimizer,loss,merge_summary],feed_dict = feed_dict)
                minibatch_cost += temp_cost / len(minibatches)
                summary_writer.add_summary(summary,i)
            val_acc = sess.run(accuracy,feed_dict={X_train: val_images, y_train: val_labels})
            
            print('\nEpoch {} \n Loss: {} Validation Accuracy: {}'.format(i+1, minibatch_cost,val_acc))

train(train_images,train_labels,test_images,test_labels,
         learning_rate = 1e-1,epochs=25)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir 'graphs/'

!kill 1575

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

