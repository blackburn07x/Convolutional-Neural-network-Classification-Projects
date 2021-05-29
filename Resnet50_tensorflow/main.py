import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

from resnet import *
from layers import *

#TRAINING MNSIT DIGIT AS TEST DATASET FOR THE NETWORK

#placeholders
X = tf.placeholder(tf.float32,shape=[None,28,28,1])
Y =  tf.placeholder(tf.float32,shape=[None,10])
from tensorflow.keras.utils import to_categorical

#DATASET
from tensorflow.keras.datasets import mnist
# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

trainX=np.float32(trainX)
testX = np.float32(testX)

trainX = np.reshape(trainX,(-1,28,28,1))
testX = np.reshape(testX,(-1,28,28,1))

trainy = to_categorical(trainy)
testy = to_categorical(testy)

res = Resnet50(X)
outs,preds = res.build_network_(X)
learning_rate = 1e-4
entropy = tf.nn.softmax_cross_entropy_with_logits(logits = preds,labels = Y)
loss = tf.reduce_mean(entropy)

c_p = tf.equal(tf.argmax(preds,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(c_p,tf.float32))
#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

def mini_batches(train_images,train_labels,batch_size=128):
    m = len(train_images)
    permutation = list(np.random.permutation(m))
    shuffled_X = train_images[permutation,:,:,:]
    shuffled_Y = train_labels[permutation,:]
    num_batches = int(np.floor(m/batch_size))
    batches=[]
    for i in range(num_batches):
        batch_x = shuffled_X[i * batch_size : i * batch_size + batch_size,:,:,:]
        batch_y = shuffled_Y[i * batch_size :i*batch_size + batch_size,:]
        batches.append([batch_x,batch_y])
    return batches


#Training loop
nepochs=10
batch_size=64
closs=[] #for plotting losses
cacc = [] #for plotting accuracies
n_batches = (trainX.shape[0]//batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(nepochs):
        batch_loss=0.0
        batch_accuracy = 0.0
        minibatches = mini_batches(trainX, trainy, batch_size=64)
        for minibatch in minibatches:
            batch_x,batch_y = minibatch
            feed_dict_train = {X:batch_x,Y:batch_y}
            b_loss,_ = sess.run([loss,optimizer],feed_dict= feed_dict_train)
            train_acc = accuracy.eval(feed_dict = {X:batch_x, Y:batch_y})
            batch_loss+= b_loss
            batch_accuracy +=train_acc
        average_loss = batch_loss /n_batches * 100
        average_acc = batch_accuracy / n_batches * 100
        closs.append(average_loss)
        cacc.append(average_acc)
        print("Epoch: {0} ==> \n TRAIN LOSS = {1:0.6f} TRAIN ACCURACY: {2:0.6f}".format(epoch + 1,average_loss,average_acc))
    #test_accuracy
    test_acc = accuracy.eval(feed_dict={X:testX,Y:testy})
    print("\nTEST ACCURACY: {}".format(test_acc * 100))