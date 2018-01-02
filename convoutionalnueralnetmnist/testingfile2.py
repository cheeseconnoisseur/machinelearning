import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 128

x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def conv2d(x, w):
    #this means that it will take it one pixel at a time
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #this means it will take the window for pooling two pixels at a time
    #ksize is the size of the window 2*2 whilst strides is how much it moves 2px
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding ='SAME')

def convolusional_neural_network(x):

    #5,5 is the convution size and 1,32 is one input and 32 outputs/features
    weights = {'w_con_1':tf.Variable(tf.random_normal([5,5,1,32])),
    #input from first covultion will now be 32 and output 64
               'w_con_2':tf.Variable(tf.random_normal([5,5,32,64])),
               #image now 7 by 7 time by number of features
               'w_fullyconnect':tf.Variable(tf.random_normal([7*7*64,1024])),
               'output':tf.Variable(tf.random_normal([1024, n_classes]))}

#biases are just to add on to the output so you just need a bias for every
#output of a nueron whih is the final number in the above dictionary making
#the biases what they are below
    biases = {'b_con_1':tf.Variable(tf.random_normal([32])),
               'b_con_2':tf.Variable(tf.random_normal([64])),
               'b_fullyconnect':tf.Variable(tf.random_normal([1024])),
               'boutput':tf.Variable(tf.random_normal([n_classes]))}

    #rehape x into a 28*28*1 vector for tf
    print("gay1")
    x =tf.reshape(x, shape=[-1,28,28,1])
    print("gay2")

    #this does the first convolution
    conv1 = conv2d(x, weights['w_con_1']) + biases['b_con_1']
    print("gay3")
    #this does the first pooling
    conv1 = maxpool2d(conv1)
    print("gay4")
    conv2 = conv2d(conv1, weights['w_con_2']) + biases['b_con_2']
    print("gay5")
    conv2 = maxpool2d(conv2)
    print("gay6")


    fc = tf.reshape(conv2, [-1,7*7*64])
    print("gay7")
    fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['w_fullyconnect']), biases['b_fullyconnect']))
    print("gay8")

    output = tf.matmul(fc, weights['output'])+biases['boutput']
    print("gay9")

    return output


def train_neural_network(x):
    prediction = convolusional_neural_network(x)
    print("gay10")
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    print("gay11")
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    print("gay12")
    hm_epochs = 10

    with tf.Session() as sess:
        print("gay12")
        sess.run(tf.global_variables_initializer())
        print("gay13")
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('epoch',epoch, 'completed out of', hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
