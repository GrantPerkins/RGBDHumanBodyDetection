import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data

""" DEFINING LAYER CONSTRUCTORS """


def conv_layer(input, input_channels, filter_size, filters, name):
    """
    Creates a new convolutional layer
    :param input: layer to convolve
    :param input_channels: number of channels in last layer
    :param filter_size: length of square filter
    :param filters: number of filters to be created and output channels
    :param name: name of layer
    :return: new tf.nn.conv2d
    """
    with tf.variable_scope(name) as scope:
        filter_shape = [filter_size, filter_size, input_channels, filters]
        # constructed filters (weights and biases), in the form of a tf.Variable
        # non-zero initialization to prevent equal weights of nodes
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05))
        b = tf.Variable(tf.constant(0.05, shape=[filters]))

        # stride -> [batch, x, y, channel]
        layer = tf.nn.conv2d(input=input, filter=W, strides=[1, 1, 1, 1], padding="SAME")
        # add bias after convolution
        layer += b
        return layer


def max_pool_layer(input, name):
    """
    Creates a new max pooling layer
    :param input: layer to max pool
    :param name: name of layer
    :return: new tf.nn.max_pool
    """
    with tf.variable_scope(name) as scope:
        # down-sampling by 50%
        # stride -> [batch, x, y, channel]
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        return layer


def relu_layer(input, name):
    """
    Creates a new ReLU layer (Rectified Linear Unit)
    :param input: layer to rectify
    :param name: name of layer
    :return: new tf.nn.relu
    """
    with tf.variable_scope(name) as scope:
        # replace negative pixels with 0, negative pixels don't make sense
        layer = tf.nn.relu(input)
        return layer


def fully_connect_layer(input, input_nodes, output_nodes, name):
    """
    Creates a new fully-connected layer
    :param input: layer to connect from
    :param input_nodes: number of nodes in input layer
    :param output_nodes: number of nodes in new layer
    :param name: name of layer
    :return: new layer (tf.matmul(W, x) + b)
    """
    with tf.variable_scope(name) as scope:
        W = tf.Variable(tf.truncated_normal([input_nodes, output_nodes], stddev=0.05))
        b = tf.Variable(tf.constant(0.05, shape=[output_nodes]))

        # y = Wx + b
        # rows[0] == columns[1]
        layer = tf.matmul(input, W) + b
        return layer


# shape = number_images, pixels
x = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='x')
# shape = number_images, height, width, channels
x_image = tf.reshape(x, [-1, 28, 28, 1])
# shape = number_images, labels
y = tf.placeholder(tf.float32, shape=[None, 10], name='y')

""" DEFINING CNN """
# Convolutional layer 1
conv1 = conv_layer(input=x_image, input_channels=1, filter_size=5, filters=6, name="conv1")
print("shape: ",conv1.shape)
# Max pooling layer 1
pool1 = max_pool_layer(input=conv1, name="pool1")
print("shape", pool1.shape)
# ReLU layer 1
relu1 = relu_layer(input=pool1, name="relu1")
# Convolutional layer 2
conv2 = conv_layer(input=relu1, input_channels=6, filter_size=5, filters=16, name="conv2")
# Max pooling layer 2
pool2 = max_pool_layer(input=conv2, name="pool2")
# ReLU layer 2
relu2 = relu_layer(input=pool2, name="relu2")
# Flat layer
num_features = relu2.get_shape()[1:4].num_elements()
flat = tf.reshape(relu2, [-1, num_features])
# Fully-connected layer 1
fc1 = fully_connect_layer(input=flat, input_nodes=num_features, output_nodes=128, name="fc1")
# ReLU layer 3
relu3 = relu_layer(input=fc1, name="relu3")
# Fully-connected layer 2
fc2 = fully_connect_layer(input=relu3, input_nodes=128, output_nodes=10, name="fc2")

""" ANALYSIS AND TRAINING """
with tf.variable_scope("softmax"):
    y_ = tf.argmax(tf.nn.softmax(fc2), 1)

with tf.variable_scope("cros_ent"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=fc2)
    cost = tf.reduce_mean(cross_entropy)

with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

with tf.variable_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(fc2, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_writer = tf.summary.FileWriter("Training_FileWriter/")
valid_writer = tf.summary.FileWriter("Validation_FileWriter/")
saver = tf.train.Saver()

# Add cost and accuracy to summary
tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", accuracy)
merged_summary = tf.summary.merge_all()

train_steps = 100
batch_size = 100
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

""" SESSION """
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # add model to TensorBoard
    print("Started")
    train_writer.add_graph(sess.graph)

    # Restore variables if trained already
    """
    saver.restore(sess, "/var/model.cpkt")
    f, l = data.validation.next_batch(1)
    print("\n\n\n\n\n")
    print("PREDICTED VALUE : " + str(sess.run(y_, feed_dict={x: f, y: l})[0]))
    print(l)

    img = Image.new('1', (28, 28), "white")  # Create a new white image
    pixels = img.load()  # Create the pixel map
    for i in range(img.size[0]):  # For every pixel:
        for j in range(img.size[1]):
            pixels[j, i] = 1 if f[0][i*28+j] != 0 else 0

    img.show()
    """
    # Training and validation
    for step in range(2):
        start = time()
        train_accuracy = 0

        length = int(len(data.train.labels) / batch_size)
        for batch in range(0, length):
            # get next 100 training features and labels
            x_batch, y_batch = data.train.next_batch(batch_size)

            batch_dict = {x: x_batch, y: y_batch}
            sess.run(optimizer, feed_dict=batch_dict)

            train_accuracy += sess.run(accuracy, feed_dict=batch_dict)

            summ = sess.run(merged_summary, feed_dict=batch_dict)
            train_writer.add_summary(summ, step * length + batch)

        train_accuracy /= length

        summ, vali_accuracy = sess.run([merged_summary, accuracy],
                                       feed_dict={x: data.validation.images, y: data.validation.labels})
        valid_writer.add_summary(summ, step)

        end = time()

        print("Step " + str(step + 1) + " completed : Time usage " + str(int(end - start)) + " seconds")
        print("\tAccuracy:")
        print("\t- Training Accuracy:\t{}".format(train_accuracy))
        print("\t- Validation Accuracy:\t{}".format(vali_accuracy))

    save_path = saver.save(sess, "/var/model.cpkt")
    print("Model saved to %s" % save_path)
    # """
    # while 1:pass
