"""
Copyright (c) 2018 Grant Perkins
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time
from versions.depthview import Dataset


class SparseAutoEncoder:
    """
    SparseAutoEncoder is an implementation of a sparse autoencoder for the MNIST dataset. This implementation
    defines an encoding function that reduces an image to its primitive features. It is reduced from 28x28 to 10x10.
    """

    def __init__(self, length, width, rho=0.01, theta=.0001, beta=3):
        """
        Sets all constants and initializes weights and biases
        :param input_size: Size of input and output layer (1D)
        :param hidden_size: Size of hidden layer (1D)
        :param rho: Desired average sparsity
        :param theta: Weight decay parameter, actually lambda in papers
        :param beta: Weight of sparsity penalty term
        """
        self.length, self.width = length, width
        self.input_size = length * width
        self.hidden_size = length * width
        self.rho = rho
        self.theta = theta
        self.beta = beta

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        # print("SAE optimizer constructed")

        with tf.variable_scope("Variables"):
            self.W1 = self._new_variable([self.input_size, self.hidden_size], "Weight1")
            self.b1 = self._new_variable([1, self.hidden_size], "Bias1")

            self.W2 = self._new_variable([self.hidden_size, self.input_size], "Weight2")
            self.b2 = self._new_variable([1, self.input_size], "Bias2")

        # print("SAE Weights and biases constructed")

        self.sess = tf.Session()
        # print("SAE session constructed")
        self.train_writer = tf.summary.FileWriter("saefilewriter/")
        self.saver = tf.train.Saver(name="Save")
        # print("Done constructing")

    def _new_variable(self, shape, name):
        """
        Initializes a variable of given shape with random contents
        :param shape: Shape of the tensor
        :return: a tf.Variable of given shape, random contents
        """
        values = tf.random_normal(shape, stddev=.05)
        return tf.Variable(values, name=name)

    def _encode(self, X):
        """
        Encodes data X into a smaller layer
        :param X: images as a tensor
        :return: encoded layer
        """
        with tf.variable_scope("Encode"):
            return tf.nn.sigmoid(tf.matmul(X, self.W1) + self.b1, name="sigmoid")

    def _decode(self, H):
        """
        Decodes encoded data H into a larger layer
        :param H: encoded images as tensor
        :return: decoded layer
        """
        with tf.variable_scope("Decode"):
            return tf.nn.sigmoid(tf.matmul(H, self.W2) + self.b2, name="sigmoid")

    def _kl_divergence(self, rho_hat):
        """
        Computes the Kullback-Leibler divergence
        Divergence is between rho and rho hat
        :param rho_hat: average activation of all nodes in encoding layer
        :return: KL divergence value
        """
        with tf.variable_scope("KLDivergence"):
            return self.rho * (tf.log(self.rho + 1e-10) - tf.log(rho_hat + 1e-10)) + (1 - self.rho) * (
                    tf.log((1 - self.rho + 1e-10)) - tf.log(1 - rho_hat + 1e-10))

    def _cost(self, X):
        """
        Computes current cost of the autoencoder
        :param X: input images
        :return: current cost
        """
        with tf.variable_scope("Cost"):
            H = self._encode(X)
            X_hat = self._decode(H)

            with tf.variable_scope("Loss"):
                diff = X - X_hat
                rho_hat = tf.reduce_mean(H, axis=0, name="rho_hat")
                kl = self._kl_divergence(rho_hat)
                cost = .5 * tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1)) \
                       + .5 * self.theta * (tf.reduce_sum(self.W1 ** 2) + tf.reduce_sum(self.W2 ** 2)) \
                       + self.beta * tf.reduce_sum(kl)
                return cost

    def train(self, data, train_steps=3000):
        # 1365.569870710373
        """
        Trains autoencoder
        :param data: all input images
        :param train_steps: amount of training steps.
        :return: None
        """
        print("Training started")
        start = time()
        batch_size = 10

        images = np.array(data.frames)
        print(images.shape)
        print("All concatenated")
        print("Images cropped to ", images.shape)
        images = np.reshape(images, [images.shape[0], self.length * self.width])

        X = tf.placeholder(tf.float32, shape=[None, self.length * self.width], name="X")
        cost = self._cost(X)
        optimizer = self.optimizer.minimize(cost)
        self.sess.run(tf.global_variables_initializer())
        self.train_writer.add_graph(self.sess.graph)
        for step in range(train_steps):
            try:
                for batch in range(100):
                    x_batch = data.next_batch(batch_size, images)
                    # x_batch = np.array(data.next_batch(batch_size))
                    self.sess.run(optimizer, feed_dict={X: x_batch})
            except:
                print("recycling data")
                data.restart()
            print("Seconds since training started:", time() - start, "Step", step)
        save_path = self.saver.save(self.sess, "/var/sae.cpkt")
        print("Model saved to", save_path)

    def save_weight_picture(self, plots_len=4, file_name="trained_weights_2.png"):
        """
        Saves some of the trained weights of the encoding layer as an image
        :param image_len: length of a square image
        :param plots_len: length of the square grid of plots
        :param file_name: name of output file
        :return: None
        """
        images = self.W1.eval(self.sess)
        images = images.transpose()

        figure, axes = plt.subplots(nrows=1, ncols=4)
        for i, axis in enumerate(axes.flat):
            axis.imshow(images[i + 500, :].reshape(self.length, self.width), cmap=plt.get_cmap("binary"))
            axis.set_axis_off()
        plt.savefig(file_name)
        print("Picture of weights saved to", file_name)
        plt.imshow(images[476, :].reshape(self.length, self.width), cmap=plt.get_cmap("binary"))

        file_name = file_name[:-4] + "2" + ".png"
        plt.savefig(file_name)
        print("Picture of weights saved to", file_name)
        plt.close()

    def encode(self, image):
        """
        Encodes image
        :param image: a tensor containing an image
        :return: encoded image
        """
        return self.sess.run(self._encode(image))


def main():
    """
    Creates and trains sparse autoencoder
    :return: None
    """
    length, width = 40, 40
    # data = Dataset(length, width)
    sae = SparseAutoEncoder(length, width)
    # sae.train(data)
    sae.saver.restore(sae.sess, "/var/sae.cpkt")
    sae.save_weight_picture(2)


if __name__ == '__main__':
    main()
