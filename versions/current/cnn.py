import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ConvolutionalNeuralNetwork:
    """
    Convolves give image with trained weights from SAE
    """

    def __init__(self, data, sae, length, width):
        self.sae = sae
        self.length = length
        self.width = width

        filters = tf.reshape(sae.W1, [length, width, 1, length * width])
        image = tf.placeholder(tf.float32, shape=[None, 240, 320, 1], name="Image")
        conv = self.conv_layer(image, filters, "Conv")
        pool = self.avg_pool_layer(conv, 5, 1, "Pool")
        self.images = []
        self.frames = data.reshape(data.shape[0], 240, 320)
        # print(data.shape)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for frame in data:
                # print("Convolving with pre-trained filters")
                # print(frame.shape, frame.size)
                frame = frame.reshape(1, 240, 320, 1)
                img = np.squeeze(sess.run(pool, feed_dict={image: frame}))
                new_max = [[0 for _ in range(len(img[0]))] for _ in range(len(img))]
                # print("Max and average pooling")
                # print(img[0][0].size)
                for y in range(len(img)):
                    for x in range(len(img[y])):
                        new_max[y][x] = np.amax(img[y][x])
                new_max = np.array(new_max)
                self.images.append(new_max.reshape(240, 320))
                plt.axis('off')
                fig = plt.imshow(new_max.reshape(240,320), cmap=plt.get_cmap("binary"))
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.savefig("poster/cnn.png", bbox_inches="tight", pad_inches=0)

    def conv_layer(self, input, filters, name):
        """
        Creates new convolutional layer from filter
        :param input: input image
        :param filters: filters from sae
        :param name: name of layer
        :return: convolutional layer
        """
        with tf.variable_scope(name):
            layer = tf.nn.conv2d(input=input, filter=filters, strides=[1, 1, 1, 1], padding="SAME")
            return layer

    def avg_pool_layer(self, input, size, stride, name):
        """
        Downsamples convolved image to a more reasonable size, using average pooling
        :param input: input image
        :param size: size of filter (makes a square)
        :param stride: stride of filter
        :param name: name of layer
        :return: average pooling layer
        """
        with tf.variable_scope(name):
            size = [1, size, size, 1]
            stride = [1, stride, stride, 1]
            layer = tf.nn.avg_pool(input, size, strides=stride, padding="SAME", name=name)
            return layer
