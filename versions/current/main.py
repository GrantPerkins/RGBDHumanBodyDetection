from versions.current.sparse import SparseAutoEncoder
from versions.current.cnn import ConvolutionalNeuralNetwork
from versions.current.regions import Histogram
from versions.current.slidingwindow import SlidingWindowLocalization
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf


def main(data):
    tf.reset_default_graph()
    length, width = 40, 40
    # print("Creating SAE")
    sae = SparseAutoEncoder(length, width)
    # print("Restoring SAE")
    with tf.Graph().as_default():
        sae.saver.restore(sae.sess, "/var/sae.cpkt")
    # print("Creating CNN")
    cnn = ConvolutionalNeuralNetwork(data, sae, length, width)
    # print("Calculating histogram")
    hist = Histogram(cnn.images)  # , cnn.frame.reshape(240, 320).astype(np.float32))
    centroids = hist.centroids
    # print("Starting sliding window")
    swl = SlidingWindowLocalization(data)  # (cnn.frame.reshape(240, 320))
    boxes = []
    coors = []
    # print(len(cnn.images), len(swl.px))
    for i in range(len(cnn.images)):
        px = swl.px[i]
        py = swl.py[i]

        fig = plt.imshow(cnn.frames[i], cmap=plt.get_cmap("binary"))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig("poster/original.png", bbox_inches="tight", pad_inches=0)
        plt.close()

        # print("px", px, py)
        centroid = min(centroids[i], key=lambda c1: math.sqrt((c1[0] - px) ** 2 + (c1[1] - py) ** 2))
        coors.append(centroid)
        xs, ys = zip(*hist.centroids[i])
        img = cnn.frames[i]
        fig = plt.imshow(img, cmap=plt.get_cmap("binary"))
        plt.scatter(x=xs, y=ys, c='r')
        # plt.scatter(x=centroid[0], y=centroid[1], c='b')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # print(*centroid)
        plt.savefig("poster/centroids_{}.png".format(i), bbox_inches="tight", pad_inches=0)
        plt.close()

        fig = plt.imshow(img, cmap=plt.get_cmap("binary"))
        # plt.scatter(x=xs, y=ys, c='r')
        plt.scatter(x=centroid[0], y=centroid[1], c='b')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # print(*centroid)
        plt.savefig("poster/centroids.png", bbox_inches="tight", pad_inches=0)
        plt.close()

        box = swl.boxes[i]
        width = box[1][0] - box[0][0]
        height = box[1][1] - box[0][1]

        box[0][0] = centroid[0] - width // 2
        box[0][0] = max(0, box[0][0])
        box[1][0] = centroid[0] + width // 2
        box[1][0] = min(320, box[1][0])
        box[0][1] = centroid[1] - height // 2
        box[0][1] = max(0, box[0][1])
        box[1][1] = centroid[1] + height // 2
        box[1][1] = min(240, box[1][1])

        one, two = box
        x1, y1 = map(int, one)
        x2, y2 = map(int, two)
        # print(one, two)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=10, edgecolor='g', facecolor='none')
        plt.axis('off')
        fig = plt.imshow(cnn.frames[i], cmap=plt.get_cmap("binary"))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # plt.scatter(x=xs, y=ys, c='r')
        plt.gca().add_patch(rect)
        plt.savefig("poster/swl.png".format(i), bbox_inches="tight", pad_inches=0)

        boxes.append(box)
    return coors, boxes
