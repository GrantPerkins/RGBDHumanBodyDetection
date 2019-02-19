from versions.swl2.regions import Filters
from versions.swl2.slidingwindow import SlidingWindowLocalization
import math
import numpy as np
import matplotlib.pyplot as plt


def main(data):
    # print("Calculating filters")
    filt = Filters(data)  # , cnn.frame.reshape(240, 320).astype(np.float32))
    centroids = filt.centroids
    # print("Starting sliding window")
    swl = SlidingWindowLocalization(data)  # (cnn.frame.reshape(240, 320))
    accuracies = []
    coors = []
    boxes = []
    for i in range(len(data)):
        px = swl.px[i]
        py = swl.py[i]

        # print("px", px, py)
        centroid = min(centroids[i], key=lambda c1: math.sqrt((c1[0] - px) ** 2 + (c1[1] - py) ** 2))
        coors.append(centroid)
        # xs, ys = zip(*filt.centroids[i])
        # img = np.array(data.uncropped)[i].reshape(240, 320)
        # plt.imshow(img, cmap=plt.get_cmap("binary"))
        # plt.scatter(x=xs, y=ys, c='r')
        # plt.scatter(x=centroid[0], y=centroid[1], c='b')
        # # print(*centroid)
        # plt.savefig("centroids_{}.png".format(i))
        # plt.close()
        # skeleton = sk_data.skeletons[i]
        # real = skeleton[1]
        #
        # xs, ys = zip(*skeleton)
        # plt.imshow(img, cmap=plt.get_cmap("binary"))
        # plt.scatter(x=xs, y=ys, c='g')
        # plt.savefig("skeleton_{}.png".format(i))
        # plt.close()

        box = swl.boxes[i]
        # print(box)
        width = box[1][0] - box[0][0]
        height = box[1][1] - box[0][1]
        # print(width, height)

        box[0][0] = centroid[0] - width // 2
        box[0][0] = max(0, box[0][0])
        box[1][0] = centroid[0] + width //2
        box[1][0] = min(320, box[1][0])
        box[0][1] = centroid[1] - height // 2
        box[0][1] = max(0, box[0][1])
        box[1][1] = centroid[1] + height // 2
        box[1][1] = min(240, box[1][1])

        # one, two = box
        # a, b = map(int, one)
        # c, d = map(int, two)
        # # print(centroid)
        # # print(a,b,c,d)
        # # plt.imshow(img[b:d, a:c], cmap=plt.get_cmap("binary"))
        # # plt.savefig("calc_box_{}.png".format(i))
        # one, two = sk_data.boxes[i]
        # a, b = map(int, one)
        # c, d = map(int, two)
        # plt.imshow(img[b:d, a:c], cmap=plt.get_cmap("binary"))
        # plt.savefig("real_box_{}.png".format(i))

        boxes.append(box)
    return coors, boxes

"""
def accuracy(calc, actual):
    # print("BOXES:", calc, actual)
    area_calc = (calc[1][0] - calc[0][0]) * (calc[1][1] - calc[0][1])
    area_actual = (actual[1][0] - actual[0][0]) * (actual[1][1] - actual[0][1])
    i_width, i_height = 0, 0
    xs = sorted([calc[0][0], calc[1][0], actual[0][0], actual[1][0]])
    i_width = xs[2] - xs[1]
    ys = sorted([calc[0][1], calc[1][1], actual[0][1], actual[1][1]])
    i_height = ys[2] - ys[1]

    i_area = i_width * i_height
    # print("Areas:", area_calc, area_actual, i_area)
    return i_area / (area_actual + area_calc - i_area)


if __name__ == "__main__":
    main()
"""