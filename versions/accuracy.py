from versions.swl1 import main as swl1_main
from versions.swl2 import main as swl2_main
from versions.sae1 import main as sae1_main
from versions.sae2 import main as sae2_main
from versions.sae3 import main as sae3_main
from versions.current import main as current_main
from versions.depthview import Dataset
from versions.skeletonview import SkeletonDataset
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def bounding_accuracy(calc_boxes, actual_boxes):
    def box_accuracy(calc, actual):
        area_calc = (calc[1][0] - calc[0][0]) * (calc[1][1] - calc[0][1])
        area_actual = (actual[1][0] - actual[0][0]) * (actual[1][1] - actual[0][1])
        xs = sorted([calc[0][0], calc[1][0], actual[0][0], actual[1][0]])
        i_width = xs[2] - xs[1]
        ys = sorted([calc[0][1], calc[1][1], actual[0][1], actual[1][1]])
        i_height = ys[2] - ys[1]

        i_area = i_width * i_height
        return i_area / (area_actual + area_calc - i_area)

    accuracies = [box_accuracy(c, a) for c, a in zip(calc_boxes, actual_boxes)]
    return sum(accuracies) / len(accuracies)


def point_accuracy(centroids, skeletons):
    def accuracy(centroid, skeleton):
        real = skeleton[1]
        return 1 - ((centroid[0] - real[0]) ** 2 + (centroid[1] - real[1]) ** 2) ** .5 / 400

    accuracies = [accuracy(c, s) for c, s in zip(centroids, skeletons)]
    return sum(accuracies) / len(accuracies)


def main():
    a = 6
    s = 1 # change for person number
    e = 1
    # data = np.array(Dataset(a, s, e, crop=False).uncropped)[61:62]
    # sk_data = SkeletonDataset(a, s, e)
    # current_main.main(data)
    for s in range(8, 10):
        print("\n\n\n---------", s, "---------")
        data = np.array(Dataset(a, s, e, crop=False).uncropped)
        sk_data = SkeletonDataset(a, s, e)
        print("\nStarting current")
        centroids, boxes = current_main.main(data)
        print("current", bounding_accuracy(boxes, sk_data.boxes))
        print("current point", point_accuracy(centroids, sk_data.skeletons))
        print("\nStarting SWL.1")
        centroids, boxes = swl1_main.main(data)
        print("SWL.1", bounding_accuracy(boxes, sk_data.boxes))
        print("SWL.1 point", point_accuracy(centroids, sk_data.skeletons))
        print("\nStarting SWL.2")
        centroids, boxes = swl2_main.main(data)
        print("SWL.2", bounding_accuracy(boxes, sk_data.boxes))
        print("SWL.2 point", point_accuracy(centroids, sk_data.skeletons))
        print("\nStarting SAE.1")
        centroids, boxes = sae1_main.main(data)
        print("SAE.1", bounding_accuracy(boxes, sk_data.boxes))
        print("SAE.1 point", point_accuracy(centroids, sk_data.skeletons))
        print("\nStarting SAE.2")
        centroids, boxes = sae2_main.main(data)
        print("SAE.2", bounding_accuracy(boxes, sk_data.boxes))
        print("SAE.2 point", point_accuracy(centroids, sk_data.skeletons))
        print("\nStarting SAE.3")
        centroids, boxes = sae3_main.main(data)
        print("SAE.3", bounding_accuracy(boxes, sk_data.boxes))
        print("SAE.3 point", point_accuracy(centroids, sk_data.skeletons))


if __name__ == "__main__":
    main()
