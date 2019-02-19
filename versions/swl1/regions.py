import numpy as np
import cv2
import matplotlib.pyplot as plt

class Filters:
    """
    Splits image to generate grayscale filters, which are applied.
    Contours determined, OpenCV moments algorithm used to find centroids.
    """

    def __init__(self, images): #, original):
        self.centroids = []
        images = images.reshape(images.shape[0], 240, 320)
        for image in images:
            length, width = image.shape
            img = Filters.depth_to_gray(image).reshape(length * width)
            filters = [[0, 255/2], [255/2, 255]]

            self.centroids.append([])
            # print("Calculating all centroids")
            for i, filter in enumerate(filters):
                start, end = filter
                thresh = cv2.inRange(image, start, end)
                # cv2.imwrite("./thresh/thresh_" + str(i) + ".png", thresh)
                img, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 25 * 25:
                        self.centroids[-1].append(self.calc_centroid(contour))
            # print(*self.centroids)
            # print("Done calculating centroids")

    @staticmethod
    def depth_to_gray(image):
        """
        Converts crazy depth image format to grayscale.
        :param image: a depth image numpy array
        :return: a grayscale numpy array
        """
        high, low = np.amax(image), np.amin(image)
        val_range = high - low
        for y in range(len(image)):
            for x in range(len(image[y])):
                image[y][x] = float(255.0 * (float(image[y][x]) / val_range))
        return image

    def calc_centroid(self, contour):
        """
        Calculates centroid of given contour
        :param contour: OpenCV contour object
        :return: [centroid_x, centroid_y]
        """
        M = cv2.moments(contour)
        cx = int(M['m10'] / (M['m00'] + 1e-5))
        cy = int(M['m01'] / (M['m00'] + 1e-5))
        return [cx, cy]


if __name__ == "__main__":
    from versions.swl1 import main

    main.main()
