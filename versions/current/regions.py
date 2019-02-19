import numpy as np
import cv2


class Histogram:
    """
    Uses a historgram to generate grayscale filters, which are applied.
    Contours determined, OpenCV moments algorithm used to find centroids.
    """

    def __init__(self, images): #, original):
        self.centroids = []
        for image in images:
            length, width = image.shape
            img = Histogram.depth_to_gray(image).reshape(length * width)
            hist, bins = np.histogram(img, bins=10)
            pairs = list(zip(hist, bins))
            # plt.title("Histogram of Depth Values")
            # plt.xlabel("Depth value")
            # plt.ylabel("Frequency")
            # plt.hist(img, bins=10)
            # plt.savefig("hist.png")
            # plt.close()

            # plt.imshow(image.reshape(240,320), cmap=plt.get_cmap("binary"))
            # plt.savefig("testtest.png")
            # plt.close()

            step = pairs[1][1] / 2
            filters = []
            # print("Creating filters")
            for i, pair in enumerate(pairs):
                if i == 0:
                    filters.append([0, step])
                elif i == len(pairs) - 1:
                    filters.append([filters[i - 1][1], step])
                else:
                    filters.append([filters[i - 1][1], filters[i - 1][1] + 2 * step])

            self.centroids.append([])
            # print("Calculating all centroids")
            for i, filter in enumerate(filters):
                start, end = filter
                thresh = cv2.inRange(image, start, end)
                cv2.imwrite("poster/thresh_" + str(i) + ".png", thresh)
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
    from versions.current import main

    main.main()
