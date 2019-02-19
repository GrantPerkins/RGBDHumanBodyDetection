import cv2
import numpy as np


class SlidingWindowLocalization:
    """
    Uses sliding window localization to localize the human's centroid using a Haar Cascade model
    """

    def __init__(self, images):
        height = 1
        width = 0.5
        self.px = []
        self.py = []
        i_count = 0
        self.classifier = cv2.CascadeClassifier("C:/Users/gcper/Code/STEM/haarcascade_fullbody.xml")

        self.rects = []
        self.boxes = []
        for j, image in enumerate(images):
            # print("Sliding on image ", j)
            image = image.reshape(240, 320)
            rects = []
            for x, y, size, window in self.slide(image, height, width):
                person = self.is_person(window)
                if person:
                    rects.append([size, [x, y]])
            points = sorted(rects, key=lambda i: i[1][0])[len(rects) // 3: ((len(rects) * 2) // 3) + 1]
            # for rect in points:
            #     one, two = rect
            #     a, b = map(int, one)
            #     c, d = map(int, two)
            #     plt.imshow(image[d-(b//2):d+(b//2), c-a//2:c+a//2], cmap=plt.get_cmap("binary"))
            #     plt.savefig("windows/{}.png".format(i_count))
            #     plt.close()
            #     i_count += 1

            l = len(points)
            if l != 0:
                if len(rects) > 0:
                    points = rects
            if len(points) != 0:
                self.px.append(sum([c[1][0] for c in points]) // len(points))
                self.py.append(sum([c[1][1] for c in points]) // len(points))
                left_x = sum([points[i][1][0] - (points[i][0][0] // 2) for i in range(l)]) // l
                left_y = sum([points[i][1][1] - (points[i][0][1] // 2) for i in range(l)]) // l
                size_w = sum([points[i][0][0] for i in range(l)]) // l
                size_h = sum([points[i][0][1] for i in range(l)]) // l
                self.boxes.append([[left_x, left_y], [left_x + size_w, left_y + size_h]])
                # print([[left_x, left_y], [left_x + size_w, left_y + size_h]])
                # plt.imshow(image[left_y:left_y+size_h, left_x:left_x+size_w], cmap=plt.get_cmap("binary"))
                # plt.savefig("box_{}.png".format(j))
                # plt.close()
            else:
                # print("0 humans detected")
                self.px.append(0)
                self.py.append(0)
                self.boxes.append([[0, 0], [0, 0]])

        # print("SLW centers found")

    def sliding_window(self, image, step_size, window_size):
        # slide a window across the image
        for y in range(0, image.shape[0] - window_size[1], step_size):
            for x in range(0, image.shape[1] - window_size[0], step_size):
                # yield the current window
                tmp_x, tmp_y = x + (window_size[0] // 2), y + (window_size[1] // 2)
                yield (tmp_x, tmp_y, window_size, image[y:y + window_size[1], x:x + window_size[0]])

    def slide(self, image, height, width):
        # image size 240 by 320
        for size in range(200, 240, 20)[::-1]:
            for window in self.sliding_window(image, 5, [int(width * size), int(height * size)]):
                # generator function yields windows of varying sizes, 240 down to 100 in steps of 20
                yield window

    def is_person(self, image):
        return bool(len(self.classifier.detectMultiScale(image.astype(np.uint8))))


if __name__ == "__main__":
    from versions.current import main

    main.main()
