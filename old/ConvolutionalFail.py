import matplotlib.pyplot as plt
import numpy as np
from versions.current.depthview import DepthVideo


class ConvolutionalNeuralNetwork:
    def __init__(self, sae, length, width):
        print("Opening video")
        vid = DepthVideo("C:\\Users\\gcper\\Code\\STEM\\data\\MSRDailyAct3D_pack1\\" + "a01_s01_e01_depth.bin", length,
                         width)
        print("Getting frame")
        vid.get_frame()
        frame = np.reshape(np.array(vid.frame), [1, 240, 320, 1])

        image = sae.ses.run(sae.encode(frame))

        plt.imshow(image.reshape(240, 320), cmap=plt.get_cmap("binary"))

        plt.savefig("static.png")
        plt.close()


def main():
    from versions.current import main
    main.main()


if __name__ == "__main__":
    main()
