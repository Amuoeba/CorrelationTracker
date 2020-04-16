# Imports from external libraries
import ex2_utils
from ex2_utils import Tracker
import cv2
import numpy as np


# Imports from internal libraries

class CorrelationTracker(Tracker):

    def name(self):
        return 'correlation'

    def initialize(self, image, region):
        self.window = max(region[2], region[3]) * 2

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])

    def track(self, image):
        patch = ex2_utils.get_patch(image,self.position,self.size)
        cv2.imshow("Debug",image)
        cv2.imshow("Patch", patch[0])
        cv2.waitKey(0)
        print(f"Position:{self.position}, Size:{self.size}")
        return [0, 10, 2, 5]


class MSParams:
    def __init__(self):
        self.forgetting = 0.1


if __name__ == "__main__":
    print("Testing correlation tracker")



