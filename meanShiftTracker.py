# Imports from external libraries
import numpy as np
import cv2
# Imports from internal libraries
from ex2_utils import Tracker
import ex2_utils
import ex1_utils
from sequence_utils import VOTSequence
import configs as cfg
from my_utills import convergence_map
import time

image = np.array([])
vis_image = np.array([])
click = {"x": None, "y": None}


def on_click_rectangle(event, x, y, flags, params):
    global image
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = x - 2
        y1 = y - 2
        x2 = x + 3
        y2 = y + 3
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        click["x"] = x
        click["y"] = y
        print(x, y)


class Kernels:
    @staticmethod
    def uniform(w, h):
        kernel = np.array([1 for x in range(w * h)])
        kernel = kernel.reshape((h, w))
        return kernel

    @staticmethod
    def epanechnikov(w, h, sigma):
        return ex2_utils.create_epanechnik_kernel(w, h, sigma)

    @staticmethod
    def gausean(w, h, sgima):
        return False


def genereate_x_coord_kernel(w, h):
    a = int(np.floor(w / 2))
    b = int(np.ceil(w / 2))
    row = list(range(-a, b))
    kernel = []
    for i in range(h):
        kernel.append(row)
    kernel = np.array(kernel)
    return kernel


def generate_y_coord_kernel(w, h):
    kernel = genereate_x_coord_kernel(h, w)
    return kernel.T


def test_meanshift():
    global image, vis_image
    image = ex2_utils.generate_responses_1()
    # image = ex2_utils.genereate_response_2()
    vis_image = image.copy()
    # cv2.imshow("test_map", vis_image * 255)
    # cv2.setMouseCallback("test_map", on_click_rectangle)
    # cv2.waitKey(0)
    # cv2.imshow("test_map", vis_image * 255)
    # cv2.waitKey(0)
    #
    # x_iter, y_iter = click["x"], click["y"]
    x_iter = 34
    y_iter = 61
    window_shape = (51, 51)
    finish = False
    i=0
    visited = set()
    while i< 100 and not finish:
        patch = ex2_utils.get_patch(image, (x_iter, y_iter), window_shape)[0]
        x_iter, y_iter,finish = mean_shift(patch, x_iter, y_iter, kernel="epanechnikov")
        if (x_iter,y_iter) in visited:
            break
        visited.add((x_iter,y_iter))
        i+=1

    print(f"Finished in {i} steps ")
    print(f"Converged at x: {x_iter}, y: {y_iter}")
    print(f"Start at x: {click['x']}, y: {click['y']}")


def test_meanshift_convergence(shape=(5, 5)):
    global image, vis_image
    # image = ex2_utils.generate_responses_1()
    image = ex2_utils.genereate_response_2()
    start_converge_map = {}
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            x_start = x
            y_start = y
            window_shape = shape
            finish = False
            i = 0
            while i < 30 and not finish:
                patch = ex2_utils.get_patch(image, (x_start, y_start), window_shape)[0]
                x_start, y_start, finish = mean_shift(patch, x_start, y_start, kernel="epanechnikov")
                i += 1

            if x_start >= image.shape[0]:
                x_start = image.shape[0]-1
            if y_start >= image.shape[1]:
                y_start = image.shape[1]-1

            if (x_start,y_start) not in start_converge_map:
                start_converge_map[(x_start,y_start)] = [(x,y)]
            else:
                start_converge_map[(x_start,y_start)].append((x,y))
    print(start_converge_map)
    return start_converge_map




def mean_shift(patch, x_start, y_start, kernel="uniform"):
    h = patch.shape[0]
    w = patch.shape[1]

    kernels = {"uniform": Kernels.uniform(w, h),
               "epanechnikov": Kernels.epanechnikov(w, h, cfg.EPANECH_SIGMA),
               "gausian": Kernels.gausean(5, 5, 2)}

    if kernel not in kernel:
        raise ValueError(f"Vrong kerenl value {kernel}")
    kernel = kernels[kernel]
    kernel = kernel[:h, :w]

    # print(kernel)

    x_coord = genereate_x_coord_kernel(w, h)
    y_coord = generate_y_coord_kernel(w, h)

    # neighbours = int((kernel.shape[0] - 1) / 2)
    # x_neighbours = int((w - 1) / 2)
    # y_neighbours = int((h - 1) / 2)
    # y1 = y_start - y_neighbours
    # y2 = y_start + y_neighbours + 1
    # x1 = x_start - x_neighbours
    # x2 = x_start + x_neighbours + 1
    # patch = image[y1:y2, x1:x2]

    num_x = np.multiply(np.multiply(x_coord, patch),
                        kernel)
    denom_x = np.multiply(patch, kernel)
    num_y = np.multiply(np.multiply(y_coord, patch),
                        kernel)
    denom_y = np.multiply(patch, kernel)

    x_next = (np.sum(num_x) / np.sum(denom_x))
    y_next = (np.sum(num_y) / np.sum(denom_y))

    if np.isnan(x_next):
        x_next = 0
    if np.isnan(y_next):
        y_next = 0

    # print(f"Change: {x_next}, {y_next}")

    should_finish = False
    if np.abs(x_next) < 0.01 and np.abs(y_next) < 0.01:
        should_finish = True


    # print(f"x:{x_next}, y:{y_next}")
    # if np.abs(x_next) > 0.05:
    if x_next < 0:
        x_next = int(np.floor(x_next))
    else:
        x_next = int(np.ceil(x_next))
    # else:
    #     x_next = 0

    # if np.abs(x_next) > 0.05:
    if y_next < 0:
        y_next = int(np.floor(y_next))
    else:
        y_next = int(np.ceil(y_next))
    # else:
    #     y_next = 0

    x_next = int(x_start + x_next)
    y_next = int(y_start + y_next)

    # print(f"Start position: {x_start}, {y_start}")
    # print(f"End position: {x_next}, {y_next}")

    vis_image = image.copy() * 255

    # xvis1 = x_next - x_neighbours
    # yvis1 = y_next - y_neighbours
    # xvis2 = x_next + x_neighbours + 1
    # yvis2 = y_next + y_neighbours + 1
    # cv2.rectangle(vis_image, (xvis1, yvis1), (xvis2, yvis2), (255, 0, 0), 1)
    # cv2.imshow("iterations", vis_image)
    # cv2.waitKey(100)
    # print(f"x:{x_next}, y:{y_next}")
    return x_next, y_next,should_finish


class MeanShiftTracker(Tracker):
    def __init__(self, params):
        super().__init__(params)
        self.window = None
        self.template = None
        self.position = None
        self.size = None
        self.current_model = None
        self.kernel = None

    def initialize(self, image, region):
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])
        self.template = ex2_utils.get_patch(image, self.position, self.size)[0]
        # print(self.size)
        # print(self.position)
        # print(self.window)
        # cv2.imshow("Init", self.template)
        # cv2.waitKey(0)

        ep_kernel = ex2_utils.create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.kernel_sigma)
        self.kernel = ep_kernel[:self.template.shape[0], :self.template.shape[1]]

        # print(f"template shape: {self.template.shape}")
        # print(f"Kernel shape: {self.kernel.shape}")
        # print(f"Size 0: {self.size[0]}")
        # print(f"Size 1: {self.size[1]}")
        h1 = ex2_utils.extract_histogram(self.template, 16, weights=self.kernel)
        self.current_model = h1 / np.sum(h1)

    def track(self, image):
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)
        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)
        i = 0
        # print(f"Position at start: {self.position}")
        should_finish = False
        while i < 20 and not should_finish:
            current_frame = ex2_utils.get_patch(image, self.position, self.size)[0]
            # print(f"TRACK template shape: {current_frame.shape}")
            # print(f"TRACK Kernel shape: {self.kernel.shape}")
            # print(f"TRACK Size : {self.size}")
            # print(f"TRACK position: {self.position}")

            h2 = ex2_utils.extract_histogram(current_frame, 16, weights=self.kernel)
            h2 = h2 / np.sum(h2)
            vu = np.sqrt(self.current_model / (h2 + self.parameters.epsilon))
            back_proj = ex2_utils.backproject_histogram(current_frame, vu, 16)

            # cv2.imshow("Backpproj",back_proj)
            # cv2.imshow("Kernel",self.kernel)
            # cv2.waitKey(0)

            next_x, next_y,should_finish = mean_shift(back_proj,
                                        int(self.position[0]),
                                        int(self.position[1]),
                                        kernel="uniform", )

            if self.position == (next_x, next_y):
                keep_tracking = False
            self.position = (next_x, next_y)
            # print(f"Position at iteration; {i} --> {self.position}")
            i += 1

        self.current_model = (1 - self.parameters.alpha) * self.current_model + self.parameters.alpha * h2
        # self.template = ex2_utils.get_patch(image, self.position, self.size)[0]

        # cv2.imshow("back proj", back_proj)
        # cv2.waitKey(0)

        bounding_box = [self.position[0] - self.size[0] / 2,
                        self.position[1] - self.size[1] / 2,
                        self.size[0],
                        self.size[1]]
        # bounding_box = [left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]]

        return bounding_box


class MSParams:
    def __init__(self):
        self.epsilon = 0.0001
        self.alpha = 0.05
        self.enlarge_factor = 1
        self.kernel_sigma = 0.3



if __name__ == "__main__":
    print("Running mean shift tracker")



    test_meanshift()

    # Generate convergence maps on default example
    # for s in list(range(3,25,2)):
    #     res = test_meanshift_convergence(shape=(s,s))
    #     convergence_map(res,f"_my_fun_{s}_{s}")



    # space = ex2_utils.generate_responses_1()
    # cv2.imshow("test_2",space*255)
    # # cv2.imwrite(f"{cfg.result_path}my_function.jpg",space*255)
    # cv2.waitKey(0)







    # image_path1 = "/home/erik/Documents/Projects/Faculty/advanced_methods_in_CV/project2/untitled/data/bolt1/00000001.jpg"
    # image_path2 = "/home/erik/Documents/Projects/Faculty/advanced_methods_in_CV/project2/untitled/data/bolt1/00000002.jpg"
    # test_image1 = cv2.imread(image_path1)
    # test_image2 = cv2.imread(image_path2)
    #
    # R = cv2.selectROI(test_image1)
    # r = []
    # for x in R:
    #     if x % 2 == 0:
    #         r.append(x + 1)
    #     else:
    #         r.append(x)
    #
    # print(f"R: {r}")
    # imCrop1 = test_image1[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    # imCrop2 = test_image2[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    # # imCrop1, b = ex2_utils.get_patch(test_image1, (100, 100), (51, 51))
    # # imCrop2, b = ex2_utils.get_patch(test_image2, (100, 100), (51, 51))
    # cv2.imshow("Image", imCrop1)
    # cv2.waitKey(0)
    #
    # epanechnikovKernel = ex2_utils.create_epanechnik_kernel(imCrop1.shape[1], imCrop1.shape[0], 2)
    #
    # print(f"Crop shape: {imCrop1.shape}")
    # print(f"epanechnikov shape {epanechnikovKernel.shape}")
    # h1 = ex2_utils.extract_histogram(imCrop1, 4, weights=epanechnikovKernel)
    # h1 = h1 / np.sum(h1)
    # h2 = ex2_utils.extract_histogram(imCrop2, 4, weights=epanechnikovKernel)
    # h2 = h2 / np.sum(h2)
    #
    # vu = np.sqrt(h1 / (h2 + 0.001))
    # back_proj = ex2_utils.backproject_histogram(imCrop2, vu, 4)
    #
    # print(f"histogram1:\n{h1}")
    # print(f"histogram2:\n{h2}")
    # print(f"Vu:\n{vu}")
    # print(f"Back proj:\n{back_proj}")
    #
    # cv2.imshow("test", test_image1)
    # cv2.imshow("patch", imCrop1)
    # cv2.imshow("back_proj", back_proj)
    # cv2.waitKey(0)
