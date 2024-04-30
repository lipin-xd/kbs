import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

colors = [np.array([139, 69, 19], dtype=np.float32) / 255,  # brown
          np.array([0, 255, 0], dtype=np.float32) / 255,  # green
          np.array([255, 192, 203], dtype=np.float32) / 255,  # pink
          np.array([255, 165, 0], dtype=np.float32) / 255,  # orange
          np.array([128, 0, 128], dtype=np.float32) / 255,  # purple
          np.array([255, 0, 0], dtype=np.float32) / 255,  # red
          np.array([255, 255, 255], dtype=np.float32) / 255,  # white
          np.array([255, 255, 0], dtype=np.float32) / 255]  # yellow


def distance_based_segmentation(color_space, threshold_0: float, threshold_1: float, threshold_2: float):
    image_path = "./beans"
    beans = [
        cv.cvtColor(cv.imread(image_path + "/brown.png"), color_space).astype(np.float32) / 255,
        cv.cvtColor(cv.imread(image_path + "/green.png"), color_space).astype(np.float32) / 255,
        cv.cvtColor(cv.imread(image_path + "/pink.png"), color_space).astype(np.float32) / 255,
        cv.cvtColor(cv.imread(image_path + "/orange.png"), color_space).astype(np.float32) / 255,
        cv.cvtColor(cv.imread(image_path + "/purple.png"), color_space).astype(np.float32) / 255,
        cv.cvtColor(cv.imread(image_path + "/red.png"), color_space).astype(np.float32) / 255,
        cv.cvtColor(cv.imread(image_path + "/white.png"), color_space).astype(np.float32) / 255,
        cv.cvtColor(cv.imread(image_path + "/yellow.png"), color_space).astype(np.float32) / 255
    ]

    mean_stds = np.zeros((8, 3, 2), dtype=np.float32)

    for i, _ in enumerate(beans):
        mean_stds[i, 0, 0] = beans[i][:, :, 0].mean()  # mean_beans[index, channel, mean/std]
        mean_stds[i, 0, 1] = beans[i][:, :, 0].std()
        mean_stds[i, 1, 0] = beans[i][:, :, 1].mean()
        mean_stds[i, 1, 1] = beans[i][:, :, 1].std()
        mean_stds[i, 2, 0] = beans[i][:, :, 2].mean()
        mean_stds[i, 2, 1] = beans[i][:, :, 2].std()
    print(mean_stds)

    img_bgr = cv.imread("./jbeans.png")
    img = cv.cvtColor(img_bgr, color_space).astype(dtype=np.float32) / 255

    row = img.shape[0]
    col = img.shape[1]

    arr = np.zeros(8, np.int64)
    count = 0

    for r in range(row):
        for c in range(col):
            flag = False
            for i, _ in enumerate(mean_stds):
                if abs(img[r, c, 0] - mean_stds[i, 0, 0]) < threshold_0 * mean_stds[i, 0, 1] and \
                        abs(img[r, c, 1] - mean_stds[i, 1, 0]) < threshold_1 * mean_stds[i, 1, 1] and \
                        abs(img[r, c, 2] - mean_stds[i, 2, 0]) < threshold_2 * mean_stds[i, 2, 1]:
                    img[r, c] = colors[i]
                    arr[i] += 1
                    flag = True
                    break
            if not flag:
                img[r, c] = 0
                count += 1

    # print(arr)
    # print(sum(arr))
    # print(img.size / 3)
    # print(count)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title("origin img")
    axes[0].imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
    if color_space == cv.COLOR_BGR2RGB:
        axes[1].set_title("segmented in RGB color space")
    elif color_space == cv.COLOR_BGR2HSV:
        axes[1].set_title("segmented in HSV color space")
    else:
        axes[1].set_title("segmented in Lab color space")
    axes[1].imshow(img)
    plt.show()


def KMeans_based_segmentation():
    img_bgr = cv.imread("./jbeans.png")
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title("origin img")
    axes[0].imshow(img_rgb)
    kmeans = KMeans(n_clusters=8, n_init=10)
    kmeans.fit(img_rgb.reshape(-1, 3))

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    seg_rgb_vals = centers[labels]

    seg_img_rgb = seg_rgb_vals.reshape(img_rgb.shape)
    axes[1].set_title("segmented RGB img")
    axes[1].imshow(seg_img_rgb)
    plt.show()


if __name__ == "__main__":
    threshold_0 = threshold_1 = threshold_2 = 2.5 # for RGB_segmentation
    distance_based_segmentation(cv.COLOR_BGR2RGB, threshold_0, threshold_1, threshold_2)
    threshold_0, threshold_1, threshold_2 = 8, 3, 3  # rfor HSV_segmentation
    distance_based_segmentation(cv.COLOR_BGR2HSV, threshold_0, threshold_1, threshold_2)
    threshold_0, threshold_1, threshold_2 = 1, 5, 3  # for HSV_segmentation
    distance_based_segmentation(cv.COLOR_BGR2Lab, threshold_0, threshold_1, threshold_2)
    KMeans_based_segmentation()
