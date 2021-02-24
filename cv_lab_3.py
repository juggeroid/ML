import numpy as np
import cv2
import argparse
import sklearn
import skimage.measure

def get_random_convolution_kernels(size = 3, batch_size = 5, min_val = -1, max_val = 1):
    assert(min_val < max_val)
    return [np.random.randint(min_val, max_val, (size, size)) for _ in range(0, batch_size)]

def conv_2d(image, kernel):
    xd = image.shape[0]
    yd = image.shape[1]
    kernel = np.fliplr(kernel)
    kernel = np.flipud(kernel)
    processed = np.zeros_like(image)
    # Append an additional vector to the top, bottom and to the left, right sides.
    padded = np.zeros((xd + 1 + 1, yd + 1 + 1))
    padded[1 : -1, 1 : -1] = image
    for y in np.arange(0, yd):
        for x in np.arange(0, xd):
            block = padded[x: x + 3, y: y + 3]
            processed[x, y] = (kernel * block).sum()
    return processed

def normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    processed = (image - mean) / std
    return processed

def relu(image):
    processed = (np.abs(image) + image) / 2
    return processed

def max_pooling(image):
    processed = skimage.measure.block_reduce(image, (2, 2), np.max)
    return processed

def softmax(image):
    e_x = np.exp(image - np.max(image))
    processed = e_x / e_x.sum(axis = 0)
    return processed

def main(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    """
    Produces a mess, just use a default smoothing kernel.
    random_kernels = get_random_convolution_kernels(size = 3,
                                                    batch_size = 5,
                                                    min_val = -1,
                                                    max_val = 1)
    """
    cv2.imshow("Gray",   image)
    cv2.imwrite("1.jpg", image)
    cv2.waitKey(0)

    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    image = conv_2d(image, kernel)
    cv2.imshow("Conv2D", image)
    cv2.imwrite("conv2d.jpg", image)
    cv2.waitKey(0)

    image = normalize(image)
    cv2.imshow("Normalized", image)
    cv2.imwrite("normalized.jpg", image)
    cv2.waitKey(0)

    image = relu(image)
    cv2.imshow("ReLU-ed", image)
    cv2.imwrite("relu-ed.jpg", image)
    cv2.waitKey(0)

    image = max_pooling(image)
    cv2.imshow("2x2-pooled", image)
    cv2.imwrite("2x2pooled.jpg", image)
    cv2.waitKey(0)

    image = softmax(image)
    cv2.imshow("Per-pixel softmax-ed", image)
    cv2.imwrite("softmaxed.jpg", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description = 'Convolution in numpy.')
    args.add_argument('path', help = 'Path to the image.')
    args = args.parse_args()
    main(args.path)
