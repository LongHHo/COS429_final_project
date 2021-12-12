
import numpy as np
import cv2
import matplotlib.pyplot as plt


############## SIFT ############################################################

def extract_sift(img, step_size=1):
    """
    Extract SIFT features for a given grayscale image.
    Feel free to use OpenCV functions.
    Args:
        img: Grayscale image of shape (H, W)
        step_size: Size of the step between keypoints
    """
    sift = cv2.SIFT_create()


    keypoints = [cv2.KeyPoint(j, i, step_size) for i in range(0, img.shape[0], step_size) for j in range(0, img.shape[1], step_size)]

    _, descriptors = sift.compute(img, keypoints)

    return descriptors


def extract_sift_for_dataset(data, step_size=1):
    num_examples = data.shape[0]
    all_features = np.zeros((num_examples, 128 * (data.shape[1] // step_size) * (data.shape[2] // step_size)))
    for i in range(num_examples):
        img = data[i]
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
        descriptors = extract_sift(img, step_size)
        all_features[i] = descriptors.reshape(-1)
    return all_features
