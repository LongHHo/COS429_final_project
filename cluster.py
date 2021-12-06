import numpy as np
import cv2
from sift_extract import extract_sift



def cluster_means(image_data):

    # input
    # image_data: shape (clusters, num_images, feature_length)

    # output: 
    # cluster_averages: shape (clusters, feature_length)

    num_clusters = image_data.shape[0]
    feature_length = image_data.shape[-1]

    cluster_averages = np.zeros((num_clusters, feature_length))

    for i in range(len(image_data)):
        cluster_averages[i] = np.mean(image_data[i], axis = 0)
    

    return cluster_averages


def assign_image(img, cluster_averages, step_size=2):

    img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
    descriptors = extract_sift(img, step_size)

    # reshape feature length to be 1-d
    descriptors = descriptors.flatten()

    num_clusters = cluster_averages.shape[0]

    dists = np.zeros(num_clusters)

    for i in range(num_clusters):
        dists[i] = np.linalg.norm(cluster_averages[i] - descriptors)


    return np.argmin(dists)

def assign_featurevect(vect, cluster_averages):
#     input:
#         image feature vector flattened to 1d array
#         vect: shape (num_sift_descriptors*128)
#         cluster_averages: (num_clusters, num_sift_descriptors*128)
            
#     output: int
#         assigned class index based on nearest neighbor
    
    
    # reshape feature length to be 1-d
    descriptors = vect.flatten()
    
    num_clusters = cluster_averages.shape[0]

    dists = np.zeros(num_clusters)

    for i in range(num_clusters):
        dists[i] = np.linalg.norm(cluster_averages[i] - descriptors)


    return np.argmin(dists)
    


