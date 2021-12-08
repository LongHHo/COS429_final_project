import numpy as np
import cv2


import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
from pathlib import Path
# from tensorflow.keras import applications
# from tensorflow.keras import layers
# from tensorflow.keras import losses
# from tensorflow.keras import optimizers
# from tensorflow.keras import metrics
# from tensorflow.keras import Model
# from tensorflow.keras.applications import resnet


from sklearn.utils import shuffle


def get_pairwise_batch(batch_size, train_data):
    """
    Create batch of n pairs, half same class, half different class
    """
    n_classes, n_examples, w, h, d = train_data.shape
    

    rng = np.random.default_rng()

    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes,size=(batch_size,),replace=False)
    
    # initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((batch_size, h, w, d)) for i in range(2)]
    
    # initialize vector for the targets
    targets=np.zeros((batch_size,))
    
    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size//2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i,:,:,:] = train_data[category, idx_1].reshape(w, h, d)
        idx_2 = rng.randint(0, n_examples)
        
        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category  
        else: 
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1,n_classes)) % n_classes
        
        pairs[1][i,:,:,:] = train_data[category_2,idx_2].reshape(w, h, d)
    
    return pairs, targets


# returns three lists: anchor, positive, negative
def get_triple_batch(batch_size, train_data):
    """
    Create three lists of anchor images, positive images, negative images
    """
    n_classes, n_examples, w, h, d = train_data.shape
    

    rng = np.random.default_rng()

    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes,size=(batch_size,),replace=False)
    
    # initialize 2 empty arrays for the input image batch
    anchor = np.zeros((batch_size, h, w, d))

    positive = np.zeros((batch_size, h, w, d))

    negative = np.zeros((batch_size, h, w, d))

    
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        anchor[i,:,:,:] = train_data[category, idx_1].reshape(w, h, d)

        
        idx_pos = rng.randint(0, n_examples)
        cat_pos = category
        positive[i,:,:,:] = train_data[cat_pos,idx_pos].reshape(w, h, d)

        idx_neg = rng.randint(0, n_examples)
        cat_neg = (category + rng.randint(1,n_classes)) % n_classes
        negative[i,:,:,:] = train_data[cat_neg,idx_neg].reshape(w, h, d)

    return anchor, positive, negative

    



def make_oneshot_task(N, test_data):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    """ N must be less than the number of classes in the test dataset: 20 """
    n_classes, n_examples, w, h, d = test_data.shape
    
    rng = np.random.default_rng()

    indices = rng.randint(0, n_examples,size=(N,))


    categories = rng.choice(n_classes,size=(N,),replace=False)            
    
    true_category = categories[0]
    ex1, ex2 = rng.choice(n_examples,replace=False,size=(2,))
    test_image = np.asarray([test_data[true_category,ex1,:,:]]*N).reshape(N, w, h,1)
    support_set = test_data[categories,indices,:,:]
    support_set[0,:,:] = test_data[true_category,ex2]
    support_set = support_set.reshape(N, w, h,1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image,support_set]
    return pairs, targets

  
def test_oneshot(model, N, k, verbose = 0, test_data=None):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N,test_data)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
    return percent_correct