import os
from typing import List
import numpy as np
from scipy.spatial.distance import cdist
import tensorflow as tf
from tensorboard.plugins import projector

def kNN(support_feats: np.ndarray, support_labels: List[int], example_feat: np.ndarray, k: int = 1, metric: str = 'euclidean') -> int:
    distances = cdist(support_feats, [example_feat], metric=metric).reshape(-1)
    sorted_labels = np.array(support_labels)[distances.argsort()]
    return np.bincount(sorted_labels[:k]).argmax()

def convert_embeddings_to_tf(embeddings: np.ndarray, labels: np.ndarray, log_dir: str = 'logs/'):
    # create log dir if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save labels in metadata file
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for label in labels:
            f.write("{}\n".format(label))

    config = projector.ProjectorConfig()

    embedding_var = tf.Variable(embeddings, trainable=False, name='metadata')

    embed = config.embeddings.add()
    embed.tensor_name = embedding_var.name
    embed.metadata_path = 'metadata.tsv'

    # define the model without training
    sess = tf.compat.v1.InteractiveSession()
    
    tf.compat.v1.global_variables_initializer().run()
    saver = tf.compat.v1.train.Saver()
    
    saver.save(sess, os.path.join(log_dir, 'embeddings.ckpt'))

    writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
    projector.visualize_embeddings(writer, config)
    sess.close()
