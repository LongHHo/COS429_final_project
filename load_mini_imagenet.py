import pickle
import numpy as np

RANDOM_SEED = 12345

def load_mini_imagenet_data(path: str):
    data_file = open(path, "rb")
    data = pickle.load(data_file)
    X = data["image_data"]
    y = np.empty((X.shape[0]), dtype=int)
    for i, (_, vals) in enumerate(data["class_dict"].items()):
        y[vals] = i
    return X, y

def split_for_n_shot(data: np.ndarray, labels: np.ndarray, n=5):
    rng = np.random.default_rng(RANDOM_SEED)
    num_labels = len(np.unique(labels))
    support_indices = np.empty((num_labels*n), dtype=int)
    for i, label in enumerate(np.unique(labels)):
        class_labels = np.argwhere(labels == label).reshape(-1)
        support_indices[i*n:i*n+n] = rng.choice(class_labels, size=n, replace=False)
    support_indices = np.sort(support_indices)
    mask = np.ones(labels.shape, dtype=bool)
    mask[support_indices] = False
    X_support = data[support_indices]
    y_support = labels[support_indices]
    X_test = data[mask]
    y_test = labels[mask]
    return X_support, y_support, X_test, y_test