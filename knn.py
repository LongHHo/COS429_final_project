from typing import List
import numpy as np
from scipy.spatial.distance import cdist

def kNN(support_feats: np.ndarray, support_labels: List[int], example_feat: np.ndarray, k: int = 1, metric: str = 'euclidean') -> int:
    distances = cdist(support_feats, [example_feat], metric=metric).reshape(-1)
    sorted_labels = np.array(support_labels)[distances.argsort()]
    return np.bincount(sorted_labels[:k]).argmax()