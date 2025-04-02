import numpy as np


from typing import Tuple

def my_parametrization(d: int, nb_samples: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """Set empirically determined initial parameters for $k$-means."""
    nb_clusters = 4
    my_labels = np.random.randint(0, nb_clusters, size=nb_samples)
    my_center_coords = np.random.rand(nb_clusters, d)
    return (nb_clusters, my_labels, my_center_coords)
