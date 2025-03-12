"""
nearest_neighbor module
"""

import abc
from abc import ABC
from typing import Callable, Tuple

import numpy as np


def euclidean_distance(x_1: np.ndarray, x_2: np.ndarray) -> float:
    """
    Euclidean distance between points

    :param numpy.ndarray x_1: Coordinates of first point (vector: shape (p,))
    :param numpy.ndarray x_2: Coordinates of second point (vector: shape (p,))
    :return: Euclidean distance between x_1 and x_2
    :rtype: float
    """
    # Ex1
    return np.sqrt(np.sum((x_1 - x_2)**2))


class NearestNeighborSearch(ABC):
    """Base class for NearestNeighborSearch methods"""

    def __init__(
        self,
        X: np.ndarray,
        metric: Callable[[np.ndarray, np.ndarray], float] = euclidean_distance,
    ):
        """Initialize the implementation, i.e. fill whatever data
        structure is needed"""
        assert X.shape[0], "Empty 1st dim"
        assert X.ndim == 2, "Only 2D arrays are supported"
        self.X = X
        self.metric = metric

    @abc.abstractmethod
    def query(self, x: np.ndarray) -> Tuple[float, int]:
        """Return distance to and index of nearest neighbor of x in original
        data X"""
        assert x.ndim == 1, "Query point x must be 1D"
