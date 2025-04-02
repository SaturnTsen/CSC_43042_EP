"""
kdtree module
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from TD.nearest_neighbor import NearestNeighborSearch


def median(X: np.ndarray, start: int, stop: int, c: int) -> float:
    """
    Returns median of array X between indices start and stop for coordinate c
    """
    assert stop - start >= 0, "Requested stop > start"
    assert X.ndim == 2, "2D required"
    arr = X[start:stop, c]
    sorted_indices = arr.argsort()
    med = arr[sorted_indices[(stop - start)//2]]
    return med


def swap(X: np.ndarray, idx1, idx2) -> None:
    """Swaps two rows of a 2D numpy array"""
    # This is erroneous
    # X[idx1, :], X[idx2, :] = X[idx2, :], X[idx1, :]
    # This is the correct implementation
    X[[idx1,idx2]]=X[[idx2,idx1]]

def partition(X: np.ndarray, start: int, stop: int, c: int) -> int:
    """
    Partitions the array X between start and stop wrt to its median along a coordinate c
    """
    # Ex4
    # You may or may not use the median function above, up to you
    # med = median(X, start, stop, c)
    idx = start + (stop - start)//2
    indices = X[start:stop, c].argsort()
    # NOTE 第一遍写的时候，直接对X进行了indices索引，导致缺少了start偏移量
    X[start:stop] = X[start:stop, :][indices]
    return idx


@dataclass
class Node:
    idx: int
    med: float = np.inf
    c: int = 0
    left: Self = (
        None  # Self denotes that left (and right) is of same type as self == Node
    )
    right: Self = None


class KDTree(NearestNeighborSearch):
    def __init__(self, X):
        """
        Contrary to LinearScan, KDTree's constructor must build the tree
        To that end, we will loop through the coordinates of X,
        hence the need for the `dim` attribute below.
        """
        super().__init__(X)
        self.dim = X.shape[1]
        self.build()

    def _build(self, start: int, stop: int, c: int) -> Node | None:
        """
        Builds a node with a correct index by partitioning X along c between start and stop,
        including left and right children nodes
        """
        assert stop >= start, "Indices issue"
        if stop == start:
            return
        if stop == start + 1:
            return Node(start)
        next_c = (c + 1) % self.dim
        idx = partition(self.X, start, stop, c)
        node = Node(idx)
        node.left = self._build(start, idx, next_c)
        node.right = self._build(idx + 1, stop, next_c)
        return node

    def reset(self):
        """
        Resets current estimation of distance to and index of nearest neighbor
        """
        self._current_dist = np.inf
        self._current_idx = -1

    def build(self):
        """
        Builds the kdtree
        """
        self.reset()
        self.root = self._build(0, len(self.X), 0)

    def _defeatist(self, node: Node | None, x: np.ndarray, c = 0):
        """
        Defeatist search of nearest neighbor of x in node
        """
        if node is None:
            return
        dist = self.metric(self.X[node.idx], x)
        if self._current_dist > dist:
            self._current_dist = dist
            self._current_idx = node.idx
        if x[c] <= self.X[node.idx, c]:
            self._defeatist(node.left, x, (c+1) % self.dim)
        else:
            self._defeatist(node.right, x, (c+1) % self.dim)

    def _backtracking(self, node: Node | None, x: np.ndarray, c = 0):
        """
        Backtracking search of nearest neighbor of x in node
        """
        if node is None:
            return
        # Ex7
        dist = self.metric(self.X[node.idx], x)
        if self._current_dist > dist:
            self._current_dist = dist
            self._current_idx = node.idx
        if x[c] - self._current_dist <= self.X[node.idx, c]:
            self._backtracking(node.left, x, (c+1) % self.dim)
        if x[c] + self._current_dist >= self.X[node.idx, c]:
            self._backtracking(node.right, x, (c+1) % self.dim)

    def query(self, x, mode: str = "backtracking"):
        """
        Queries given mode 'backtracking' or 'defeatist'
        """
        super().query(x)
        self.reset()
        if mode == "defeatist":
            self._defeatist(self.root, x)
        elif mode == "backtracking":
            self._backtracking(self.root, x)
        else:
            raise ValueError("Incorrect mode!")
        return self._current_dist, self._current_idx

    def set_xaggle_config(self):
        self.mode = 'backtracking'  # Choose search strategy for xaggle
