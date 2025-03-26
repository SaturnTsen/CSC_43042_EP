# module cloud
"""Minimal implementation of data points and clouds."""

import numpy as np
from typing import List

class Point:
    """A point in a dataset.

    coords is initialized with a list of floats (or any other iterable
    that can be used to initialize a numpy ndarray).  The name string
    is optional.

    Attributes:
        coords: np.ndarray -- coordinates of the point
        name: str = ''
    """

    def __init__(self, coords : List[float], name=''):
        self.coords = np.array(coords)
        self.name = name

    def __str__(self):
        return f'{self.coords} ({self.name})'

    def __repr__(self):
        return f'Point({repr(self.coords)}, name={repr(self.name)})'

    def update_coords(self, new_coords: np.ndarray) -> None:
        """Copy the values of new_coords to coords."""
        self.coords = new_coords.copy()

    def dist(self, other) -> float:
        """Euclidean distance from the other Point."""
        return np.sqrt(np.sum((self.coords - other.coords)**2))


class Cloud:
    """
    A very minimalistic cloud of points.
    """
    def __init__(self):
        self._points = []

    def __str__(self) -> str:
        top = f'Cloud:\n'
        index_width = len(str(len(self._points) - 1))  # Width of max index
        rest = ''.join(f' {i:>{index_width}} {p}\n'
                       for (i, p) in enumerate(self._points))
        return top + rest

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self):
        return iter(self._points)

    def __getitem__(self, i : int):
        return self._points[i]

    def add_point(self, p : Point) -> None:
        self._points.append(p)


def load_cloud_from_file(infile):
    """Load a point cloud from a file object (already opened for reading."""
    c = Cloud()
    _ = infile.readline()  # Throw away header
    for line in infile:
        parts = line.split(',')
        coords = [float(x) for x in parts[:-1]]
        name = parts[-1].strip()
        c.add_point(Point(coords, name))
    return c

