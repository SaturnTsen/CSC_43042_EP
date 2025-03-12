"""
linear_scan module
"""
import numpy as np
from TD.nearest_neighbor import NearestNeighborSearch


class LinearScan(NearestNeighborSearch):
    def query(self, x):
        # Ensures x is of correct shape
        super().query(x)
        # Store the index of nearest neighbor
        nearest_neighbor_index = -1
        current_min_dist = np.inf
        # Ex2: Loop through points
        for (idx, point) in enumerate(self.X):
            distance = self.metric(x, point)
            if current_min_dist > distance:
                current_min_dist = distance
                nearest_neighbor_index = idx 
        return current_min_dist, nearest_neighbor_index
