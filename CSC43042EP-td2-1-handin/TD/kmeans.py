import numpy as np

class Point:
    """A point in a dataset.

    Attributes:
        d: int             -- dimension of the ambient space
        coords: np.ndarray -- coordinates of the point
        label: int = 0     -- label of the cluster the point is assigned to
    """
    def __init__(self, d: int):
        assert d > 0, "The dimension needs to be positive."
        self.d = d
        self.coords = np.zeros(d, dtype=float)
        self.label = 0
        
    def update_coords(self, new_coords: np.ndarray) -> None:
        """Copy the values of new_coords to coords."""
        self.coords = new_coords.copy()
        
    def squared_dist(self, other) -> float:
        """The square of the Euclidean distance to other."""
        assert self.d == other.d
        return np.sum((self.coords - other.coords)**2)
    
class Cloud:
    """A cloud of points for the k-means algorithm.

    Data attributes:
    - d: int              -- dimension of the ambient space
    - points: list[Point] -- list of points
    - k: int              -- number of centers
    - centers: np.ndarray -- array of centers
    """

    def __init__(self, d: int, k: int):
        self.d = d
        self.k = k
        self.points = []
        self.centers = np.array([Point(d) for _ in range(self.k)])

    def add_point(self, p: Point, label: int) -> None:
        """Copy p to the cloud, in cluster label."""
        new_point = Point(self.d)
        new_point.update_coords(p.coords)
        self.points.append(new_point)
        self.points[-1].label = label
        
    def intracluster_variance(self) -> float:
        result = 0.0
        for point in self.points:
            result += point.squared_dist(self.centers[point.label])
        return result / len(self.points)
        
    def set_voronoi_labels(self) -> int:
        cnt = 0
        for point in self.points:
            original_label = point.label
            min_dist = np.inf
            for idx, center in enumerate(self.centers):
                dist = point.squared_dist(center)
                if  dist < min_dist:
                    min_dist = dist
                    point.label = idx
            if point.label != original_label:
                cnt += 1
        return cnt
    
    def set_centroid_centers(self) -> None:
        means = np.zeros((self.k, self.d), float)
        cnts = np.zeros(self.k, int)
        for point in self.points:
            means[point.label] += point.coords
            cnts[point.label] += 1
        means[cnts != 0] /= cnts[cnts != 0, np.newaxis]
        for idx, center in enumerate(self.centers):
            if cnts[idx] != 0:
                center.update_coords(means[idx])
            
    def init_random_partition(self) -> None:
        for point in self.points:
            point.label = np.random.randint(0, self.k)
        self.set_centroid_centers()
    
    def lloyd(self) -> None:
        """Lloyd’s algorithm.
        Assumes the clusters have already been initialized somehow.
        """
        while True:
            changed = self.set_voronoi_labels()
            self.set_centroid_centers()
            if changed == 0:
                break
            
    def init_forgy(self) -> None:
        """Forgy's initialization: distinct centers are sampled
        uniformly at random from the points of the cloud.
        """
        chosen_indices = np.random.choice(len(self.points), self.k, replace=False)
        for idx, chosen_index in enumerate(chosen_indices):
            self.centers[idx].update_coords(self.points[chosen_index].coords)

    def init_plusplus(self) -> None:
        """K-means++ initialization.
        """
        rand_idx = np.random.randint(len(self.points))
        self.centers[0].update_coords(self.points[rand_idx].coords)
        for i in range(1, self.k):
            dists = np.array([min(point.squared_dist(center) \
                                for center in self.centers[:i]) \
                                for point in self.points])
            probs = dists / dists.sum()
            chosen_index = np.random.choice(len(self.points), p=probs)
            self.centers[i].update_coords(self.points[chosen_index].coords)
        