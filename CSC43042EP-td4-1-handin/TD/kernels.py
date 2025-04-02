import numpy as np
from typing import Self
from abc import abstractmethod
from scipy.special import gamma
from sklearn.neighbors import NearestNeighbors
from typing import *

class Kernel:
    """A class for kernel density estimation, which also stores the cloud of points
    Attributes:
        d: int                 -- dimension of the ambient space
        data: list[np.ndarray] -- list of coordinates of the points (each of dimension self.d)
    """
    def __init__(self: Self, d: int, data: List[np.ndarray]):
        self.data = data
        self.d = d

    @abstractmethod
    def density(self: Self, x: np.ndarray) -> float:
        raise NotImplementedError()
    
class Radial(Kernel):
    def __init__(self: Self, d: int , data: List[np.ndarray], bandwidth: float):
        # vectorize data to boost computation
        self.data = np.array(data)
        self.d = d
        self.bandwidth = bandwidth
    
    @abstractmethod
    def volume(self: Self) -> float:
        """Returns the volume of the unit ball associated with the kernel.
        """
        raise NotImplementedError()
    
    @abstractmethod 
    def profile(self: Self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Returns the kernel profile function k evaluated at the given real number t
        """
        raise NotImplementedError()

    def density(self: Self, x: np.ndarray) -> float:
        n = len(self.data)
        c_k_d = 1/self.volume()
        d = self.d
        sigma = self.bandwidth
        return c_k_d / (n * sigma**d) * np.sum(self.profile(np.sum((x-self.data)**2, axis=1)/sigma**2))

class Flat(Radial):
    def volume(self: Self) -> float:
        return np.pi**(self.d/2) / gamma(self.d/2 + 1)
        
    def profile(self: Self, t: Union[float, np.ndarray]) -> float:
        return np.array(t <= 1).astype(float)



class Gaussian(Radial):
    def __init__(self: Self, d: int, data: List[np.ndarray], bandwidth: float):
        # Vectorize data
        self.data = np.vstack(data)
        self.d = d
        self.bandwidth = bandwidth
    
    def volume(self: Self) -> float:
        return (2 * np.pi) ** (self.d / 2)

    def profile(self, t: Union[float, np.ndarray]) -> float:
        return np.exp(-t / 2)
    
    def guess_bandwidth(self: Self) -> None:
        n = len(self.data)
        m = np.mean(self.data, axis=0, keepdims=True)
        sigma_hat = np.sqrt(1 / (n - 1) * np.sum((self.data - m) ** 2))
        self.bandwidth = (n * (self.d + 2) / 4) ** (-1 / (self.d + 4)) * sigma_hat
    
    def guess_bandwidth_challenge(self: Self) -> None:
        # """Silverman's rule."""
        n = len(self.data)
        self.bandwidth = (n * (self.d + 2) / 4) ** (-1 / (self.d + 4))
    
class Knn(Kernel):
    """A class for kernel density estimation with k-Nearest Neighbors
       derived from Kernel
    Attributes not already in Kernel:
        k: int      -- parameter for k-NN
        V: float    -- "volume" constant appearing in density
        neigh:    sklearn.neighbors.NearestNeighbors   -- data structure and methods for efficient k-NN computations
    """
    def __init__(self: Self, d: int, data: list[np.ndarray], k: int, V: float):
        super().__init__(d,data)
        self.k, self.V = k, V
        self.neigh = NearestNeighbors(n_neighbors=self.k)
        self.fit_knn()

    def fit_knn(self: Self):
        """Computes the inner data structure acccording to the data points."""
        self.neigh.fit(np.array(self.data))

    def knn(self: Self, x: np.ndarray, vk: int) -> List[np.ndarray]:
        """The vk nearest-neighbors (vk can be different from self.k)."""
        return [np.array(self.data[i]) for i in self.neigh.kneighbors([x], n_neighbors=vk)[1][0] ]

    def k_dist_knn(self: Self, x: np.ndarray, vk: int) -> float:
        """The distance to vk-th nearest-neighbor."""
        return self.neigh.kneighbors([x], n_neighbors=vk)[0][0][vk-1]
    
    def density(self: Self, x: np.ndarray) -> float:
        n = len(self.data)
        return self.k / (2 * n * self.V * self.k_dist_knn(x, self.k))

    def meanshift(self: Self, k: int) -> None:
        updated_data = []
        for p in self.data:
            neighbors: List[np.ndarray] = self.knn(p, k)
            updated_data.append(sum(neighbors) / k)
        self.data = updated_data
        self.fit_knn()