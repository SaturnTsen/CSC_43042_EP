import numpy as np

from kmeans import *
from kmeans_application import *


def main():
    # Read data
    data = np.genfromtxt("../csv/samples_excerpt.csv", delimiter=",")
    (nmax, d) = data.shape

    # Your data
    (nb_clusters, my_labels, my_center_coords) = my_parametrization(d, nmax)

    # Create cloud
    cloud = Cloud(d, nb_clusters)
    for i in range(nmax):
        p = Point(d)
        p.update_coords(data[i])
        cloud.add_point(p, my_labels[i])

    # Perform clustering
    for i in range(nb_clusters):
        cloud.centers[i].update_coords(my_center_coords[i])
        cloud.centers[i].label = i
    cloud.lloyd()

    # Some magic that tests the accuracy and is not
    # mentioned here

if __name__ == "__main__":
    main()
