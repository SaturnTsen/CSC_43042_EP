import time
from sklearn.neighbors import KernelDensity
from TD.kernels import *
import numpy as np
from scipy.stats import norm, uniform
from scipy.integrate import quad
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad
import bisect
import matplotlib.pyplot as plt

# from scipy.integrate import quad, romberg
import matplotlib 
matplotlib.use('Agg')

def generate_data_2d(n_samples:int, index:int, density_type="gaussian_mixture"):
    if density_type == "gaussian_mixture":
        if index==1:
            weights = [0.6, 0.4]
            means = [[-2, 2], [2, -2]]
            covs = [[[1, 0.], [0., 1]], [[1, -0.], [-0., 1]]]  # Matrices de covariance
        if index==2:
            weights = [0.6, 0.4]
            means = [[-2, 2], [2, -2]]
            covs = [[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]]  # Matrices de covariance
        if index==3:
            weights = [0.3, 0.3,0.4]
            means = [[-2, 2], [2, -2], [3, 0]]
            covs = [[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], [[18, -0.5], [-0.5, 60]]]  # Matrices de covariance
        if index==4:
            weights = [0.3, 0.2,0.2,0.3]
            means = [[-2, 2], [2, -2], [3, 0], [0, 0]]
            covs = [[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], [[1, -0.5], [-0.5, 5]], [[1, -3], [-3, 10]]]  # Matrices de covariance
        data = np.concatenate([
            np.random.multivariate_normal(mean=means[i], cov=covs[i], size=int(n_samples * weights[i]))
            for i in range(len(weights))
        ])
        true_density = lambda x, y: np.sum([weights[i] * multivariate_normal.pdf([x, y], mean=means[i], cov=covs[i]) for i in range(len(weights))])

    elif density_type == "uniform":
        data = np.random.uniform(low=-3, high=3, size=(n_samples, 2))
        true_density = lambda x, y: uniform.pdf(x, loc=-3, scale=6) * uniform.pdf(y, loc=-3, scale=6)

    else:  # Default to 2D standard normal
        data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=n_samples)
        true_density = lambda x, y: multivariate_normal.pdf([x, y], mean=[0, 0], cov=[[1, 0], [0, 1]])
    return data, true_density






def total_variation_distance_2d(p, q, lower=-np.inf, upper=np.inf):
    """Calcule la distance en variation totale entre deux densités 2D."""
    def integrand(x, y):
        return np.abs(q(x, y) - p(x, y))
    tv_distance, error_estimate =   dblquad(integrand, lower, upper, lambda x: lower, lambda x: upper, epsabs=0.1)  # scipy.integrate.dblquad
    return 0.5 *tv_distance

def l2_distance_2d(p, q, lower=-np.inf, upper=np.inf):
    """Calcule la distance L2 entre deux densités 2D."""
    integrand = lambda x, y: np.square(q(x, y) - p(x, y))
    l2_distance, _ = dblquad(integrand, lower, upper, lambda x: lower, lambda x: upper,
                             epsabs=0.5)  # scipy.integrate.dblquad
    return l2_distance



### Display plots
for job in [1, 2, 3, 4]:
    #### Density estimation with scikit-learn
    n_samples = [100,100,200,500][job-1]

    print(f"You are running estimate_TV on job {job}. The number of samples is {n_samples}")
    init_time=time.time()
    data, true_density = generate_data_2d(n_samples, index=job)
    kde = KernelDensity(bandwidth='silverman', kernel='gaussian')
    kde.fit(data)
    estimated_density_sk = lambda x, y: np.exp(kde.score_samples(np.array([[x, y]])))[0]
    tv_distance_sk = total_variation_distance_2d(true_density, estimated_density_sk)
    print(f"The TV distance reached by scikit-learn on this instance is {tv_distance_sk:.2f}")

    #### Density estimation with your code (change guess_bandwidth() to guess_bandwidth_challenge()!)
    kg = Gaussian(2, data, 1.)
    estimated_density = lambda x, y: kg.density(np.array([x, y]))

    #### Comment the following line and uncomment the next one:
    # kg.guess_bandwidth()
    kg.guess_bandwidth_challenge()

    print(f"Your Gaussian kernel has chosen a bandwidth of {kg.bandwidth:.2f}")
    estimated_density = lambda x, y: kg.density(np.array([x, y]))
    tv_distance = total_variation_distance_2d(true_density, estimated_density)
    print(f"The TV distance reached by your kernel on this instance is {tv_distance:.2f}")


    #### Print results:
    ratio=tv_distance / tv_distance_sk
    print(f"Ratio (your TV distance)/(scikit-learn's TV distance)= {ratio:.2f}")

    intervalles=[0.9,1.2,1.8,4]
    notes=["A+", "A", "B", "C", "D"]
    print(f"This places you in the range of grade {notes[bisect.bisect_left(intervalles,ratio)]}")

    print(f"Time taken {time.time()-init_time}s")


    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z_true = np.array([[true_density(x, y) for x in x_range] for y in y_range])
    Z_estimated = np.array([[estimated_density(x, y) for x in x_range] for y in y_range])
    Z_estimated_sk = np.array([[estimated_density_sk(x, y) for x in x_range] for y in y_range])

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, Z_true, cmap="viridis")  # D'abord la densité
    plt.scatter(data[:, 0], data[:, 1], s=10, c='red', label="Data points")  # Ensuite les points par-dessus
    plt.title("True density and data points")
    plt.legend()  # To show the label

    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, Z_estimated_sk, cmap="viridis")
    plt.title(f"Estimated (sk) TV={tv_distance_sk:.2f}")

    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, Z_estimated, cmap="viridis")
    plt.title(f"Estimated (yours) TV={tv_distance:.2f}")

    plt.suptitle(f"2D density estimation")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()
    plt.savefig(f'./results/res_{job}.png')
