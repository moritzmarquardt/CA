import numpy as np
import skimage
import matplotlib.pyplot as plt
from extract_support_points import extract_support_points
from model_experiment import model_experiment_full

baseline, image = model_experiment_full()

# RGB
diff_RGB = skimage.img_as_float(image.img) - skimage.img_as_float(baseline.img)

# Regularize
smooth_RGB = skimage.restoration.denoise_tv_bregman(
    diff_RGB, weight=0.025, eps=1e-4, max_num_iter=100, isotropic=True
)

samples = [
    (slice(50, 150), slice(100, 200)),
    (slice(50, 150), slice(1600, 1700)),
    (slice(50, 150), slice(2700, 2800)),
]
n = 3
colours_RGB = [[-0.29, -0.15, -0.21], [-0.67, -0.35, 0.05], [-0.01, -0.01, 0.01]]
concentrations = np.append(np.linspace(1, 0.9, n - 1), 0)


def color_to_concentration(
    k, colours, concentrations, signal: np.ndarray
) -> np.ndarray:
    x = np.array(colours)  # data points / control points / support points
    y = np.array(concentrations)  # goal points
    X = np.ones((x.shape[0], x.shape[0]))  # kernel matrix
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            X[i, j] = k(x[i], x[j])

    alpha = np.linalg.solve(X, y)

    # Estimator / interpolant
    def estim(signal):
        sum = 0
        for n in range(alpha.shape[0]):
            sum += alpha[n] * k(signal, x[n])
        return sum

    # estim = scipy.interpolate.LinearNDInterpolator(colours, concentrations, 0)

    ph_image = np.zeros(signal.shape[:2])
    for i in range(signal.shape[0]):
        for j in range(signal.shape[1]):
            ph_image[i, j] = estim(signal[i, j])
    return ph_image


# define linear kernel shifted to avoid singularities
def k_lin(x, y, a=0):
    return np.inner(x, y) + a


# define gaussian kernel
def k_gauss(x, y, gamma=9.7285528):  # rgb: 9.72855284651 , lab: 24.2242494195
    return np.exp(-gamma * np.inner(x - y, x - y))


# Convert a discrete ph stripe to a numeric pH indicator.
ph_image = color_to_concentration(
    k_gauss, colours_RGB, concentrations, smooth_RGB
)  # gamma=10 value retrieved from ph analysis kernel calibration war bester punk für c=0.95 was
# physikalisch am meisten sinn ergibt


ph_image[ph_image > 1] = 1  # für visualisierung von größer 1 values
ph_image[ph_image < 0] = 0
plt.figure("evolution of signal processing in a subregion")
plt.imshow(ph_image)
plt.ylabel("vertical pixel")
plt.xlabel("horizontal pixel")
plt.show()
