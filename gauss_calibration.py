import numpy as np
import skimage
from extract_support_points import extract_support_points
import scipy.optimize as so

from model_experiment import model_experiment

baseline, image = model_experiment()

# LAB
diff = (
    (skimage.color.rgb2lab(baseline.img) - skimage.color.rgb2lab(image.img))
    + [0, 128, 128]
) / [100, 255, 255]

# RGB
# diff = skimage.img_as_float(baseline.img) - skimage.img_as_float(image.img)

# Regularize
smooth = skimage.restoration.denoise_tv_bregman(
    diff, weight=0.025, eps=1e-4, max_num_iter=100, isotropic=True
)

samples = [
    (slice(50, 150), slice(100, 200)),
    # (slice(50, 150), slice(400, 500)),
    # (slice(50, 150), slice(600, 700)),
    # (slice(50, 150), slice(800, 900)),
    # (slice(50, 150), slice(1000, 1100)),
    # (slice(50, 150), slice(1200, 1300)),
    # (slice(50, 150), slice(1400, 1500)),
    (slice(50, 150), slice(1600, 1700)),
    (slice(50, 150), slice(2700, 2800)),
]
n, colours = extract_support_points(signal=smooth, samples=samples)
concentrations = np.append(np.linspace(1, 0.9, n - 1), 0)


def color_to_concentration(
    k, colours, concentrations, signal: np.ndarray
) -> np.ndarray:
    # signal is rgb, transofrm to lab space because it is uniform and therefore
    # makes sense to interpolate in
    # signal = skimage.color.rgb2lab(signal)
    # colours = skimage.color.rgb2lab(colours)

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

    ph_image = np.zeros(signal.shape[:2])
    for i in range(signal.shape[0]):
        for j in range(signal.shape[1]):
            ph_image[i, j] = estim(signal[i, j])
    return ph_image


# for g in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 9, 9.5, 10, 10.5, 11, 20, 30, 40, 50]:

#     def k_gauss(x, y, gamma=g):
#         return np.exp(-gamma * np.inner(x - y, x - y))

#     ph_image = color_to_concentration(k_gauss, colours, concentrations, smooth)
#     print(g, np.average(ph_image[80:120, 1000:1040]))


def objective_function(gamma):
    def k_gauss(x, y, gamma=gamma):
        return np.exp(-gamma * np.inner(x - y, x - y))

    ph_image = color_to_concentration(k_gauss, colours, concentrations, smooth)
    # print(g, np.average(ph_image[80:120, 1000:1040]))

    return np.abs(0.95 - np.average(ph_image[80:120, 1000:1040]))


# + np.abs(0.975 - np.average(ph_image[80:120, 500:540]))


res = so.minimize_scalar(objective_function, bounds=(1, 100))
print(res)
