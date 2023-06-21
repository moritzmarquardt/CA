import numpy as np
import skimage
import matplotlib.pyplot as plt
from extract_support_points import extract_support_points
from model_experiment import model_experiment

baseline, image = model_experiment()

# LAB
diff_LAB = (
    (skimage.color.rgb2lab(image.img) - skimage.color.rgb2lab(baseline.img))
    + [0, 128, 128]
) / [100, 255, 255]

# RGB
diff_RGB = skimage.img_as_float(image.img) - skimage.img_as_float(baseline.img)

# HSV
# diff = skimage.color.rgb2hsv(baseline.img) - skimage.color.rgb2hsv(image.img)

# Regularize
smooth_RGB = skimage.restoration.denoise_tv_bregman(
    diff_RGB, weight=0.025, eps=1e-4, max_num_iter=100, isotropic=True
)
smooth_LAB = skimage.restoration.denoise_tv_bregman(
    diff_LAB, weight=0.025, eps=1e-4, max_num_iter=100, isotropic=True
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
n, colours_RGB = extract_support_points(signal=smooth_RGB, samples=samples)
n, colours_LAB = extract_support_points(signal=smooth_LAB, samples=samples)
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
def k_gauss(x, y, gamma=10):
    return np.exp(-gamma * np.inner(x - y, x - y))


# Convert a discrete ph stripe to a numeric pH indicator.
ph_image = color_to_concentration(
    k_gauss, colours_RGB, concentrations, smooth_RGB
)  # gamma=10 value retrieved from ph analysis kernel calibration war bester punk für c=0.95 was
# physikalisch am meisten sinn ergibt

plt.figure("cut ph val")
plt.plot(np.average(ph_image, axis=0))
plt.xlabel("horizontal pixel")
plt.ylabel("concentration")
# plt.plot(np.average(ph_image[50:70, :], axis=0))
# plt.imshow(pwc.color_to_concentration(smooth))


ph_image[ph_image > 1] = 1  # für visualisierung von größer 1 values
ph_image[ph_image < 0] = 0
fig = plt.figure()
fig.suptitle("evolution of signal processing in a subregion")
ax = plt.subplot(212)
ax.imshow(ph_image)
ax.set_ylabel("vertical pixel")
ax.set_xlabel("horizontal pixel")

# ax = plt.subplot(312)
# ax.set_title("difference image - baseline")
# ax.imshow(diff)
ax = plt.subplot(211)
ax.imshow(skimage.img_as_ubyte(image.img))
ax.set_ylabel("vertical pixel")
ax.set_xlabel("horizontal pixel")

# plt.figure("indicator")
# indicator = np.arange(101) / 100
# plt.axis("off")
# plt.imshow([indicator, indicator, indicator, indicator, indicator])
plt.show()
