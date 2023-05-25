import numpy as np
import darsia
import skimage
from pathlib import Path
import matplotlib.pyplot as plt
from plots_chapter5_3 import extract_support_points

folder = Path("./data/tracer_timeseries/images")
baseline_path = folder / Path("20220914-142404.TIF")
image_path = folder / Path("20220914-151727.TIF")
# image_path = Path(".data/test/singletracer.JPG")

# Setup curvature correction (here only cropping)
curvature_correction = darsia.CurvatureCorrection(
    config={
        "crop": {
            # Define the pixel values (x,y) of the corners of the ROI.
            # Start at top left corner and then continue counterclockwise.
            "pts_src": [[300, 600], [300, 4300], [7600, 4300], [7600, 600]],
            # Specify the true dimensions of the reference points
            "width": 0.92,
            "height": 0.5,
        }
    }
)
transformations = [curvature_correction]

# Read-in images
# baseline = darsia.imread(baseline_path, transformations=transformations)
# image = darsia.imread(image_path, transformations=transformations)

baseline = darsia.imread(baseline_path, transformations=transformations).subregion(
    voxels=(slice(2300, 2500), slice(2200, 5200))
)
image = darsia.imread(image_path, transformations=transformations).subregion(
    voxels=(slice(2300, 2500), slice(2200, 5200))
)

# LAB
# diff = (
#     (skimage.color.rgb2lab(baseline.img) - skimage.color.rgb2lab(image.img))
#     + [0, 128, 128]
# ) / [100, 255, 255]

# RGB
diff = skimage.img_as_float(baseline.img) - skimage.img_as_float(image.img)

# Regularize
smooth = skimage.restoration.denoise_tv_bregman(
    diff, weight=0.1, eps=1e-4, max_num_iter=100, isotropic=True
)

pats = [
    (slice(50, 150), slice(100, 200)),
    (slice(50, 150), slice(1600, 1700)),
    (slice(50, 150), slice(2600, 2700)),
]
concentrations = np.array([1, 0.9, 0])
n, colours = extract_support_points(diff, pats)

# RGB CHOICE:
# colours = np.array([[0.28, 0.14, 0.2], [0.65, 0.34, 0.0], [0.01, 0.01, 0.0]])
# choice when histo analysis also allows for negative values:
# colours = np.array([[0.28, 0.14, 0.2], [0.66, 0.34, -0.06], [0.0, 0.0, -0.02]])
# LAB choice
# colours = np.array([[0.17, 0.56, 0.48], [0.38, 0.53, 0.75], [0.01, 0.5, 0.5]])


def color_to_ph(k, colours, concentrations, signal: np.ndarray) -> np.ndarray:
    # signal is rgb, transofrm to lab space because it is uniform and therefore
    # makes sense to interpolate in
    signal = skimage.color.rgb2lab(signal)
    colours = skimage.color.rgb2lab(colours)

    x = np.array(colours)  # data points / control points / support points
    y = np.array(concentrations)  # goal points
    X = np.ones((x.shape[0], x.shape[0]))  # kernel matrix
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            X[i, j] = k(x[i], x[j])

    alpha = np.linalg.solve(X, y)
    ph_image = np.zeros(signal.shape[:2])
    for i in range(signal.shape[0]):
        for j in range(signal.shape[1]):
            sum = 0
            for n in range(alpha.shape[0]):
                sum = sum + alpha[n] * k(signal[i, j], x[n])
            ph_image[i, j] = sum
    sum = 0
    for n in range(alpha.shape[0]):
        sum = sum + alpha[n] * k(colours[1], x[n])
    print(sum)
    return ph_image


# define linear kernel
def k_lin(x, y):
    return np.power(np.inner(x, y) + 1, 1)


# define gaussian kernel
def k_gauss(x, y, gamma=0.00001):
    return np.exp(-gamma * np.inner(x - y, x - y))


# Convert a discrete ph stripe to a numeric pH indicator.
ph_image = color_to_ph(
    k_lin, colours, concentrations, smooth
)  # gamma=10 value retrieved from ph analysis kernel calibration war bester punk für c=0.95 was
# physikalisch am meisten sinn ergibt
fig = plt.figure()
fig.suptitle("evolution of signal processing in a subregion")
ax = plt.subplot(313)
ax.set_title("ph-identifier")
ax.imshow(ph_image)
ax = plt.subplot(312)
ax.set_title("difference image - baseline")
ax.imshow(diff)
ax = plt.subplot(311)
ax.set_title("original image")
ax.imshow(skimage.img_as_ubyte(image.img))

# plt.figure("indicator")
# indicator = np.arange(101) / 100
# plt.axis("off")
# plt.imshow([indicator, indicator, indicator, indicator, indicator])


plt.figure("cut ph val")
plt.plot(np.average(ph_image, axis=0))
# plt.imshow(pwc.color_to_ph(smooth))
plt.show()
