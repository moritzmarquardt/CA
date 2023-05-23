import numpy as np
import darsia
import skimage
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches

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


baseline = darsia.imread(baseline_path, transformations=transformations).subregion(
    voxels=(slice(2300, 2500), slice(2200, 5200))
)
image = darsia.imread(image_path, transformations=transformations).subregion(
    voxels=(slice(2300, 2500), slice(2200, 5200))
)

diff = skimage.img_as_float(baseline.img) - skimage.img_as_float(image.img)


# Regularize
smooth = skimage.restoration.denoise_tv_bregman(
    diff, weight=0.1, eps=1e-4, max_num_iter=100, isotropic=True
)

fig, ax = plt.subplots()
ax.imshow(smooth)
rect = patches.Rectangle(
    (100, 50),
    100,
    100,
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(rect)
rect = patches.Rectangle(
    (1600, 50),
    100,
    100,
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(rect)
rect = patches.Rectangle(
    (2600, 50),
    100,
    100,
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(rect)

###################################################
# für einen patch als probe

patch = diff[50:150, 100:200, :]

r, g, b = cv2.split(patch)

bins = 100
# Setup histograms
r_hist = np.histogram(r, bins=bins, range=(0, 1))[0]
g_hist = np.histogram(g, bins=bins, range=(0, 1))[0]
b_hist = np.histogram(b, bins=bins, range=(0, 1))[0]
plt.figure("patch, r,b,g")
plt.subplot(1, 4, 1)
plt.imshow(patch)
plt.subplot(1, 4, 2)
plt.plot(np.linspace(0, 1, bins), r_hist)
plt.plot(np.argmax(r_hist) / bins, r_hist[np.argmax(r_hist)], "x")
plt.subplot(1, 4, 3)
plt.plot(np.linspace(0, 1, bins), g_hist)
plt.subplot(1, 4, 4)
plt.plot(np.linspace(0, 1, bins), b_hist)


#############################################################################################
#   RGB RGB RGB RGB RGB RGB RGB RGB RGB
#############################################################################################

#################################
# alle patches durch um alle farben für interpol zu retrieven
pats = [
    diff[50:150, 100:200, :],
    diff[50:150, 1600:1700, :],
    diff[50:150, 2600:2700, :],
]
n = np.shape(pats)[0]  # number of patches
colours = np.zeros((n, 3))
i = 0
for patch in pats:
    r, g, b = cv2.split(patch)
    bins = 100
    # Setup histograms
    r_hist = np.histogram(r, bins=bins, range=(0, 1))[0]
    g_hist = np.histogram(g, bins=bins, range=(0, 1))[0]
    b_hist = np.histogram(b, bins=bins, range=(0, 1))[0]
    char_colour = [
        np.argmax(r_hist) / bins,
        np.argmax(g_hist) / bins,
        np.argmax(b_hist) / bins,
    ]
    colours[i] = char_colour
    i = i + 1

print(colours)
####################################
# nn interpol
###################################
concentrations = np.array([1, 0.9, 0])


def closest_color_RGB(signal: np.ndarray) -> np.ndarray:
    signal_shape = signal.shape[:2]
    distance = np.zeros((n, *signal_shape), dtype=float)
    for i in range(n):
        mono_colored_image = np.outer(np.ones(signal_shape), colours[i]).reshape(
            signal.shape
        )
        distance[i] = np.sqrt(np.sum(np.power(signal - mono_colored_image, 2), axis=2))
    identifier = np.argmin(distance, axis=0).astype(float)
    # for i in range(n):  # replace argmin 1,2,3,... with the actual concentration values
    # TODO not hardcode the int to concentration
    # order is important, if 0 gets translatet to 1 first, then 1 gets translated back to
    # another concentration
    identifier[identifier == 1] = concentrations[1]
    identifier[identifier == 0] = concentrations[0]
    identifier[identifier == 2] = concentrations[2]
    return identifier


plt.figure("nn interpol rgb")
plt.imshow(closest_color_RGB(smooth))

###########################################################################################
#   LAB LAB LAB LAB LAB LAB LAB LAB LAB
#############################################################################################

diff = (
    (skimage.color.rgb2lab(baseline.img) - skimage.color.rgb2lab(image.img))
    + [0, 128, 128]
) / [100, 255, 255]

smooth = skimage.restoration.denoise_tv_bregman(
    diff, weight=0.1, eps=1e-4, max_num_iter=100, isotropic=True
)

fig, ax = plt.subplots()
ax.imshow(diff)
rect = patches.Rectangle(
    (100, 50),
    100,
    100,
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(rect)
rect = patches.Rectangle(
    (1600, 50),
    100,
    100,
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(rect)
rect = patches.Rectangle(
    (2600, 50),
    100,
    100,
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(rect)

#################################
# alle patches durch um alle farben für interpol zu retrieven
pats = [
    diff[50:150, 100:200, :],
    diff[50:150, 1600:1700, :],
    diff[50:150, 2600:2700, :],
]
n = np.shape(pats)[0]  # number of patches
colours = np.zeros((n, 3))
i = 0
for patch in pats:
    r, g, b = cv2.split(patch)
    bins = 100
    # Setup histograms
    r_hist = np.histogram(r, bins=bins, range=(0, 1))[0]
    g_hist = np.histogram(g, bins=bins, range=(0, 1))[0]
    b_hist = np.histogram(b, bins=bins, range=(0, 1))[0]
    char_colour = [
        np.argmax(r_hist) / bins,
        np.argmax(g_hist) / bins,
        np.argmax(b_hist) / bins,
    ]
    colours[i] = char_colour
    i = i + 1

print(colours)
####################################
# nn interpol
###################################
concentrations = np.array([1, 0.9, 0])


def closest_color_LAB(signal: np.ndarray) -> np.ndarray:
    signal_shape = signal.shape[:2]  # picture size e.g. (200, 4500)
    distance = np.zeros((n, *signal_shape), dtype=float)
    for i in range(n):
        mono_colored_image = np.outer(np.ones(signal_shape), colours[i]).reshape(
            signal.shape
        )
        distance[i] = np.sqrt(np.sum(np.power(signal - mono_colored_image, 2), axis=2))
    identifier = np.argmin(distance, axis=0).astype(
        float
    )  # nötig für die concentration assignment
    print(identifier[5, 1700])
    identifier[identifier == 1] = concentrations[1]
    print(identifier[5, 1700])
    identifier[identifier == 0] = concentrations[0]
    print(identifier[5, 1700])
    identifier[identifier == 2] = concentrations[2]
    print(identifier[5, 1700])
    return identifier


plt.figure("nn interpol lab")
plt.imshow(closest_color_LAB(smooth))

plt.show()
