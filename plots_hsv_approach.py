import numpy as np
import darsia
import skimage
import skimage.color
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import string

letters = list(string.ascii_uppercase)


# from plots_chapter5_3 import extract_support_points

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

# RGB
diff = skimage.color.rgb2hsv(image.img) - skimage.color.rgb2hsv(baseline.img)
diff = -diff  # to comply with the darsia definition
# diff = skimage.color.rgb2hsv(diff)
# Regularize
smooth = skimage.restoration.denoise_tv_bregman(
    diff, weight=0.025, eps=1e-4, max_num_iter=100, isotropic=True
)

samples = [
    (slice(50, 150), slice(100, 200)),
    (slice(50, 150), slice(1600, 1700)),
]
concentrations = np.array([1, 0.95])

# visualise patches
fig, ax = plt.subplots()
ax.imshow(smooth)  # visualise abs colours, because relative cols are neg
ax.set_xlabel("horizontal pixel")
ax.set_ylabel("vertical pixel")

# double check number of patches
n = np.shape(samples)[0]  # number of patches
print("number of support patches: " + str(n))

# init colour vector
colours = np.zeros((n, 3))
# enumerate through all patches
for i, p in enumerate(samples):
    # visualise patches on image
    rect = patches.Rectangle(
        (p[1].start, p[0].start),
        p[1].stop - p[1].start,
        p[0].stop - p[0].start,
        linewidth=1,
        edgecolor="w",
        facecolor="none",
    )
    ax.text(p[1].start + 130, p[0].start + 100, letters[i], fontsize=15, color="white")
    ax.add_patch(rect)

    # histo analysis
    patch = smooth[p]
    # patch = skimage.color.rgb2hsv(patch)
    vals = patch[:, :, 0]
    h_hist, bins = np.histogram(vals, bins=100, range=(-1, 1))
    plt.figure("h" + letters[i])
    plt.stairs(h_hist, bins)


fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6, 2))
fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)
plt.ylabel("vertical pixel")
plt.xlabel("horizontal pixel")

# SIGNAL split
# reduction blue: B
# hsv = skimage.color.rgb2hsv(smooth)
hsv = np.copy(smooth)
scalar_blue = hsv[:, :, 2]
mask_hue = np.logical_and(
    hsv[:, :, 0] > -0.5,
    hsv[:, :, 0] < -0.4,
)
scalar_blue[~mask_hue] = 0
# ax1 = fig.add_subplot(211)
axes[0].imshow(scalar_blue, vmin=0, vmax=1)

# reduction green A
# hsv = skimage.color.rgb2hsv(smooth)
hsv = np.copy(smooth)
scalar_green = hsv[:, :, 2]
mask_hue = np.logical_and(
    hsv[:, :, 0] > -0.08,
    hsv[:, :, 0] < -0.04,
)
scalar_green[~mask_hue] = 0
# ax2 = fig.add_subplot(212)
axes[1].imshow(scalar_green, vmin=0, vmax=1)


fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6, 2))
fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)
plt.ylabel("vertical pixel")
plt.xlabel("horizontal pixel")

axes[0].imshow(scalar_blue + scalar_green, vmin=0, vmax=1)

# scale and weight scalar signals
weighted_signal = (
    scalar_blue / np.max(scalar_blue) * 0.95 + scalar_green / np.max(scalar_green) * 1
)
axes[1].imshow(weighted_signal, vmin=0, vmax=1)


# plt.figure("cut ph val")
# plt.plot(np.average(weighted_signal, axis=0))
# plt.xlabel("horizontal pixel")
# plt.ylabel("average concentration")
# plt.figure("cut ph val")
# plt.plot(np.average(scalar_blue + scalar_green, axis=0))
plt.show()
