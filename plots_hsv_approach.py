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
diff = skimage.img_as_float(image.img) - skimage.img_as_float(baseline.img)
diff = -diff  # to comply with the darsia definition

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
ax.imshow(np.abs(smooth))  # visualise abs colours, because relative cols are neg
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
    patch = skimage.color.rgb2hsv(patch)
    vals = patch[:, :, 0]
    h_hist, bins = np.histogram(vals, bins=100, range=(0, 1))
    plt.figure("h" + letters[i])
    plt.stairs(h_hist, bins)

plt.figure()
plt.subplot(4, 1, 1)
plt.xlabel("horizontal pixel")
plt.ylabel("vertical pixel")
plt.imshow(smooth)

# SIGNAL split
# reduction blue:
hsv = skimage.color.rgb2hsv(smooth)
scalar_blue = hsv[:, :, 2]
mask_hue = np.logical_and(
    hsv[:, :, 0] > 0.05,
    hsv[:, :, 0] < 0.1,
)
scalar_blue[~mask_hue] = 0
plt.subplot(4, 1, 3)
plt.xlabel("horizontal pixel")
plt.ylabel("vertical pixel")
plt.imshow(scalar_blue, vmin=0, vmax=1)

# reduction green
hsv = skimage.color.rgb2hsv(smooth)
scalar_green = hsv[:, :, 2]
mask_hue = np.logical_and(
    hsv[:, :, 0] > 0.9,
    hsv[:, :, 0] < 0.95,
)
scalar_green[~mask_hue] = 0
plt.subplot(4, 1, 2)
plt.xlabel("horizontal pixel")
plt.ylabel("vertical pixel")
plt.imshow(scalar_green, vmin=0, vmax=1)

plt.subplot(4, 1, 4)
plt.xlabel("horizontal pixel")
plt.ylabel("vertical pixel")
plt.imshow(scalar_blue + scalar_green, vmin=0, vmax=1)


plt.figure("cut ph val")
plt.plot(np.average(scalar_blue + scalar_green, axis=0))

# scale and weight scalar signals
weighted_signal = (
    scalar_blue / np.max(scalar_blue) * 0.95 + scalar_green / np.max(scalar_green) * 1
)
plt.figure("weighted signal")
plt.imshow(weighted_signal, vmin=0, vmax=1)
plt.xlabel("horizontal pixel")
plt.ylabel("vertical pixel")
plt.figure("cut ph val")
plt.plot(np.average(weighted_signal, axis=0))
plt.xlabel("horizontal pixel")
plt.ylabel("average concentration")

plt.show()
