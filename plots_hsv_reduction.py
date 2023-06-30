import numpy as np
import skimage
import skimage.color
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import string
from model_experiment import model_experiment

letters = list(string.ascii_uppercase)
baseline, image = model_experiment()


# RGB
diff = skimage.color.rgb2hsv(image.img) - skimage.color.rgb2hsv(baseline.img)
diff = -diff  # to comply with the darsia definition

# Regularize
smooth = skimage.restoration.denoise_tv_bregman(
    diff, weight=0.025, eps=1e-4, max_num_iter=100, isotropic=True
)

samples = [
    (slice(50, 150), slice(100, 200)),
    (slice(50, 150), slice(1600, 1700)),
]
concentrations = np.array([1, 0.9])

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
    scalar_blue / np.max(scalar_blue) * 0.9 + scalar_green / np.max(scalar_green) * 1
)
axes[1].imshow(weighted_signal, vmin=0, vmax=1)

plt.figure()
plt.imshow(scalar_blue + scalar_green, vmin=0, vmax=1)
plt.xlabel("horizontal pixel")
plt.ylabel("vertical pixel")

plt.figure("cut ph val")
plt.plot(np.average(weighted_signal, axis=0))
plt.xlabel("horizontal pixel")
plt.ylabel("signal value")


# plt.figure("cut ph val")
# plt.plot(np.average(weighted_signal, axis=0))
# plt.xlabel("horizontal pixel")
# plt.ylabel("average concentration")
# plt.figure("cut ph val")
# plt.plot(np.average(scalar_blue + scalar_green, axis=0))
plt.show()
