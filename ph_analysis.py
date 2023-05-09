import numpy as np
import darsia
import skimage
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndi
from ph_tailoredClasses import PHIndicator

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
    voxels=(slice(2400, 2600), slice(2000, 5000))
)
image = darsia.imread(image_path, transformations=transformations).subregion(
    voxels=(slice(2400, 2600), slice(2000, 5000))
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

plt.figure("smooth")
plt.imshow(smooth)

# split the signals
a, b, c = cv2.split(smooth)
# reduce to one line and smooth the signals
sigma = 100
smooth_a = ndi.gaussian_filter1d(np.average(a, axis=0), sigma=sigma)
smooth_b = ndi.gaussian_filter1d(np.average(b, axis=0), sigma=sigma)
smooth_c = ndi.gaussian_filter1d(np.average(c, axis=0), sigma=sigma)

# Plot single components
plt.figure("colour space components")
plt.plot(smooth_a, color="red")
plt.plot(smooth_b, color="green")
plt.plot(smooth_c, color="blue")

colours = np.array(
    [
        [smooth_a[0], smooth_b[0], smooth_c[0]],
        [smooth_a[2000], smooth_b[2000], smooth_c[2000]],
        [smooth_a[2999], smooth_b[2999], smooth_c[2999]],
    ]
)
concentrations = np.array([1, 0.95, 0])
print("colours", colours)

# RGB CHOICE:
# green_rgb = np.array([0.3125, 0.1647, 0.1913])
# blue_rgb = np.array([0.6693, 0.3575, -0.05])
# black_rgb = np.array([0, 0, 0])

# plt.figure("colours")
# colour = np.ones((10, 10, 3))
# colour[:, :] = green_rgb
# plt.imshow(colour)
# concentration_blue = 0.5  # need expert knowledge

# Convert a discrete ph stripe to a numeric pH indicator.
pwc = PHIndicator(colours, concentrations)
ph_image = pwc.color_to_ph(smooth)
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

plt.figure("indicator")
indicator = np.arange(101) / 100
plt.axis("off")
plt.imshow([indicator, indicator, indicator, indicator, indicator])


plt.figure("cut ph val")
plt.plot(np.average(ph_image, axis=0))
# plt.imshow(pwc.color_to_ph(smooth))
plt.show()
