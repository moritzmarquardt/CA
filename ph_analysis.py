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
    voxels=(slice(2400, 2600), slice(2500, 5000))
)
image = darsia.imread(image_path, transformations=transformations).subregion(
    voxels=(slice(2400, 2600), slice(2500, 5000))
)

diff = skimage.img_as_float(baseline.img) - skimage.img_as_float(image.img)

# Regularize
smooth = skimage.restoration.denoise_tv_bregman(
    diff, weight=0.1, eps=1e-4, max_num_iter=100, isotropic=True
)

# Reduce to one line
r, g, b = cv2.split(smooth)
r_one_line = np.average(r, axis=0)
g_one_line = np.average(g, axis=0)
b_one_line = np.average(b, axis=0)

# Smooth the signals
sigma = 100
smooth_r = ndi.gaussian_filter1d(r_one_line, sigma=sigma)
smooth_g = ndi.gaussian_filter1d(g_one_line, sigma=sigma)
smooth_b = ndi.gaussian_filter1d(b_one_line, sigma=sigma)

# Plot single components
plt.figure("rgb")
plt.plot(smooth_r, color="red")
plt.plot(smooth_g, color="green")
plt.plot(smooth_b, color="blue")

green_rgb = np.array([0.3125, 0.1647, 0.1913])  # equals concentration 1
blue_rgb = np.array([0.6693, 0.3575, -0.05])  # TODO consider negative values for blue!
black_rgb = np.array([0, 0, 0])
# concentration_blue = 0.5  # need expert knowledge

# Convert a discrete ph stripe to a numeric pH indicator.
pwc = PHIndicator([black_rgb, blue_rgb, green_rgb], [0, 0.9, 1])
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


plt.figure("cut ph val")
plt.plot(np.average(ph_image, axis=0))
# plt.imshow(pwc.color_to_ph(smooth))
plt.show()
