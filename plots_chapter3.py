import darsia as da
from pathlib import Path
import matplotlib.pyplot as plt
import skimage
import numpy as np

folder = Path("./data/tracer_timeseries/images")
baseline_path = folder / Path("20220914-142404.TIF")
image_path = folder / Path("20220914-151727.TIF")
# image_path = Path(".data/test/singletracer.JPG")

# Setup curvature correction (here only cropping)
# CURVATURE
curvature_correction = da.CurvatureCorrection(
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
# curvature_correction.pre_bulge_correction(horizontal_bulge=5e-9)
# curvature_correction.bulge_corection(left=70, right=60, top=0, bottom=0)
# COLOR
# Preprocessing of the images:
# roi_color_checker = [
#     [6787, 77],
#     [6793, 555],
#     [7510, 555],
#     [7502, 78],
# ]
# color_correction = da.ColorCorrection(
#     roi=roi_color_checker,
#     verbosity=False,
#     # baseline=
# )
# transformations = [curvature_correction, color_correction]
transformations = [curvature_correction]

# original images:
baseline = da.imread(baseline_path)
image = da.imread(image_path)
# plt.figure("original baseline")
# plt.imshow(skimage.img_as_float(baseline.img))
plt.figure("original tracer image")
plt.xlabel("horizontal pixel")
plt.ylabel("vertical pixel")
plt.imshow(skimage.img_as_float(image.img))


# Corrected
baseline = da.imread(baseline_path, transformations=transformations)
image = da.imread(image_path, transformations=transformations)
plt.figure("corrected baseline")
plt.xlabel("horizontal pixel")
plt.ylabel("vertical pixel")
plt.imshow(skimage.img_as_float(baseline.img))
plt.figure("corrected tracer image")
plt.xlabel("horizontal pixel")
plt.ylabel("vertical pixel")
plt.imshow(skimage.img_as_float(image.img))

# Difference
diff = skimage.img_as_float(image.img) - skimage.img_as_float(baseline.img)
plt.figure("difference image - baseline")
plt.xlabel("horizontal pixel")
plt.ylabel("vertical pixel")
plt.imshow(diff)
plt.figure("norm of difference")
plt.xlabel("horizontal pixel")
plt.ylabel("vertical pixel")
plt.imshow(np.abs(diff))


# Smoothing
smooth = skimage.restoration.denoise_tv_bregman(
    diff, weight=0.025, eps=1e-4, max_num_iter=100, isotropic=True
)
plt.figure("smooth difference")
plt.xlabel("horizontal pixel")
plt.ylabel("vertical pixel")
plt.imshow(smooth)
plt.figure("norm of smooth")
plt.xlabel("horizontal pixel")
plt.ylabel("vertical pixel")
plt.imshow(np.abs(smooth))

plt.show()
