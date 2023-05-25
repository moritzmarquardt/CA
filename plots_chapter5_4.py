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


baseline = darsia.imread(baseline_path, transformations=transformations).subregion(
    voxels=(slice(2300, 2500), slice(2200, 5200))
)
image = darsia.imread(image_path, transformations=transformations).subregion(
    voxels=(slice(2300, 2500), slice(2200, 5200))
)

# define support colours based on patches in the image that are representative for that colour
pats = [
    (slice(50, 150), slice(100, 200)),
    (slice(50, 150), slice(1600, 1700)),
    (slice(50, 150), slice(2600, 2700)),
]


#############################################################################################
#   RGB RGB RGB RGB RGB RGB RGB RGB RGB
#############################################################################################
####################################
# nn interpol
###################################
diff = skimage.img_as_float(baseline.img) - skimage.img_as_float(image.img)
# Regularize
smooth = skimage.restoration.denoise_tv_bregman(
    diff, weight=0.1, eps=1e-4, max_num_iter=100, isotropic=True
)
print("RGB:")
n, colours = extract_support_points(smooth, pats)
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
####################################
# nn interpol
###################################
diff = (
    (skimage.color.rgb2lab(baseline.img) - skimage.color.rgb2lab(image.img))
    + [0, 128, 128]
) / [100, 255, 255]
smooth = skimage.restoration.denoise_tv_bregman(
    diff, weight=0.1, eps=1e-4, max_num_iter=100, isotropic=True
)
print("LAB:")
n, colours = extract_support_points(smooth, pats)
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
    identifier[identifier == 1] = concentrations[1]
    identifier[identifier == 0] = concentrations[0]
    identifier[identifier == 2] = concentrations[2]
    return identifier


plt.figure("nn interpol lab")
plt.imshow(closest_color_LAB(smooth))

plt.show()
