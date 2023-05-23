"""
plot the model experiment plots with region of interest
"""
import darsia as da
from pathlib import Path
import darsia
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

folder = Path("./data/tracer_timeseries/images")
baseline_path = folder / Path("20220914-142404.TIF")
image_path = folder / Path("20220914-151727.TIF")
# image_path = Path(".data/test/singletracer.JPG")

# Setup curvature correction (here only cropping)
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
transformations = [curvature_correction]

# Read-in images
image_full = darsia.imread(image_path, transformations=transformations)

# choose representative roi for the whole chapter 5
# philosophy: all three colours with same amout and close to injection point and big gradient and
# horizontal for easy 1d reduction and no consideration of gravity
image = da.imread(image_path, transformations=transformations).subregion(
    voxels=(slice(2300, 2500), slice(2200, 5200))
)

plt.figure()
plt.imshow(skimage.img_as_float(image.img))

fig, ax = plt.subplots()
ax.imshow(skimage.img_as_float(image_full.img))
rect = patches.Rectangle(
    (2200, 2300),
    3000,
    200,
    linewidth=1,
    edgecolor="black",
    facecolor="none",
)
ax.add_patch(rect)

plt.show()
