"""
define the model experiment used for the work
import the baseline and tracer image by using  baseline, image = model_experiment() in the script

when running only this script, the model experiment is plotted
"""
import darsia
from pathlib import Path
import matplotlib.pyplot as plt
import skimage
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

# Read-in images
baseline = darsia.imread(baseline_path, transformations=transformations).subregion(
    voxels=(slice(2300, 2500), slice(2200, 5200))
)
image = darsia.imread(image_path, transformations=transformations).subregion(
    voxels=(slice(2300, 2500), slice(2200, 5200))
)


baseline_full = darsia.imread(baseline_path, transformations=transformations)
image_full = darsia.imread(image_path, transformations=transformations)


def model_experiment():
    return baseline, image


def model_experiment_full():
    return baseline_full, image_full


if __name__ == "__main__":
    plt.figure("corrected tracer image")
    plt.imshow(skimage.img_as_ubyte(image.img))
    plt.xlabel("horizontal pixel")
    plt.ylabel("vertical pixel")

    plt.figure("corrected baseline image")
    plt.imshow(skimage.img_as_ubyte(baseline.img))
    plt.xlabel("horizontal pixel")
    plt.ylabel("vertical pixel")

    plt.figure("corrected tracer image full")
    plt.imshow(skimage.img_as_ubyte(image_full.img))
    plt.xlabel("horizontal pixel")
    plt.ylabel("vertical pixel")

    plt.figure("corrected baseline image full")
    plt.imshow(skimage.img_as_ubyte(baseline_full.img))
    plt.xlabel("horizontal pixel")
    plt.ylabel("vertical pixel")

    fig, ax = plt.subplots()
    ax.imshow(skimage.img_as_float(image_full.img))
    rect = patches.Rectangle(
        (2200, 2400),
        3000,
        5,
        linewidth=1,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_xlabel("horizontal pixel")
    ax.set_ylabel("vertical pixel")

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
    ax.set_xlabel("horizontal pixel")
    ax.set_ylabel("vertical pixel")

    plt.show()
