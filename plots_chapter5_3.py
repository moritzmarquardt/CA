import numpy as np
import darsia
import skimage
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches


def extract_support_points(signal, samples):
    fig, ax = plt.subplots()
    ax.imshow(signal)
    n = np.shape(samples)[0]  # number of patches
    print("number of support patches: " + str(n))
    colours = np.zeros((n, 3))
    i = 0
    for p in samples:
        rect = patches.Rectangle(
            (p[1].start, p[0].start),
            p[1].stop - p[1].start,
            p[0].stop - p[0].start,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

        # histo analysis
        patch = signal[p]
        # plt.figure("patch nr. " + str(i))
        # plt.imshow(patch)

        r, g, b = cv2.split(patch)
        bins = 100
        r_hist, b_r = np.histogram(r, bins=bins, range=(-1, 1))
        g_hist, b_g = np.histogram(g, bins=bins, range=(-1, 1))
        b_hist, b_b = np.histogram(b, bins=bins, range=(-1, 1))
        char_colour = [
            (b_r[np.argmax(r_hist)] + b_r[np.argmax(r_hist) + 1]) / 2,
            (b_g[np.argmax(g_hist)] + b_g[np.argmax(g_hist) + 1]) / 2,
            (b_b[np.argmax(b_hist)] + b_b[np.argmax(b_hist) + 1]) / 2,
        ]
        colours[i] = char_colour
        # plt.figure("char colour nr." + str(i))
        # plt.imshow(np.ones((100, 100, 3)) * char_colour)
        i = i + 1

    # r, g, b = cv2.split(signal[samples[0]])
    # plt.figure("hist r für erste colour")
    # h, b = np.histogram(r, bins=100, range=(-1, 1))
    # plt.stairs(h, b)
    print("characteristic colours: " + str(colours))
    return n, colours


if __name__ == "__main__":
    folder = Path("./data/tracer_timeseries/images")
    baseline_path = folder / Path("20220914-142404.TIF")
    image_path = folder / Path("20220914-151727.TIF")
    curvature_correction = darsia.CurvatureCorrection(
        config={
            "crop": {
                "pts_src": [[300, 600], [300, 4300], [7600, 4300], [7600, 600]],
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
    diff = skimage.img_as_float(baseline.img) - skimage.img_as_float(image.img)
    # Regularize
    smooth = skimage.restoration.denoise_tv_bregman(
        diff, weight=0.1, eps=1e-4, max_num_iter=100, isotropic=True
    )
    print("RGB:")
    n, colours = extract_support_points(smooth, pats)

    #############################################################################################
    #   LAB LAB LAB LAB LAB LAB LAB LAB LAB
    #############################################################################################
    diff = (
        (skimage.color.rgb2lab(baseline.img) - skimage.color.rgb2lab(image.img))
        + [0, 128, 128]
    ) / [100, 255, 255]
    smooth = skimage.restoration.denoise_tv_bregman(
        diff, weight=0.1, eps=1e-4, max_num_iter=100, isotropic=True
    )
    print("LAB:")
    n, colours = extract_support_points(smooth, pats)

    plt.show()
