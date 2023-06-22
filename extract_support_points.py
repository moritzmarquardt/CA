import numpy as np
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import string
from model_experiment import model_experiment

letters = list(string.ascii_uppercase)


def extract_support_points(signal, samples):
    # visualise patches
    fig, ax = plt.subplots()
    ax.imshow(np.abs(signal))  # visualise abs colours, because relative cols are neg
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
        # ax.text(
        #     p[1].start + 130, p[0].start + 100, letters[i], fontsize=15, color="white"
        # )
        ax.text(p[1].start + 5, p[0].start + 95, letters[i], fontsize=12, color="white")
        ax.add_patch(rect)

        # histo analysis
        patch = signal[p]
        flat_image = np.reshape(patch, (10000, 3))  # all pixels in one dimension
        # patch visualisation
        # plt.figure("patch" + letters[i])
        # plt.imshow(np.abs(patch))
        H, edges = np.histogramdd(
            flat_image, bins=100, range=[(-1, 1), (-1, 1), (-1, 1)]
        )
        print(H.shape)
        index = np.unravel_index(H.argmax(), H.shape)
        col = [
            (edges[0][index[0]] + edges[0][index[0] + 1]) / 2,
            (edges[1][index[1]] + edges[1][index[1] + 1]) / 2,
            (edges[2][index[2]] + edges[2][index[2] + 1]) / 2,
        ]
        colours[i] = col

    c = np.abs(colours)
    plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(colours[:, 0], colours[:, 1], colours[:, 2], c=c)
    for i, c in enumerate(colours):
        ax.text(c[0], c[1], c[2], letters[i])

    print("characteristic colours: " + str(colours))
    return n, colours


if __name__ == "__main__":
    baseline, image = model_experiment()

    # define support colours based on patches in the image that are representative for that colour
    pats = [
        (slice(50, 150), slice(100, 200)),
        (slice(50, 150), slice(400, 500)),
        (slice(50, 150), slice(600, 700)),
        (slice(50, 150), slice(800, 900)),
        (slice(50, 150), slice(1000, 1100)),
        (slice(50, 150), slice(1200, 1300)),
        (slice(50, 150), slice(1400, 1500)),
        (slice(50, 150), slice(1600, 1700)),
        (slice(50, 150), slice(2700, 2800)),
    ]

    #############################################################################################
    #   RGB RGB RGB RGB RGB RGB RGB RGB RGB
    #############################################################################################
    diff = skimage.img_as_float(image.img) - skimage.img_as_float(baseline.img)
    # Regularize
    smooth = skimage.restoration.denoise_tv_bregman(
        diff, weight=0.025, eps=1e-4, max_num_iter=100, isotropic=True
    )
    print("RGB:")
    n, colours = extract_support_points(smooth, pats)
    plt.show()

    #############################################################################################
    #   LAB LAB LAB LAB LAB LAB LAB LAB LAB
    #############################################################################################
    diff = (
        (skimage.color.rgb2lab(image.img) - skimage.color.rgb2lab(baseline.img))
        + [0, 128, 128]
    ) / [100, 255, 255]
    smooth = skimage.restoration.denoise_tv_bregman(
        diff, weight=0.025, eps=1e-4, max_num_iter=100, isotropic=True
    )
    print("LAB:")
    n, colours = extract_support_points(smooth, pats)

    plt.show()
