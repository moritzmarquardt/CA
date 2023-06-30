import numpy as np
import skimage
import matplotlib.pyplot as plt
from extract_support_points import extract_support_points
from model_experiment import model_experiment

baseline, image = model_experiment()

# define support colours based on patches in the image that are representative for that colour
pats = [
    (slice(50, 150), slice(100, 200)),
    # (slice(50, 150), slice(400, 500)),
    # (slice(50, 150), slice(600, 700)),
    # (slice(50, 150), slice(800, 900)),
    # (slice(50, 150), slice(1000, 1100)),
    # (slice(50, 150), slice(1200, 1300)),
    # (slice(50, 150), slice(1400, 1500)),
    (slice(50, 150), slice(1600, 1700)),
    (slice(50, 150), slice(2700, 2800)),
]


#############################################################################################
#   RGB RGB RGB RGB RGB RGB RGB RGB RGB
#############################################################################################
####################################
# nn interpol
###################################
diff = skimage.img_as_float(image.img) - skimage.img_as_float(baseline.img)
# Regularize
smooth = skimage.restoration.denoise_tv_bregman(
    diff, weight=0.025, eps=1e-4, max_num_iter=100, isotropic=True
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


# plt.figure("nn interpol rgb")
# plt.imshow(closest_color_RGB(smooth))
# plt.xlabel("horizontal pixel")
# plt.ylabel("vertical pixel")

plt.figure("cut ph val")
plt.plot(np.average(closest_color_RGB(smooth), axis=0))
plt.xlabel("horizontal pixel")
plt.ylabel("concentration")

###########################################################################################
#   LAB LAB LAB LAB LAB LAB LAB LAB LAB
#############################################################################################
####################################
# nn interpol
###################################
diff = (
    (skimage.color.rgb2lab(image.img) - skimage.color.rgb2lab(baseline.img))
    + [0, 128, 128]
) / [100, 255, 255]
smooth = skimage.restoration.denoise_tv_bregman(
    diff, weight=0.025, eps=1e-4, max_num_iter=100, isotropic=True
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
plt.xlabel("horizontal pixel")
plt.ylabel("vertical pixel")

plt.show()
