import numpy as np
import darsia
import skimage
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndi
import scipy.optimize as spo

folder = Path("./data/tracer_timeseries/images")
baseline_path = folder / Path("20220914-142404.TIF")
image_path = folder / Path("20220914-151727.TIF")

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
# baseline = darsia.imread(baseline_path, transformations = transformations)
# image = darsia.imread(image_path, transformations = transformations)
baseline = darsia.imread(baseline_path, transformations=transformations).subregion(
    voxels=(slice(2400, 2600), slice(2500, 7000))
)
image = darsia.imread(image_path, transformations=transformations).subregion(
    voxels=(slice(2400, 2600), slice(2500, 7000))
)

# Take difference
plt.figure("img")
plt.imshow(skimage.img_as_ubyte(image.img))

plt.figure("base")
plt.imshow(skimage.img_as_ubyte(baseline.img))

plt.figure("diff")
diff = skimage.img_as_float(baseline.img) - skimage.img_as_float(image.img)
plt.imshow(diff)

diff_neg = diff < 0
diff[diff_neg] = 0

# plt.figure("diff reverse")
# diff_reverse = (skimage.img_as_float(image.img) - skimage.img_as_float(baseline.img))
# plt.imshow(diff_reverse)

# Regularize
smooth = skimage.restoration.denoise_tv_bregman(
    diff, weight=0.1, eps=1e-4, max_num_iter=100, isotropic=True
)

plt.figure("smooth diff")
plt.imshow(smooth)

# Reduce to one line
r, g, b = cv2.split(smooth)
r_one_line = np.average(r, axis=0)
g_one_line = np.average(g, axis=0)
b_one_line = np.average(b, axis=0)

# Reduce to one line
r_diff, g_diff, b_diff = cv2.split(diff)
r_diff_one_line = np.average(r_diff, axis=0)
g_diff_one_line = np.average(g_diff, axis=0)
b_diff_one_line = np.average(b_diff, axis=0)

# Plot single components
plt.figure("rgb - diff")
plt.plot(r_diff_one_line, color="red")
plt.plot(g_diff_one_line, color="green")
plt.plot(b_diff_one_line, color="blue")

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

## HSV analysis
# hsv = cv2.cvtColor(smooth.astype(np.float32), cv2.COLOR_RGB2HSV)
# h, s, v = cv2.split(hsv)
# h_one_line = np.average(h, axis=0)
# s_one_line = np.average(s, axis=0)
# v_one_line = np.average(v, axis=0)
#
# smooth_h = ndi.gaussian_filter1d(h_one_line, sigma=sigma)
# smooth_s = ndi.gaussian_filter1d(s_one_line, sigma=sigma)
# smooth_v = ndi.gaussian_filter1d(v_one_line, sigma=sigma)

# plt.figure("hsv")
# plt.plot(smooth_h / 360, color = "red")
# plt.plot(smooth_s, color = "green")
# plt.plot(smooth_v, color = "blue")

# Extract characteristic HSV signals in the two regimes.
green_hsv = [0.455 * 360, 0.174, 0.834]
blue_hsv = [0.260 * 360, 0.840, 0.638]

# green_rgb = cv2.cvtColor(np.array(green_hsv).reshape((1,1,-1)).astype(np.float32), cv2.COLOR_HSV2RGB)
# blue_rgb = cv2.cvtColor(np.array(blue_hsv).reshape((1,1,-1)).astype(np.float32), cv2.COLOR_HSV2RGB)

green_rgb = np.array([0.3125, 0.1647, 0.1913])  # equals concentration 1
blue_rgb = np.array([0.6693, 0.3575, 0])  # TODO consider negative values for blue!
black_rgb = np.array([0, 0, 0])
concentration_blue = 0.5  # need expert knowledge


class PHIndicator:
    """Class implementing a continuous pH indicator, from a discrete pH stripe."""

    def __init__(self, c: list[np.ndarray], v: list[float]):
        """
        Args:
            c (list of np.ndarray): characteristic colors on the continuous
                pH stripe
            v (list of float): characteristic pH reference values on the
                continuous pH stripe.

        """
        # Define samples of the pH strip
        self.c = c
        self.v = v
        self.num = len(self.v)

    def ph_to_color(self, alpha: np.ndarray) -> np.ndarray:
        """Convert from scalar signal (pH value) to color.

        Args:
            alpha (np.ndarray): image with pH values.

        Returns:
            np.ndarray: corresponding color image

        """
        assert isinstance(alpha, np.ndarray)
        alpha_shape = alpha.shape
        i_array = np.zeros_like(alpha, dtype=int)
        vi_array = np.zeros_like(alpha, dtype=float)
        vip1_array = np.zeros_like(alpha, dtype=float)
        for i in range(1, self.num - 1):
            interval = self.v[i] > alpha
            i_array[interval] = i
            vi_array[interval] = self.v[i]
            vip1_array[interval] = self.v[i + 1]
        alpha_star = np.divide(vip1_array - alpha, vip1_array - vi_array)
        s = np.outer(np.ones(alpha_shape), self.c[i]).reshape(
            (*alpha_shape, 3)
        ) + np.outer(alpha_star, (self.c[i + 1] - self.c[i])).reshape((*alpha_shape, 3))
        return s

    def color_to_ph(self, signal: np.ndarray) -> np.ndarray:
        """Convert color image to pH values.

        Args:
            signal (np.ndarray): colored image

        Return:
            np.ndarry: scalar valued array containing identified pH values.

        """
        identifier = self._closest_color(signal)
        # TODO relate to self.v and identify representative pH value
        return identifier

    def _closest_color(self, signal: np.ndarray) -> np.ndarray:
        """Auxiliary routine. Determine local index identifying the closest color in
        provided color palette for each pixel.

        Args:
            signal (np.ndarray): RGB image

        Returns:
            np.ndarray: array with index of closest color for each pixel

        """
        # TODO investigate different color spaces by tranforming rgb to hsv, lab, rgbcie, luv
        # https://scikit-image.org/docs/stable/api/skimage.color.html
        signal_shape = signal.shape[:2]
        distance = np.zeros((self.num, *signal_shape), dtype=float)
        for i in range(self.num):
            mono_colored_image = np.outer(np.ones(signal_shape), self.c[i]).reshape(
                signal.shape
            )
            distance[i] = np.sqrt(
                np.sum(np.power(signal - mono_colored_image, 2), axis=2)
            )
        identifier = np.argmin(distance, axis=0)
        return identifier


# Convert a discrete ph stripe to a numeric pH indicator.
pwc = PHIndicator([black_rgb, blue_rgb, green_rgb], [0, concentration_blue, 1])
plt.figure("ph identifier")
plt.imshow(pwc.color_to_ph(smooth))
plt.show()

# Test
# alpha = np.ones((3,4))
# pwc_ = pwc(alpha)

# def pwc_concentration(signal):
#    shape = signal.shape[:2]
#    def fun(alpha):
#        return np.sum(np.power(pwc(alpha).reshape(signal.shape) - signal, 2))
#    alpha = spo.minimize(fun, x0 = np.ones(shape, dtype=float))
#    return np.clip(alpha, 0, 1), alpha<0, alpha>1
#
# plt.figure("pwc signal")
# pwc_signal = pwc_concentration(smooth)
# plt.imshow(pwc_signal)


def green_blue_concentration(signal):
    shape = signal.shape[:2]
    g = np.array(green_rgb)
    b = np.array(blue_rgb)
    normal_matrix = np.sum(np.multiply(g - b, g - b))
    normal_vector = np.outer(np.ones(shape), g - b).reshape((*shape, 3))
    rhs = signal - np.outer(np.ones(shape), b).reshape((*shape, 3))
    projection = np.sum(np.multiply(normal_vector, rhs), axis=2)
    alpha = projection / normal_matrix
    return np.clip(alpha, 0, 1), alpha < 0, alpha > 1


plt.figure("green blue scalar signal")
gb_c, gb_neg, gb_pos = green_blue_concentration(smooth)
plt.imshow(gb_c)
plt.figure("positive")
plt.imshow(gb_pos)


def blue_concentration(signal):
    shape = signal.shape[:2]
    b = np.array(blue_rgb)
    zero = np.array([0, 0, 0])
    normal_matrix = np.sum(np.multiply(b, b))
    normal_vector = np.outer(np.ones(shape), b).reshape((*shape, 3))
    rhs = signal
    projection = np.sum(np.multiply(normal_vector, rhs), axis=2)
    alpha = projection / normal_matrix
    return np.clip(alpha, 0, 1), alpha < 0, alpha > 1


plt.figure("blue scalar signal")
b_c, _, _ = blue_concentration(smooth)
plt.imshow(b_c)

plt.show()
