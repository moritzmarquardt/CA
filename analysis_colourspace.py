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
    voxels=(slice(1400, 2600), slice(2500, 7000))
)
image = darsia.imread(image_path, transformations=transformations).subregion(
    voxels=(slice(1400, 2600), slice(2500, 7000))
)

diff = skimage.img_as_float(baseline.img) - skimage.img_as_float(image.img)
# plt.figure("diff")
# plt.imshow(diff)

# plt.figure("img")
# plt.imshow(skimage.img_as_ubyte(image.img))

diff_neg = diff < 0
# diff[diff_neg] = 0


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
# plt.figure("rgb")
# plt.plot(smooth_r, color="red")
# plt.plot(smooth_g, color="green")
# plt.plot(smooth_b, color="blue")


green_rgb = np.array([0.3125, 0.1647, 0.1913])  # equals concentration 1
blue_rgb = np.array([0.6693, 0.3575, -0.55])  # consider negative values for blue!
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
        # s = 50
        # self.c = skimage.color.rgb2lab(c)
        # self.v = v
        # self.fine_c = [
        #     np.linspace(self.c[0][0], self.c[1][0], s).tolist(),
        #     np.linspace(self.c[0][1], self.c[1][1], s).tolist(),
        #     np.linspace(self.c[0][2], self.c[1][2], s).tolist(),
        # ]
        # self.fine_c = np.transpose(self.fine_c).reshape((1, s, 3))
        # print(self.fine_c.shape)
        # self.fine_v = np.linspace(0, 1, s)
        # plt.figure("ph stripe")
        # plt.imshow(skimage.color.lab2rgb(self.fine_c))
        # plt.show()
        # self.c = self.fine_c
        # self.v = self.fine_v
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
        identifier = self._closest_color_HSV(signal)
        # TODO relate to self.v and identify representative pH value
        # linear model:

        return identifier

    def color_to_ph_KERNEL(self, signal: np.ndarray) -> np.ndarray:
        # signal is rgb
        signal = skimage.color.rgb2lab(signal)
        self.c = skimage.color.rgb2lab(self.c)
        k = lambda x, y: np.inner(x, y) + 0.1
        x = np.array(self.c)
        print(x, x.shape)
        y = np.array(self.v)
        print(y, y.shape)
        # X = x.transpose() @ x
        X = np.ones(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                X[i, j] = k(x[i], x[j])
        print(x[1])
        print(np.inner(x[1], x[1]))
        print(X, X.shape)
        alpha = np.linalg.solve(X, y)
        ph = lambda y: np.sum(alpha * k(y, x))
        print(ph(self.c[1]))
        ph_image = np.zeros(signal.shape[:2])
        for i in range(signal.shape[0]):
            for j in range(signal.shape[1]):
                ph_image[i, j] = ph(signal[i, j, :])

        return ph_image

    def _closest_color_LAB(self, signal: np.ndarray) -> np.ndarray:
        # translate signal and ph sample to the colour space
        signal = skimage.color.rgb2lab(signal)
        self.c = skimage.color.rgb2lab(self.c)

        signal_shape = signal.shape[:2]  # picture size e.g. (200, 4500)
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

    def _closest_color_RGB(self, signal: np.ndarray) -> np.ndarray:
        """Auxiliary routine. Determine local index identifying the closest color in
        provided color palette for each pixel.

        Args:
            signal (np.ndarray): RGB image

        Returns:
            np.ndarray: array with index of closest color for each pixel

        """
        # TODO investigate different color spaces by tranforming rgb to hsv, lab, rgbcie, luv
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

    def _closest_color_HSV(self, signal: np.ndarray) -> np.ndarray:
        """Auxiliary routine. Determine local index identifying the closest color in
        provided color palette for each pixel.

        Args:
            signal (np.ndarray): RGB image

        Returns:
            np.ndarray: array with index of closest color for each pixel

        """
        # TODO investigate different color spaces by tranforming rgb to hsv, lab, rgbcie, luv
        # translate signal and ph sample to the colour space
        signal = skimage.color.rgb2hsv(signal)
        self.c = skimage.color.rgb2hsv(self.c)
        print(self.c)

        signal_shape = signal.shape[:2]  # picture size e.g. (200, 4500)
        print(signal.shape)
        distance = np.zeros(
            (self.num, *signal_shape), dtype=float
        )  # *signal_shape = signal_shape[0], signal_shape[1]
        print(distance.shape)
        for x in signal_shape[0]:
            for y in signal_shape[1]:
                for i in range(self.num):
                    dh = (
                        np.min(
                            abs(self.c[i][0] - signal[x, y, 0]),
                            360 - abs(self.c[i][0] - signal[x, y, 0]),
                        )
                        / 180.0
                    )
                    ds = abs(self.c[i][1] - signal[x, y, 1])
                    dv = abs(self.c[i][2] - signal[x, y, 2]) / 255.0
                    distance[x, y, i] = np.sqrt(dh * dh + ds * ds + dv * dv)

        identifier = np.argmin(distance, axis=0)

        return identifier


# Convert a discrete ph stripe to a numeric pH indicator.
pwc = PHIndicator([black_rgb, blue_rgb, green_rgb], [0, concentration_blue, 1])
plt.figure("ph identifier")
plt.imshow(pwc.color_to_ph_KERNEL(smooth))
plt.show()
