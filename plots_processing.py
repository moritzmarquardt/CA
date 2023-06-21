import matplotlib.pyplot as plt
import skimage
import numpy as np
from model_experiment import model_experiment_full

baseline, image = model_experiment_full()

# correcting
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
