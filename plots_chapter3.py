import darsia as da
from pathlib import Path
import matplotlib.pyplot as plt

# import json
import time

# file to make config file
# based on the correction walkthrough jupyter notebook

config: dict = {}

baseline_path = Path("./data/tracer_timeseries/images/20220914-142404.TIF")
baseline = da.imread(baseline_path, width=0.92, height=0.555)
print(baseline.img)
plt.imshow(baseline.img)
plt.show()

# Preprocessing of the images:
roi_color_checker = [
    [6787, 77],
    [6793, 555],
    [7510, 555],
    [7502, 78],
]


# COLOR CORECTION
color_correction = da.ColorCorrection(
    roi=roi_color_checker,
    verbosity=False,
    # baseline=
)
config["color"] = color_correction.config


# DRIFT CORRECTION
# drift_correction = da.DriftCorrection(base=baseline, roi=roi_color_checker)
# config["drift"] = drift_correction.config

# CURVATURE CORRECTION
curv_correction = da.CurvatureCorrection(image=baseline.img, width=0.92, height=0.55)

curv_correction.pre_bulge_correction(horizontal_bulge=5e-9)
# curv_correction.show_image()

curv_correction.crop(
    [
        [322, 281],
        [298, 4427],
        [7640, 4414],
        [7600, 281],
    ]
)

# print(curv_correction.config)
# curv_correction.show_image()

curv_correction.bulge_corection(left=70, right=60, top=0, bottom=0)
# curv_correction.show_image()

config["curvature"] = curv_correction.config

# test
baseline = da.imread(
    "./data/test/tracer_t0_original.jpg",
    color_correction=color_correction,
    curvature_correction=[curv_correction],
    width=0.92,
    height=0.555,
)
baseline.show()

# write config
# config_path = "./data/test/config.json"
# with open(config_path, "w") as outfile:
#     json.dump(config, outfile, indent=4)
print(config)
