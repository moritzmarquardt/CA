import darsia as da
import json
import time

# file to make config file
# based on the correction walkthrough jupyter notebook

start = time.time()

config: dict = {}

baseline_path = "./data/test/tracer_t0_original.jpg"
baseline = da.Image(baseline_path, width=0.92, height=0.555, color_space="RGB")
baseline.plt_show()

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
drift_correction = da.DriftCorrection(base=baseline, roi=roi_color_checker)
config["drift"] = drift_correction.config

# CURVATURE CORRECTION
curv_correction = da.CurvatureCorrection(
    image=baseline.imgpath, width=baseline.width, height=baseline.height
)

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
baseline = da.Image(
    "./data/test/tracer_t0_original.jpg",
    color_correction=color_correction,
    curvature_correction=curv_correction,
    width=0.92,
    height=0.555,
)
baseline.show()

# write config
config_path = "./data/test/config.json"
with open(config_path, "w") as outfile:
    json.dump(config, outfile, indent=4)
