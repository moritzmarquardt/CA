"""
running the tracer analysis
(inspired by darsia_demonstration/segmentation.py)
"""
import darsia as da
from tailoredClasses import tTracerAnalysis
import json
from pathlib import Path

print("hi")

basis_path = "./data/tracer_timeseries/"
baseline_path = basis_path + "images/20220914-142404.TIF"
tracer_path = basis_path + "images/20220914-150357.TIF"


# use an existing config file to control the analysis
with open(basis_path + "config.json") as json_file:
    config = json.load(json_file)

curvature_correction = da.CurvatureCorrection(config["curvature"])
width = config["physical_asset"]["dimensions"]["width"]
height = config["physical_asset"]["dimensions"]["height"]

# Initialize the baseline image (one is enough for this purpose)
baseline = da.imread(
    path=tracer_path,
    width=width,
    height=height,
    # color_space="RGB",
    transformations=[curvature_correction],
)
baseline.show("test")

######################################
# build tailored signal reduction

# config taken from the config file
tracer_config = {
    "color": "hsv",  # define colorspace where the reducction is based on
    "hue lower bound": 0.055,  # lower treshold for tracer detection based on hue value
    "hue upper bound": 0.1,
    "saturation lower bound": 0.8,
    "saturation upper bound": 1,
}

signal_reduction = da.MonochromaticReduction(**tracer_config)
# und auch verwendet wird


########################################
# build tailored model
model_config = {
    "model scaling": 1.0,
    "model offset": 0.0,
}

# Linear model for converting signals to data
model = da.CombinedModel(
    [
        da.LinearModel(key="model ", **model_config),
        da.ClipModel(**{"min value": 0.0, "max value": 1.0}),
    ]
)

#########################################
# build the tailored tracer analysis class with
analysis = tTracerAnalysis(
    config=Path(basis_path + "config.json"),
    baseline=[baseline_path],
    results=Path(basis_path + "results/"),
    update_setup=False,  # chache nicht nutzen und neu schreiben
    verbosity=3,
    # inspect_diff_roi=(slice(2800, 3000), slice(3200, 3400)),
    signal_reduction=signal_reduction,
    model=model,
)
print("tracer analysis build successfully")

# run a single image analysis on the test_img
test = analysis.single_image_analysis(tracer_path)
# test.show()

tracer_paths = [
    Path(basis_path + "images/20220914-142627.TIF"),
    Path(basis_path + "images/20220914-142657.TIF"),
    Path(basis_path + "images/20220914-142727.TIF"),
    Path(basis_path + "images/20220914-142757.TIF"),
    Path(basis_path + "images/20220914-142827.TIF"),
]
print("processing images ...")
calibration_images = [analysis._read(path) for path in tracer_paths]

shape_metadata = baseline.shape_metadata()
geometry = da.ExtrudedPorousGeometry(
    depth=config["physical_asset"]["dimensions"]["depth"],
    porosity=config["physical_asset"]["porosity"],
    **shape_metadata
)
options = {
    "model_position": 0,  # welches model soll calibriert werden
    "geometry": geometry,
    "injection_rate": 500,
    "initial_guess": [1.0, 0.0],
    "tol": 1e-1,
    "maxiter": 100,
}
print("calibrating ...")
calibration = analysis.tracer_analysis.calibrate_model(calibration_images, options)
print(calibration)
