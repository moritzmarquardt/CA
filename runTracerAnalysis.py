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
width = config["physical asset"]["dimensions"]["width"]
height = config["physical asset"]["dimensions"]["height"]

# Initialize the baseline image (one is enough for this purpose)
# the width and height of the physical picture is fetched from the curvatrue config
# if there are no explicit dimensions given
# this is used to find the sliceds for the patch inspection below
baseline = da.Image(
    img=tracer_path,
    width=width,
    height=height,
    color_space="RGB",
    curvature_correction=curvature_correction,
)
baseline.plt_show()

######################################
# build tailored signal reduction

# tracer_config = config["tracer"]
# print(tracer_config)
# config taken from the config file
tracer_config = {
    "cleaning_filter": "cache/cleaning_filter.npy",
    "color": "hsv",  # define colorspace where the reducction is based on
    "hue lower bound": 0.055,  # lower treshold for tracer detection based on hue value
    "hue upper bound": 0.1,
    "saturation lower bound": 0.8,
    "saturation upper bound": 1,
    "restoration resize": 0.25,
    "restoration smoothing method": "isotropic bregman",
    "restoration smoothing weight": 0.025,
    "restoration smoothing eps": 0.0001,
    "restoration smoothing max_num_iter": 100,
    "model scaling": 1.0,
    "model offset": 0.0,
    "verbosity": 0,
}

signal_reduction = da.MonochromaticReduction(**tracer_config)
# ? wie gehe ich vor, dass ich ne monocchromatische signal reduction aufbauen kann
# und auch verwendet wird
# ? Normierung
# ? Wie binde ich das in ein Modell ein?
# ? Wie wird das modell kalibriert?

#########################################
# build the tailored tracer analysis class with
# the config file, baseline image and a results folder
analysis = tTracerAnalysis(
    config=Path(basis_path + "config.json"),
    baseline=[baseline_path],
    results=Path(basis_path + "results/"),
    update_setup=False,  # chache nicht nutzen und neu schreiben
    # verbosity=0, # wird schon in config gesetzt
    inspect_diff_roi=(slice(2800, 3000), slice(3200, 3400)),
    # inspect_diff_roi=(slice(2650, 2950), slice(3100, 3500)),
    signal_reduction=signal_reduction,
)
print("tracer analysis build successfully")

# run a single image analysis on the test_img
test = analysis.single_image_analysis(tracer_path)
test.plt_show()
