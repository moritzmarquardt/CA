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

# use an existing config file to control the analysis
with open(basis_path + "config.json") as json_file:
    config = json.load(json_file)

curvature_correction = da.CurvatureCorrection(config["curvature"])
width = config["physical asset"]["dimensions"]["width"]
height = config["physical asset"]["dimensions"]["height"]

# Initialize the baseline image (one is enough for this purpose)
# the width and height of the physical picture is fetched from the curvatrue config
# if there are no explicit dimensions given
baseline = da.Image(
    img=basis_path + "images/20220914-142404.TIF",
    width=width,
    height=height,
    color_space="RGB",
    curvature_correction=curvature_correction,
)
# baseline.plt_show()

# initialize the text image with a tracer present
test_img = da.Image(
    img=basis_path + "images/20220914-150357.TIF",
    width=width,
    height=height,
    color_space="RGB",
    curvature_correction=curvature_correction,
)
# test_img.plt_show()

# define a hsv signal reduction from the config file
signal_reduction = da.MonochromaticReduction(color="hsv", kwargs=config["tracer"])
print("signal reduction build successfully")

# build the tailored tracer analysis class with
# the config file, baseline image and a results folder
analysis = tTracerAnalysis(
    config=Path(basis_path + "config.json"),
    baseline=baseline.img,
    results=Path(basis_path + "results/"),
)
print("tracer analysis build successfully")

test = analysis.single_image_analysis(test_img)
test.plt_show()
