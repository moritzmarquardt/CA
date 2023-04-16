"""
merge signals together and experiment and anlyse
"""
import darsia as da
from tailoredClasses import tTracerAnalysis
import json
from pathlib import Path

print("hi")

basis_path = "./data/tracer_timeseries/"
baseline_path = basis_path + "images/20220914-142404.TIF"
tracer_path = basis_path + "images/20220914-150357.TIF"


"""
the config file has to contain
- the physical asset dimensions
- curvature correction (here only crop is enough)
- use cache on or off
"""
with open(basis_path + "config.json") as json_file:
    config = json.load(json_file)


######################################
# build tailored signal reductions

# config taken from the config file
tracer_config_blue = {
    "color": "hsv",
    "hue lower bound": 0 / 360,
    "hue upper bound": 36 / 360,
    "saturation lower bound": 0,
    "saturation upper bound": 1,
}


tracer_config_green = {
    "color": "hsv",
    "hue lower bound": 310 / 360,
    "hue upper bound": 360 / 360,
    "saturation lower bound": 0,
    "saturation upper bound": 1,
}


signal_reduction_green = da.MonochromaticReduction(**tracer_config_green)
signal_reduction_blue = da.MonochromaticReduction(**tracer_config_blue)


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
print("TracerAnalysis building ...")

analysis_blue = tTracerAnalysis(
    config=Path(basis_path + "config.json"),
    baseline=[baseline_path],
    results=Path(basis_path + "results/"),
    update_setup=False,  # chache nicht nutzen und neu schreiben
    verbosity=0,  # 3 bedeutet, nur cut wird inspiziert
    roi=(slice(2550, 2600), slice(0, 5000)),
    signal_reduction=[signal_reduction_blue, signal_reduction_green],
    model=model,
)

print("TracerAnalysis build successfully")


# run a single image analysis on the test_img
print("apply the TracerAnalysis to test tracer image")
test = analysis_blue.single_image_analysis(tracer_path)
print("TracerAnalysis ran successful, show result")
test.show("test result")
