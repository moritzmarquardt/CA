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
model = da.LinearModel(scaling=1, offset=0)

#########################################
analysis = tTracerAnalysis(
    config=Path(basis_path + "config.json"),
    baseline=[baseline_path],
    results=Path(basis_path + "results/"),
    update_setup=False,  # chache nicht nutzen und neu schreiben
    verbosity=0,  # 3 bedeutet, nur cut wird inspiziert
    signal_reduction=[signal_reduction_blue, signal_reduction_green],
    model=model,
)


baseline = da.imread(
    path=tracer_path,
    width=config["physical_asset"]["dimensions"]["width"],
    height=config["physical_asset"]["dimensions"]["height"],
    transformations=[da.CurvatureCorrection(config["curvature"])],
)
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
# calibrate model uses the objective function of the injection rate mixin of analysis
print(calibration)
