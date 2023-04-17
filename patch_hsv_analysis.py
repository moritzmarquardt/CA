"""
script to analyze a patch of the picture to abtain a single signal reduction

it can be used to see the overwritten routine _inspect_diff in
the tailoredConcentrationAnalysis to inspect the diff and
analyse the colors visible in the given patch (of the tracer)

the goal is to define a routine that allows for simple colour
analysis of a patch to in the end find a good hue threshold
or onother colour space for a good signal reduction.
"""
import darsia as da
from tailoredClasses import tTracerAnalysis
import json
from pathlib import Path

print("hi")

basis_path = "./data/tracer_timeseries/"
# picture with no tracer present
baseline_path = basis_path + "images/20220914-142404.TIF"
# picture with tracer present; ideally with big concentration
# gradients to allow for more detailed analysis
tracer_path = basis_path + "images/20220914-150357.TIF"


"""
the config file has to contain
- the physical asset dimensions
- curvature correction (here only crop is enough)
- use cache on or off
"""
with open(basis_path + "config.json") as json_file:
    config = json.load(json_file)

# build tailored signal reduction

# modify the tracer analysis based on the hsv plots of the patch analysis
tracer_config = {
    "color": "hsv",
    "hue lower bound": 340 / 360,
    "hue upper bound": 36 / 360,
    "saturation lower bound": 0.4,
    "saturation upper bound": 1,
}
signal_reduction = da.MonochromaticReduction(**tracer_config)

# build tailored model for signal to concentration translation
model_config = {
    "model scaling": 1.0,
    "model offset": 0.0,
}
model = da.CombinedModel(
    [
        da.LinearModel(key="model ", **model_config),
        da.ClipModel(**{"min value": 0.0, "max value": 1.0}),
    ]
)

# build the tailored tracer analysis class with
analysis = tTracerAnalysis(
    config=Path(basis_path + "config.json"),
    baseline=[baseline_path],
    results=Path(basis_path + "results/"),
    update_setup=False,  # chache nicht nutzen und neu schreiben
    verbosity=0,  # overwrites the config file
    # roi=(slice(2800, 3300), slice(2000, 2500)),
    # if patch is defined, it triggers the patch analysis plots
    # those are hsv analysis of the patch and
    # a full sized picture of the image to select the roi
    patch=(slice(2800, 3300), slice(2000, 2500)),
    # slices: first is the pixel range from top to bottom, second from left to right
    # first y slice then x slice ?WARUM?
    signal_reduction=[signal_reduction],
    model=model,
)

# STEP 1: select here the roi for the colour analysis
analysis._read(tracer_path).show("choose roi in tracer image")

# STEP 2: run a single image analysis on the test_img to get hsv plots
# here you select the hue and saturation thresholds for the signal reduction
test = analysis.single_image_analysis(tracer_path)

# STEP 3: plot the resulting signal
# this step is used to verify the selection of thresholds
test.show()
