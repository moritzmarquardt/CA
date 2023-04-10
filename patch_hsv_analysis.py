"""
script to analyze a patch of the picture

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
import matplotlib.pyplot as plt
import skimage

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

if False:
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
    # show the picture to get coodinates of suiting patches / rois
    # baseline.show("test")
    fig = plt.figure("baseline image to select roi for colour analysis")
    plt.imshow(skimage.img_as_float(baseline.img))

    # use matplotlib to print coordinates of roi selected by double clicks
    def onclick(event):
        if event.dblclick:
            print(
                "double click: xdata=%f, ydata=%f"
                % (
                    event.xdata,
                    event.ydata,
                )
            )

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

######################################
# build tailored signal reduction

# config taken from the config file
tracer_config = {
    "color": "hsv",
    "hue lower bound": 340 / 360,
    "hue upper bound": 36 / 360,
    "saturation lower bound": 0.4,
    "saturation upper bound": 1,
}
signal_reduction = da.MonochromaticReduction(**tracer_config)

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
# this allows for actually analysing the patch / roi
print("TracerAnalysis building ...")
analysis = tTracerAnalysis(
    config=Path(basis_path + "config.json"),
    baseline=[baseline_path],
    results=Path(basis_path + "results/"),
    update_setup=False,  # chache nicht nutzen und neu schreiben
    verbosity=3,  # overwrites the config file
    inspect_diff_roi=(slice(2800, 3300), slice(2000, 2500)),
    # slices: first is the pixel range from top to bottom, second from left to right
    # first y slice then x slice ?WARUM?
    signal_reduction=signal_reduction,
    model=model,
)
print("TracerAnalysis build successfully")

if True:
    # run a single image analysis on the test_img
    print("apply the TracerAnalysis to test tracer image")
    test = analysis.single_image_analysis(tracer_path)
    print("TracerAnalysis ran successful, show result")
    test.show()
