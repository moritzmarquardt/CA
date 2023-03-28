import darsia as da
import json

# Traceranalysis workflow (I.E. for blue tracer in multitracer images)
print("hi")

# open config from file
with open("./jakub/config.json") as json_file:
    config = json.load(json_file)

baseline = da.Image(
    img="./data/tracer_timeseries/20220914-142404.TIF",
    # width=config["physical asset"]["dimensions"]["width"],
    # height=config["physical asset"]["dimensions"]["height"],
    curvature_correction=da.CurvatureCorrection(config["curvature"]),
)
# baseline.plt_show()

test_img = da.Image(
    img="./data/tracer_timeseries/20220914-150357.TIF",
    curvature_correction=da.CurvatureCorrection(config["curvature"]),
)
# test_img.plt_show()

signal_reduction = da.MonochromaticReduction(color="hsv", kwargs=config["tracer"])

green_analysis = da.ConcentrationAnalysis(
    base=baseline,  # baseline image
    signal_reduction=signal_reduction,  # signal reduction
    balancing=None,  # signal balancing
    restoration=da.TVD(),  # restoration
    model=da.CombinedModel(  # signal to data conversion
        [
            da.LinearModel(scaling=4.0),
            da.ClipModel(**{"min value": 0.0, "max value": 1.0}),
        ]
    ),
    verbosity=3,
)

concentration = green_analysis(test_img)
concentration.plt_show()
