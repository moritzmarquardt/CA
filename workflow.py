import darsia as da
import numpy as np
import json

config: dict = {}

baseline = da.Image(
    "./data/test/DSC28345_t0.JPG", 
    width=0.92, 
    height=0.555
    )
# baseline.show()

#Preprocessing of the images:
roi_color_checker = [
        [6717, 57],
        [6722, 537],
        [7447, 536],
        [7438, 54],
    ]

#COLOR CORECTION
color_correction = da.ColorCorrection(
    roi=roi_color_checker, 
    verbosity=False
    )
config["color"] = color_correction.config

#DRIFT CORRECTION
drift_correction = da.DriftCorrection(
    base=baseline,
    roi=roi_color_checker
    )
config["drift"] = drift_correction.config

#CURVATURE CORRECTION
curv_correction = da.CurvatureCorrection(
    image = baseline.imgpath,
    width = baseline.width,
    height = baseline.height
    )

curv_correction.pre_bulge_correction(horizontal_bulge = 5e-9)
# curv_correction.show_image()

curv_correction.crop([
        [261, 57],
        [240, 4438],
        [7625, 4439],
        [7585, 57],
    ])

# print(curv_correction.config)
# curv_correction.show_image()

curv_correction.bulge_corection(
    left = 78, 
    right = 73, 
    top = 0, 
    bottom = 0
    )
# curv_correction.show_image()

config["curvature"] = curv_correction.config
# config_path = "./data/test/config.json"
# with open(config_path, "w") as outfile:
#     json.dump(config, outfile, indent=4)


#Apply on another image with same camera setup with no drift correction, whats that?
co2_image = da.Image(
    "./data/test/DSC28537.JPG",
    color_correction=color_correction,
    curvature_correction=curv_correction
)
# co2_image.show()

baseline_co2 = da.Image(
    "./data/test/DSC28345_t0.JPG",
    color_correction=color_correction,
    curvature_correction=curv_correction
)
# baseline_co2.show()


#CONCENTRATION ANALYSIS
# Construct concentration analysis for detecting the co2 concentration
co2_analysis = da.ConcentrationAnalysis(
    baseline_co2,  # baseline image
    da.MonochromaticReduction(color="gray"),  # signal reduction
    None,  # signal balancing
    da.TVD(),  # restoration
    da.CombinedModel(  # signal to data conversion
        [
            da.LinearModel(scaling=4.0),
            da.ClipModel(**{"min value": 0.0, "max value": 1.0}),
        ]
    ),
)

# Determine co2
co2 = co2_analysis(co2_image)
co2.plt_show()