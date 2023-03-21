import darsia as da
import numpy as np
import json
import time

#1. step: make config file for corrections and preprocessing
#use correction.py
print("hi")

#open config from file
with open('./data/test/config.json') as json_file:
    config = json.load(json_file)
curv_correction = da.CurvatureCorrection(config = config["curvature"])
color_correction = da.ColorCorrection(config = config["color"])


#Apply on another image with same camera setup with no drift correction, whats that?
co2_image = da.Image(
    "./data/test/tracer_original.jpg",
    color_correction=color_correction,
    curvature_correction=curv_correction
)
# co2_image.show()

baseline_co2 = da.Image(
    "./data/test/tracer_t0_original.jpg",
    color_correction=color_correction,
    curvature_correction=curv_correction
)
# baseline_co2.show()

'''reduction = da.MonochromaticReduction(color="red")
reduced = reduction(co2_image.img)

reduced_d = da.Image(
    reduced, 
    width=0.92, 
    height=0.555,
    color_space = "RGB"
)
reduced_d.show()'''

#CONCENTRATION ANALYSIS
# Construct concentration analysis for detecting the co2 concentration
co2_analysis = da.ConcentrationAnalysis(
    baseline_co2,  # baseline image
    da.MonochromaticReduction(color="red"),  # signal reduction
    None,  # signal balancing
    da.TVD(),  # restoration
    da.CombinedModel(  # signal to data conversion
        [
            da.LinearModel(scaling=4.0),
            da.ClipModel(**{"min value": 0.0, "max value": 4.0}),
        ]
    ),
)

# Determine co2
co2 = co2_analysis(co2_image)
co2.plt_show()

#tracer analysis is analogue