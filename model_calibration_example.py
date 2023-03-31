"""
Use the calibration from color_calibration.py and check whether the
two test runs of the optimal control experiment in the PoroTwin1
project indeed show expected results.

This test considers the third test run on 26.9.22.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../.")
from fluidflower import PoroTwin1Rig

folder = Path(
    "/media/jakub/Elements/Jakub/benchmark/data/porotwin/optimal_control_3rd_test_run"
)
original = Path("original")
baseline = Path("baseline")
processed = Path("processed-development")

# Initialize rig with a baseline image
baseline_images = list(sorted((folder / baseline).glob("*")))

# Define the digital rig
ff = PoroTwin1Rig(
    baseline_images,
    config_source="./config.json",
)

# Fetch images of injection
images = list(sorted((folder / original).glob("*")))

# Safety check
assert len(baseline_images) > 0
assert len(images) > 0

# Calibrate signal-data conversion model
apply_calibration = ff.config["tracer"].get("model calibration", False)
if apply_calibration:
    # Define calibration images - pick 10 images here.
    random_calibration_indices = np.unique((np.random.rand(10) * 140).astype(np.int32))
    calibration_images = [images[i] for i in random_calibration_indices]
    # Define injection rate - expert knowledge
    injection_rate = 500
    # Define geometry of the rig
    geometry = darsia.ExtrudedPorousGeometry(
        depth=self.config["physical_asset"]["dimensions"]["depth"],
        porosity=self.config["physical_asset"]["porosity"],
        **shape_metadata
    )
    # Apply calibration
    ff.tracer_analysis.calibrate_model(
        images,
        options={
            "model_position": 0,
            "geometry": geometry,
            "injection_rate": injection_rate,
            "initial_guess": [1.0, 0.0],
            "tol": 1e-1,
            "maxiter": 100,
        },
    )

# Track evolution of injected volume - initialize containers
times = []
volumes = []

# Loop over all images and determine the total volume and timestamp
random_indices = np.unique((np.random.rand(20) * len(images)).astype(np.int32))
random_images = [images[i] for i in random_indices]

for count, img in enumerate(random_images):

    # Determine concentration
    ff.load_and_process_image(img)
    concentration, volume = ff.determine_tracer(return_volume=True)

    # Track time (in hours)
    SECONDS_TO_HOURS = 1.0 / 3600.0
    times.append(ff.img.time * SECONDS_TO_HOURS)

    # Track total injected volume (in ML)
    M3_TO_ML = 1e6
    volumes.append(volume * M3_TO_ML)

    # Plot result
    if False:
        concentration.show(f"concentration {count}", 3)

    # Store concentration map
    if False:
        concentration.write(str(Path(folder / processed / img.stem)) + ".jpg")
        concentration.write_array(
            str(Path(folder / processed / img.stem)) + ".npy", indexing="Cartesian"
        )

# Double check the injection
plt.plot(times, volumes)
plt.axline((times[0], volumes[0]), slope=500, color="black", linestyle=(0, (5, 5)))
plt.show()
