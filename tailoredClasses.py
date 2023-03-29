"""
inspired by darsia/presets/fluidflowertraceranalysis

"""
from pathlib import Path
from typing import Union, Optional
import matplotlib.pyplot as plt
import cv2


import numpy as np

import darsia


class tTracerAnalysis(darsia.TracerAnalysis):
    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config: Union[str, Path],
        results: Union[str, Path],
        update_setup: bool = False,
        verbosity: int = 0,
        **kwargs,
    ) -> None:
        """
        Setup of analysis.

        Args:
            baseline (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config (str or Path): path to config dict
            results (str or Path): path to results directory
            update_setup (bool): flag controlling whether cache in setup
                routines is emptied.
            verbosity  (bool): flag controlling whether results of the post-analysis
                are printed to screen; default is False.
        """
        # Assign tracer analysis
        print("hii from init tTracerAnalysis")
        self.roi = kwargs.get("roi")  # vor innit, weil
        darsia.TracerAnalysis.__init__(self, baseline, config, update_setup)
        # Traceranalysis has

        # Create folder for results if not existent
        self.path_to_results: Path = Path(results)
        self.path_to_results.parents[0].mkdir(parents=True, exist_ok=True)

        # Store verbosity
        self.verbosity = verbosity
        print("init of tTracerAnalysis successful")

    # ! ---- Analysis tools for detecting the tracer concentration

    def define_tracer_analysis(self) -> darsia.ConcentrationAnalysis:
        # Define signal reduction
        signal_reduction = darsia.MonochromaticReduction(**self.config["tracer"])

        # Define restoration object - coarsen, tvd, resize
        original_size = self.base.img.shape[:2]
        restoration = darsia.CombinedModel(
            [
                darsia.Resize(key="restoration ", **self.config["tracer"]),
                darsia.TVD(key="restoration ", **self.config["tracer"]),
                darsia.Resize(dsize=tuple(reversed(original_size))),
            ]
        )

        # Linear model for converting signals to data
        model = darsia.CombinedModel(
            [
                darsia.LinearModel(key="model ", **self.config["tracer"]),
                darsia.ClipModel(**{"min value": 0.0, "max value": 1.0}),
            ]
        )

        ###################################################################
        # Final concentration analysis with possibility for calibration
        # of both the balancing and the model

        verbosity = self.config["tracer"].get("verbosity", 0)

        tracer_analysis = tConcentrationAnalysis(
            self.base,
            signal_reduction,
            None,
            restoration,
            model,
            # self.labels,
            verbosity=verbosity,
            roi=self.roi,
        )
        print("hi from define tracer analysis finished with" + str(tracer_analysis))

        return tracer_analysis

    # ! ---- Calibration routines

    def calibrate_model(self, calibration_images: list[Path], options: dict) -> None:
        """
        Calibration routine aiming at matching the injection rate

        NOTE: Calling this routine will require the definition of
        a geometry for data integration.

        Args:
            calibration_images (list of Path): calibration images.
            options (dict): parameters for calibration.

        """
        # Read and process the images
        print("Calibration: Processing images...")
        images = [self._read(path) for path in calibration_images]

        # Calibrate the overall signal via a simple constant rescaling
        print("Calibration: Model...")
        self.tracer_analysis.calibrate_model(
            images,
            options=dict(options, **{"model position": 0, "geometry": self.geometry}),
        )

    # ! ----- Analysis tools

    def single_image_analysis(self, img: Path, **kwargs) -> darsia.Image:
        """
        Standard workflow to analyze the tracer concentration.

        Args:
            image (Path): path to single image.
            kwargs: optional keyword arguments, see batch_analysis.

        Returns:
            np.ndarray: tracer concentration map
            dict: dictinary with all stored results from the post-analysis.
        """

        # ! ---- Extract concentration profile

        # Load the current image
        self.load_and_process_image(img)

        # Determine tracer concentration
        tracer = self.determine_tracer()

        return tracer


class tConcentrationAnalysis(darsia.ConcentrationAnalysis):
    def __init__(
        self,
        base: Union[darsia.Image, list[darsia.Image]],
        signal_reduction: darsia.SignalReduction,
        balancing: Optional[darsia.Model] = None,
        restoration: Optional[darsia.TVD] = None,
        model: darsia.Model = darsia.Identity,
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            base, signal_reduction, balancing, restoration, model, labels, **kwargs
        )
        self.roi = kwargs.get("roi", None)

    def _inspect_diff(self, img: np.ndarray, bins: int = 100) -> None:
        """
        Routine allowing for plotting of intermediate results.
        Requires overwrite.

        Args:
            img (np.ndarray): image
        """
        print("overwritten inspect routine")
        roi = self.roi
        if self.verbosity >= 2:
            plt.figure("difference test-baseline")
            plt.imshow(img)

            if isinstance(roi, tuple):
                roi = [roi]

            for i, r in enumerate(roi):
                # Retrict to ROI
                img_roi = img[r]

                # Extract H, S, V components

                hsv = cv2.cvtColor(img_roi.astype(np.float32), cv2.COLOR_RGB2HSV)
                h_img = hsv[:, :, 0]
                s_img = hsv[:, :, 1]
                v_img = hsv[:, :, 2]

                # Extract values
                h_values = np.linspace(np.min(h_img), np.max(h_img), bins)
                s_values = np.linspace(np.min(s_img), np.max(s_img), bins)
                v_values = np.linspace(np.min(v_img), np.max(v_img), bins)

                # Setup histograms
                h_hist = np.histogram(h_img, bins=bins)[0]
                s_hist = np.histogram(s_img, bins=bins)[0]
                v_hist = np.histogram(v_img, bins=bins)[0]

                # Plot
                plt.figure(f"h {i}")
                plt.plot(h_values, h_hist)
                plt.figure(f"s {i}")
                plt.plot(s_values, s_hist)
                plt.figure(f"v {i}")
                plt.plot(v_values, v_hist)

            plt.show()
