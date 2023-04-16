"""
inspired by darsia/presets/fluidflowertraceranalysis

"""
from pathlib import Path
from typing import Union, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        roi: slice = None,
        signal_reduction: darsia.MonochromaticReduction() = None,
        model: darsia.Model() = None,
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
            roi (slice): one slice is lower right corner, the other is upper left
                für die inspect routine
                inspect routine ist in tCA überschrieben und wird nur aufgerufen, wenn
                roi is not None
        """
        # Assign tracer analysis
        print("hi from init tTracerAnalysis")
        self.roi = roi
        self.signal_reduction = signal_reduction
        self.model = model
        self.verbosity = verbosity
        darsia.TracerAnalysis.__init__(self, baseline, config, update_setup)
        # Traceranalysis has all thhe corrections for the image
        # read wrtes cleaning filters
        # calls the define tracer analysis and stores in self.tracer analysis

        # Create folder for results if not existent
        self.path_to_results: Path = Path(results)
        self.path_to_results.parents[0].mkdir(parents=True, exist_ok=True)

        # Store verbosity

        print("init of tTracerAnalysis successful")

    # ! ---- Analysis tools for detecting the tracer concentration

    def define_tracer_analysis(self) -> darsia.ConcentrationAnalysis:
        # Define signal reduction
        # signal_reduction = darsia.MonochromaticReduction(**self.config["tracer"])

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
        # model = darsia.CombinedModel(
        #     [
        #         darsia.LinearModel(key="model ", **self.config["tracer"]),
        #         darsia.ClipModel(**{"min value": 0.0, "max value": 1.0}),
        #     ]
        # )

        ###################################################################
        # Final concentration analysis with possibility for calibration
        # of both the balancing and the model

        # verbosity = self.config["tracer"].get("verbosity", 0)

        tracer_analysis = tConcentrationAnalysis(
            self.base,
            self.signal_reduction,
            None,
            restoration,
            self.model,
            # self.labels,
            verbosity=self.verbosity,
            roi=self.roi,
        )
        print("hi from define tracer analysis finished with" + str(tracer_analysis))

        return tracer_analysis

    # ! ---- Calibration routines

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


class tConcentrationAnalysis(
    darsia.ConcentrationAnalysis, darsia.InjectionRateModelObjectiveMixin
):
    def __init__(
        self,
        base: Union[darsia.Image, list[darsia.Image]],
        signal_reduction: darsia.SignalReduction,
        balancing: Optional[darsia.Model] = None,
        restoration: Optional[darsia.TVD] = None,
        model: darsia.Model = darsia.Identity,
        labels: Optional[np.ndarray] = None,
        roi: slice = None,
        **kwargs,
    ) -> None:
        super().__init__(
            base, signal_reduction, balancing, restoration, model, labels, **kwargs
        )
        self.roi = roi

    def _inspect_all_in_roi(
        self, diff, signal, clean_signal, balanced_signal, smooth_signal, concentration
    ):
        if self.verbosity >= 2:
            roi = self.roi
            plt.figure("propgation of the processing")
            plt.subplot(2, 3, 1)
            plt.imshow(diff[roi])
            plt.subplot(2, 3, 2)
            plt.imshow(signal[roi])
            plt.subplot(2, 3, 3)
            plt.imshow(clean_signal[roi])
            plt.subplot(2, 3, 4)
            plt.imshow(balanced_signal[roi])
            plt.subplot(2, 3, 5)
            plt.imshow(smooth_signal[roi])
            plt.subplot(2, 3, 6)
            plt.imshow(concentration[roi])

    def _inspect_diff(self, img: np.ndarray) -> None:
        """gets called after taking the difference of test-baseline

        Args:
            img (np.ndarray): difference

        """
        bins = 100
        if self.roi is not None:
            roi = self.roi
            img_roi = img[roi]
            height = roi[0].stop - roi[0].start
            width = roi[1].stop - roi[1].start
            # plt.figure("roi difference")
            # plt.imshow(img_roi)

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
            # fig, ax = plt.subplot(2, 2)
            fig = plt.figure("hsv values of the roi")
            ax = fig.add_subplot(2, 2, 1)
            ax.set_title("h")
            ax.plot(h_values, h_hist)
            ax = fig.add_subplot(2, 2, 2)
            ax.set_title("s")
            ax.plot(s_values, s_hist)
            ax = fig.add_subplot(2, 2, 3)
            ax.set_title("v")
            ax.plot(v_values, v_hist)
            # image_plus_roi = img
            # image_plus_roi[0:width, 0:height] = img_roi
            ax = fig.add_subplot(2, 2, 4)
            ax.imshow(img)
            rect = patches.Rectangle(
                (roi[1].start, roi[0].start),
                width,
                height,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.set_title(roi)

    def _extract_scalar_information(self, img: np.ndarray) -> np.ndarray:
        signal = np.zeros_like(img[:, :, 0])
        weights = [1, 1]  # hard code #TODO parameter for model
        i = 0
        plt.figure("signal composition")
        for signal_reduction in self.signal_reductions:
            s = signal_reduction(img)
            signal = signal + weights[i] * s
            plt.subplot(2, 1, 1)
            plt.plot(np.average(s[self.roi], axis=0))
            i = i + 1

        plt.subplot(2, 1, 2)
        plt.plot(np.average(signal[self.roi], axis=0))

        return signal
