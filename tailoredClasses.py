"""
inspired by darsia/presets/fluidflowertraceranalysis

"""
from pathlib import Path
from typing import Union, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy import interpolate

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
        patch: slice = None,
        cut: slice = None,
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
            patch (slice): one slice is lower right corner, the other is upper left
                für die inspect routine
                inspect routine ist in tCA überschrieben und wird nur aufgerufen, wenn
                patch is not None
        """
        # Assign tracer analysis
        self.patch = patch
        self.cut = cut
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
            patch=self.patch,
            cut=self.cut,
        )

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
        patch: slice = None,
        cut: slice = None,
        **kwargs,
    ) -> None:
        super().__init__(
            base, signal_reduction, balancing, restoration, model, labels, **kwargs
        )
        self.patch = patch
        self.cut = cut

    def _inspect(
        self, diff, signal, clean_signal, balanced_signal, smooth_signal, concentration
    ):
        if self.patch is not None:
            self._inspect_patch(diff, self.patch)
            # self._inspect_processing(
            #     diff,
            #     signal,
            #     clean_signal,
            #     balanced_signal,
            #     smooth_signal,
            #     concentration,
            # )

        # if self.cut is not None:
        # self._inspect_cut(diff, self.cut, self.signal_reductions)
        # self._inspect_patch(diff, self.cut)

        plt.show()

    def _inspect_processing(
        self, diff, signal, clean_signal, balanced_signal, smooth_signal, concentration
    ):
        patch = self.patch
        plt.figure("propgation of the processing")
        plt.subplot(2, 3, 1)
        plt.imshow(diff[patch])
        plt.subplot(2, 3, 2)
        plt.imshow(signal[patch])
        plt.subplot(2, 3, 3)
        plt.imshow(clean_signal[patch])
        plt.subplot(2, 3, 4)
        plt.imshow(balanced_signal[patch])
        plt.subplot(2, 3, 5)
        plt.imshow(smooth_signal[patch])
        plt.subplot(2, 3, 6)
        plt.imshow(concentration[patch])

    def _inspect_patch(self, img: np.ndarray, patch) -> None:
        """gets called after taking the difference of test-baseline

        Args:
            img (np.ndarray): difference

        """
        img_patch = img[patch]
        height = patch[0].stop - patch[0].start
        width = patch[1].stop - patch[1].start
        plt.figure("patch difference")
        plt.imshow(img_patch)

        # Extract H, S, V components
        hsv = cv2.cvtColor(img_patch.astype(np.float32), cv2.COLOR_RGB2HSV)
        h_img = hsv[:, :, 0]
        s_img = hsv[:, :, 1]
        v_img = hsv[:, :, 2]

        # Extract values
        bins = 100
        h_values = np.linspace(0, 360, bins)
        s_values = np.linspace(0, 1, bins)
        v_values = np.linspace(0, 1, bins)

        bins = 100
        # Setup histograms
        h_hist = np.histogram(h_img, bins=bins, range=(0, 360))[0]
        s_hist = np.histogram(s_img, bins=bins, range=(0, 1))[0]
        v_hist = np.histogram(v_img, bins=bins, range=(0, 1))[0]

        # Plot
        plt.figure("h")
        plt.plot(h_values, h_hist)
        plt.figure("s")
        plt.plot(s_values, s_hist)
        plt.figure("v")
        plt.plot(v_values, v_hist)

        fig, ax = plt.subplots(2)
        fig.canvas.manager.set_window_title(str(patch))
        ax[0].imshow(img)
        temp_array = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
        temp_array[:, :, 0] /= 360
        ax[1].imshow(temp_array)
        rect = patches.Rectangle(
            (patch[1].start, patch[0].start),
            width,
            height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax[0].add_patch(rect)

    def _inspect_cut(self, diff, cut, signal_reductions) -> np.ndarray:
        # habe ich lieber in die signal extraction verlagert um rechenzeit zu sparen
        plt.figure("signal composition")
        for signal_reduction in signal_reductions:
            s = signal_reduction(diff)
            s = np.multiply(s, diff[:, :, 1])  # multiply with saturation
            plt.subplot(2, 1, 1)
            plt.plot(np.average(s[cut], axis=0))

        plt.subplot(2, 1, 2)
        merged_signal = np.average(self._extract_scalar_information(diff)[cut], axis=0)
        plt.plot(merged_signal)
        # plt.figure("smoothed")
        # x = np.linspace(0, np.size(merged_signal), np.size(merged_signal))
        # spline = interpolate.splrep(x, merged_signal, s=1)
        # plt.plot(x, interpolate.BSpline(*spline)(x))
        plt.figure("diff saturation")
        plt.plot(np.average(diff[cut][:, :, 1], axis=0))

    def calibrate_model(self, images: list[darsia.Image], options: dict) -> bool:
        return super().calibrate_model(images, options)

    def define_objective_function(
        self,
        input_images: list[np.ndarray],
        images_diff: list[np.ndarray],
        times: list[float],
        options: dict,
    ):
        print("overritten objective function definition")
        return super().define_objective_function(
            input_images, images_diff, times, options
        )

    def _extract_scalar_information(self, diff: np.ndarray) -> np.ndarray:
        signal = np.zeros_like(diff[:, :, 0])
        weights = [1, 2, 1.5]  # hard code #TODO parameter for model
        i = 0
        sat = diff[:, :, 1]
        b = 0.2
        a = 2
        if self.cut is not None:
            plt.figure("signal composition")
            plt.subplot(2, 1, 1)
        for signal_reduction in self.signal_reductions:
            s = signal_reduction(diff)
            if self.cut is not None:
                plt.plot(np.average(s[self.cut], axis=0))
            signal = signal + weights[i] * s
            i = i + 1

        # signal, -a * np.power((b - diff[:, :, 1]), 2) + a * pow(b, 2)
        # signal = np.multiply(
        #     signal,
        #     1 - sat,
        # )  # influence with saturation
        # signal = signal + 1 - sat
        if self.cut is not None:
            plt.subplot(2, 1, 2)
            plt.plot(np.average(signal[self.cut], axis=0))
            # plt.figure("smoothed")
            # x = np.linspace(0, np.size(merged_signal), np.size(merged_signal))
            # spline = interpolate.splrep(x, merged_signal, s=1)
            # plt.plot(x, interpolate.BSpline(*spline)(x))
            # plt.figure("diff saturation")
            # plt.plot(np.average(diff[self.cut][:, :, 1], axis=0))

        return signal
