"""
Module containing the setup for the fluidflower rig, used for the PoroTwin1 optimal control project.
"""
import json
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import daria
import numpy as np
import skimage


class TailoredConcentrationAnalysis(daria.ConcentrationAnalysis):
    def __init__(self, base, color, resize_factor, **kwargs) -> None:
        super().__init__(base, color, **kwargs)
        self.resize_factor = resize_factor

    def postprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        signal = cv2.resize(
            signal,
            None,
            fx=self.resize_factor,
            fy=self.resize_factor,
            interpolation=cv2.INTER_AREA,
        )
        signal = skimage.restoration.denoise_tv_chambolle(signal, 0.1)
        signal = np.atleast_3d(signal)
        return super().postprocess_signal(signal)


class ImageAnalysis:
    """
    Class for analyzing tracer data and converting images to spatial concentration maps,
    in parts tailored to the setup of the PoroTwin1 optimal control experiments.
    """

    def __init__(
        self,
        baseline: Union[str, Path, list[str], list[Path]],
        config_source: Union[str, Path],
        path_to_cleaning_filter: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Constructor for PoroTwin1 Rig.

        Sets up fixed config file required for preprocessing.

        Args:
            base (str, Path or list of such): baseline images, used to
                set up analysis tools and cleaning tools
            config_source (str or Path): path to config dict
            path_to_cleaning_filter (str or Path, optional):
        """
        # Read general config file
        f = open(config_source, "r")
        self.config = json.load(f)
        f.close()

        # Some hardcoded config data (incl. not JSON serializable data)
        roi = {
            "color": (slice(0, 550, None), slice(6550, 7330, None)),
            "water": (slice(0, 600), slice(0, 6433)),
        }

        # Define correction objects
        self.color_correction = daria.ColorCorrection(roi=roi["color"])
        self.curvature_correction = daria.CurvatureCorrection(
            config=self.config["geometry"]
        )

        # Find the water zone based on the preprocessed baseline image;
        # This routine creates self.reservoir.
        print("ImageAnalysis: Preprocessing baseline image")
        tic = time.time()
        if not isinstance(baseline, list):
            baseline = [baseline]
        reference_base = baseline[0]
        processed_base = self._read(reference_base)
        self.find_water_zone(processed_base, roi["water"])
        self.neutralize_water_zone(processed_base)
        self.base = processed_base
        print(f"Done. Elapsed time: {time.time() - tic}")

        # Define concentration analysis. To speed up significantly the process,
        # invoke resizing of signals within the concentration analysis.
        # Also use pre-calibrated information.
        self.concentration_analysis = TailoredConcentrationAnalysis(
            processed_base, color="gray", resize_factor=0.2
        )
        if path_to_cleaning_filter is not None:
            self.concentration_analysis.read_calibration_from_file(
                self.config["calibration"],
                path_to_cleaning_filter,
            )
        else:
            print(
                "ImageAnalysis: Setting up cleaning filter for concentration analysis."
            )
            tic = time.time()
            self._setup_concentration_analysis(self.config["calibration"], baseline)
            print(f"Done. Elapsed time: {time.time() - tic}")

    def _setup_concentration_analysis(
        self, config: dict, baseline_images: list[Union[str, Path]]
    ) -> None:
        """
        Wrapper to find cleaning filter of the concentration analysis.

        Args:
            config (dict): dictionary with scaling parameters
            baseline_images (list of str or Path): paths to baseline images.
        """
        # Define scaling factor
        self.concentration_analysis.scaling = config["scaling"]

        # Find cleaning filter
        images = [self._read(path) for path in baseline_images]
        neutralized_images = [self.neutralize_water_zone(img) for img in images]
        self.concentration_analysis.find_cleaning_filter(neutralized_images)

    # ! ----- I/O

    def _read(self, path: Union[str, Path]) -> daria.Image:
        return daria.Image(
            img=path,
            curvature_correction=self.curvature_correction,  # , color_correction = self.color_correction
        )

    def load_and_process_image(self, path: Union[str, Path]):
        """
        Load image for further analysis. Do all corrections and processing needed.

        Args:
            path (str or Path): path to image
        """
        self.img = self._read(path)
        self.neutralize_water_zone(self.img)

    def store(
        self,
        img: np.ndarray,
        path: Path,
        cartesian_indexing: bool = True,
        store_image: bool = False,
    ) -> bool:
        """Convert to correct format (use Cartesian indexing by default)
        and store to file.
        """
        plain_path = path.with_suffix("")

        # Store the image
        if store_image:
            cv2.imwrite(
                str(plain_path) + "_img.jpg",
                skimage.util.img_as_ubyte(img),
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )

        # Store array
        np.save(
            str(plain_path) + "_concentration.npy",
            daria.matrixToCartesianIndexing(img) if cartesian_indexing else img,
        )

        return True

    # ! ----- Neutralization of water zone (dynamic, depending on baseline)

    def find_water_zone(self, img: daria.Image, roi) -> np.ndarray:
        """
        Segment the image into reservoir and non-reservoir.

        Args:
            img (np.ndarray): input image
            roi (tuple of slices): potential waterzone roi

        Returns:
            np.ndarray: boolean mask detecting the reservoir
        """

        # NOTE: For the optimal control loop example, this will be sufficient.
        # However, in general this is not ideal, as some of the blackened
        # reservoir is actual reservoir. In particular at the top, but also
        # on some specific other locations as well.

        # It turns out transforming to an artificial gray scale computed from the HSV image
        # results in a good basis to detect the sand layer (aside of the bright sand on top)
        img_probe = img.img.copy()
        img_probe = cv2.cvtColor(img_probe, cv2.COLOR_RGB2HSV)
        img_probe = cv2.cvtColor(img_probe, cv2.COLOR_BGR2GRAY)

        # Initialize
        reservoir = np.ones(img_probe.shape[:2], dtype=bool)

        # Smooth image slightly, and threshhold wth automatic threshold parameter
        probe_reg = skimage.filters.rank.median(img_probe, skimage.morphology.disk(20))
        thresh = skimage.filters.threshold_otsu(probe_reg)

        # Automatize, assuming the reservoir is below the water
        mask = probe_reg > thresh
        is_below = (
            np.average(np.where(mask), axis=0)[0]
            > np.average(np.where(~mask), axis=0)[0]
        )

        # Only distrust a specific roi:
        reservoir[roi] = mask[roi] if is_below else ~mask[roi]

        # Hardcode that in the top pixels, there is no reservoir (this is to deactivate
        # some water zone above the color checker.
        reservoir[(slice(0, 20), slice(0, reservoir.shape[1]))] = False

        self.reservoir = reservoir

    def neutralize_water_zone(self, img: daria.Image) -> None:
        """
        Blacken the water zone above the reservoir to correct for movement behind.

        Args:
            img (np.ndarray): image array
        """
        # Blacken the top layer (water zone)
        img.img[~self.reservoir] = 0

        return img

    # ! ----- Concentration analysis

    def determine_concentration(self) -> daria.Image:
        """Extract tracer from currently loaded image, based on a reference image.

        Returns:
            daria.Image: image array of spatial concentration map
        """
        # Make a copy of the current image
        img = self.img.copy()

        # Extract concentration map - includes rescaling
        concentration = self.concentration_analysis(img)

        return concentration
