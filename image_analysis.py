from pathlib import Path

import cv2
import daria
import numpy as np
import skimage
import json
import time
from daria.utils.resolution import resize


class ImageAnalysis:
    """
    Class for analyzing tracer data and converting images to spatial concentration maps,
    in parts tailored to the setup of the PoroTwin1 optimal control experiments.
    """

    def __init__(self, base: np.ndarray) -> None:
        """
        Constructor for PoroTwin1 Medium Rig.

        Sets up fixed config file required for preprocessing.
        """
        # Read general config file with data on the physical asset
        f = open("./physical_asset.json", "r")
        self.physical_asset = json.load(f)
        f.close()

        # Define correction objects
        self.curvature_correction = daria.CurvatureCorrection(
            config_source=Path("./image_analysis_config.json")
        )
        self.curvature_correction_rescaled = daria.CurvatureCorrection(
            config_source=Path("./image_analysis_config_rescaled.json")
        )

        # Store original baseline image in RGB color space and extract scalar information
        self.base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        self.base_gray = self.extract_scalar_information(self.base)

        # Preprocess baseline image
        print("ImageAnalysis: Preprocessing baseline image")
        tic = time.time()
        self.base_processed = self.preprocessing(self.base, apply_rescaling=False)
        print(f"Done. Elapsed time: {time.time() - tic}")

        # Some hardcoded config data (incl. not JSON serializable data)
        roi = {
            "color": (slice(50, 550, None), slice(6550, 7330, None)),
            "water": (slice(0, 600), slice(0, 6433)),
        }
        self.resize_factor = 0.2

        # Find the water zone based on the preprocessed baseline image;
        # This routine creates self.reservoir.
        print("Identify and process water zone")
        tic = time.time()
        self.find_water_zone(self.base_processed, roi["water"])
        self.base_without_water = self.neutralize_water_zone(
            self.base_processed, is_rescaled=False
        )
        print("Done. Elapsed time {time.time() - tic}")

        # Rescale reservoir mask complying with the image size of rescaled and preprocessed images
        self.base_rescaled = daria.utils.resolution.resize(
            self.base, self.resize_factor * 100
        )
        self.base_processed_rescaled = self.preprocessing(
            self.base_rescaled, apply_rescaling=True
        )
        self.reservoir_rescaled = skimage.util.img_as_bool(
            skimage.transform.resize(
                self.reservoir, self.base_processed_rescaled.shape[:2]
            )
        )

        # Define concentration analysis object based on the scalar ranged baseline image
        self.concentration_analysis = daria.ConcentrationAnalysis(self.base_gray)

        # Define hard-coded conversion rate to fit the color intensity to concentration.
        # The scaling factor is based on the second test run of the optimal control
        # experiment. For this, the first minutes have been used to track the injection
        # rate, and fit 500 ml/hr.
        self.concentration_analysis.update(scaling_factor=1.0023)

    def preprocessing(
        self, img: np.ndarray, apply_rescaling: bool = True
    ) -> np.ndarray:
        """Coarsening, standard curvature and color correction. Return ROI.

        Args:
            img (np.ndarray): image array
            apply_rescaling (bool): flag controlling whether rescaling is part of
                the preprocessing

        Returns:
            np.ndarray: transformed image
        """

        # Apply curvature correction
        if apply_rescaling:
            img_proc = self.curvature_correction_rescaled(
                np.atleast_3d(
                    img
                    # daria.utils.resolution.resize(img, 100)
                )
            )
        else:
            img_proc = self.curvature_correction(np.atleast_3d(img))

        # TODO Add color correction - the automatic color checker detection does not detect
        # the calibrite color checker, so some new capability has to be implemented.
        # Transform to RGB space and apply color correction
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # colorcorrection = daria.ColorCorrection()
        # img = colorcorrection.adjust(img, roi = self.config["color"]["roi"])

        return img

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
        breakpoint()

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

    def extract_scalar_information(self, img: np.ndarray) -> np.ndarray:
        """
        Have to decide how to extract scalar information.
        There is various possibilities including, converting
        to gray scale or some other channel.

        Args:
            img (np.ndarray): image 3-tensor
        """
        # Return R channel from an RGB image
        if len(img.shape) > 2 and img.shape[2] > 1:
            return np.atleast_3d(img[:, :, 0])
        else:
            assert img.shape == 2 or img.shape[2] == 1
            return img

    def find_water_zone(self, img: np.ndarray, roi: tuple) -> np.ndarray:
        """
        Segment the image into reservoir and non-reservoir.

        Args:
            img (np.ndarray): input image; default None

        Returns:
            np.ndarray: boolean mask detecting the reservoir
        """

        # NOTE: For the optimal control loop example, this will be sufficient.
        # However, in general this is not ideal, as some of the blackened
        # reservoir is actual reservoir. In particular at the top, but also
        # on some specific other locations as well.

        # It turns out transforming to an artificial gray scale computed from the HSV image
        # results in a good basis to detect the sand layer (aside of the bright sand on top)
        img_probe = img.copy()
        img_probe = cv2.cvtColor(img_probe, cv2.COLOR_RGB2HSV)
        img_probe = cv2.cvtColor(img_probe, cv2.COLOR_BGR2GRAY)

        # Initialize
        reservoir = np.ones(img.shape[:2], dtype=bool)

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

        self.reservoir = reservoir

    def neutralize_water_zone(
        self, img: np.ndarray, is_rescaled: bool = True
    ) -> np.ndarray:
        """
        Blacken the water zone above the reservoir to correct for movement behind.

        Args:
            img (np.ndarray): image array
            is_rescaled (bool): flag controlling whether the input image is rescaled

        Return:
            np.ndarray: same image, but with blackened water zone
        """
        # Blacken the top layer (water zone)
        img[~self.reservoir_rescaled if is_rescaled else ~self.reservoir] = 0

        return img

    def determine_concentration(self, img, **kwargs):
        """Extract concentration based on a reference image"""

        # Convert to RGB space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to grayscale or choose a separate color channel for analysis
        gray_img = self.extract_scalar_information(img)

        # Extract concentration map
        concentration = self.concentration_analysis(gray_img, self.resize_factor)

        # Resize image and apply all sorts of corrections
        corrected_concentration = self.preprocessing(
            concentration, apply_rescaling=True
        )

        # Neutralize water zone
        cleaned_concentration = self.neutralize_water_zone(corrected_concentration)

        return cleaned_concentration
