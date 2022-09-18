from pathlib import Path

import cv2
import daria
import numpy as np
import skimage
from daria.corrections.shape.curvature import simple_curvature_correction
from daria.utils.resolution import resize


class PoroTwin1MediumFluidFlowerAnalysis:
    """
    Class for analyzing tracer data and converting images to spatial concentration maps.

    # Example:

    # Read in baseline figure - TODO: May be updated
    baseline = cv2.imread(str(Path(.../images/porotwin/test_run/20220914-142627.TIF")))

    # Define FluidFlower with baseline - also identifies the water zone for later neutralization
    ff = PoroTwin1MediumFluidFlowerAnalysis(baseline)

    # Image of tracer at later time
    path = Path(.../images/porotwin/test_run/20220914-144227.TIF")

    # Read and convert to RGB
    tracer_image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

    # Determine tracer concentration
    tracer = ff.determine_tracer(tracer_image)

    # Store concentation to file (as numpy array)
    ff.store(tracer, path, store_image=False)
    """

    def __init__(self, base: np.ndarray) -> None:
        """
        Constructor for PoroTwin1 Medium Rig.

        Sets up fixed config file required for preprocessing.

        Args:
            resize_factor (float): scaling factor of images in x and y direction; default set to 1.
        """
        # Run the config setup routine and copy paste the resulting config file

        # For images with original size
        self.config = {
            "init": {
                "resize_factor": 1.0,
                "horizontal_bulge": 2e-09,
                "vertical_bulge": 1e-09,
            },
            "crop": {
                "pts_src": [[281, 34], [259, 4443], [7661, 4426], [7632, 20]],
                "width": 0.92,
                "height": 0.555,
                "in meters": True,
            },
            "bulge_vertical": {
                "vertical_bulge": -7.067016446290003e-10,
                "horizontal_center_offset": -6,
                "vertical_center_offset": -52,
            },
            "bulge_horizontal": {
                "horizontal_bulge": -1.1990602746616518e-09,
                "horizontal_center_offset": 0,
                "vertical_center_offset": 139,
            },
            "stretch": {
                "horizontal_stretch": -1.3896593681548202e-09,
                "horizontal_center_offset": 0,
                "vertical_stretch": -3.0852102803383441e-09,
                "vertical_center_offset": 0,
            },
            "color": {"roi": (slice(50, 550, None), slice(6550, 7330, None))},
        }

        # Config routine for images rescaled with factor 0.2
        self.config_rescaled = {
            "init": {
                "resize_factor": 0.2,
                "horizontal_bulge": 4.0000000000000007e-10,
                "vertical_bulge": 2.0000000000000003e-10,
            },
            "dimensions": {"width": 0.92, "height": 0.555},
            "crop": {
                "pts_src": [
                    [56.2, 6.800000000000001],
                    [51.800000000000004, 888.6],
                    [1532.2, 885.2],
                    [1526.4, 4.0],
                ],
                "width": 0.92,
                "height": 0.555,
                "in meters": True,
            },
            "bulge_vertical": {
                "vertical_bulge": -1.6915815688365437e-08,
                "horizontal_center_offset": 0,
                "vertical_center_offset": 10,
            },
            "bulge_horizontal": {
                "horizontal_bulge": -2.4547829674772222e-08,
                "horizontal_center_offset": 165,
                "vertical_center_offset": 28,
            },
            "stretch": {
                "horizontal_stretch": -3.6314399847746792e-08,
                "horizontal_center_offset": 0,
                "vertical_stretch": -8.9127724783092833e-08,
                "vertical_center_offset": 0,
            },
            "color": {"roi": (slice(10, 110, None), slice(1310, 1466, None))},
        }

        # Store original baseline image in RGB color space
        self.base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        self.base_gray = self.extract_scalar_information(self.base)

        # Find the water zone based on the preprocessed baseline image
        self.base_processed = self.preprocessing(self.base, apply_rescaling=False)
        self.find_water_zone(self.base_processed)

        # Rescale complying with the image size of rescaled images
        self.base_processed_rescaled = self.preprocessing(self.base)
        self.reservoir_rescaled = skimage.util.img_as_bool(
            skimage.transform.resize(
                self.reservoir, self.base_processed_rescaled.shape[:2]
            )
        )
        self.base_processed_rescaled = self.neutralize_water_zone(
            self.base_processed_rescaled
        )

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

        config = self.config_rescaled if apply_rescaling else self.config

        # Follow standard procedure for FluidFlower rigs to correct for geometrical distortions.
        img = np.atleast_3d(img)

        img = resize(img, config["init"]["resize_factor"] * 100)
        img = simple_curvature_correction(img, **config["init"])
        img = daria.extract_quadrilateral_ROI(img, **config["crop"])
        img = simple_curvature_correction(img, **config["bulge_vertical"])
        img = simple_curvature_correction(img, **config["bulge_horizontal"])
        img = simple_curvature_correction(img, **config["stretch"])

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

    def find_water_zone(self, img: np.ndarray) -> np.ndarray:
        """
        Segment the image into reservoir and non-reservoir.

        Args:
            img (np.ndarray): input image; default None

        Returns:
            np.ndarray: boolean mask detecting the reservoir
        """

        # TODO not optimal yet. Try to improve... For the optimal control it will be sufficient though.

        # Initialize
        reservoir = np.ones(img.shape[:2], dtype=bool)

        # Consider candidate for water zone
        # roi = (slice(0, 800), slice(0, 6436))

        # Iterate in 100er pieces through the reservoir
        for i in range(65):
            under_roi = (slice(330, 800), slice(i * 100, min((i + 1) * 100, 6436)))
            above_under_roi = (slice(0, 330), slice(i * 100, (i + 1) * 100))
            reservoir[above_under_roi] = 0

            probe = img[under_roi][:, :, 2]  # Consider the blue channel
            probe_reg = skimage.filters.rank.median(probe, skimage.morphology.disk(20))
            thresh = skimage.filters.threshold_otsu(probe_reg)
            reservoir[under_roi] = probe_reg < thresh

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
        # Blacken the top later
        if is_rescaled:
            img[~self.reservoir_rescaled] = 0
        else:
            img[~self.reservoir] = 0

        return img

    def determine_tracer(self, img, **kwargs):
        """Extract tracer based on a reference image"""

        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to grayscale or choose a separate color channel for analysis
        gray_img = self.extract_scalar_information(img)

        # Take (unsigned) difference
        diff = skimage.util.compare_images(gray_img, self.base_gray, method="diff")

        # Resize image and apply all sorts of corrections
        diff = self.preprocessing(diff)

        # Neutralize water zone
        diff = self.neutralize_water_zone(diff)

        # Apply smoothing filter
        diff = diff[:, :, 0]
        diff = skimage.restoration.denoise_tv_chambolle(diff, weight=0.4)

        # TODO Calibration needed. For now the images are transformed to float data
        # and stretched onto the interval [0,1].
        diff = skimage.util.img_as_float(diff)
        diff = skimage.exposure.rescale_intensity(diff)

        return diff
