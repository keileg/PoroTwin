from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
import shutil
import os
from datetime import datetime
from PIL import Image
import time

import numpy as np
import porepy as pp
import scipy.sparse as sps

import scipy.sparse.linalg as spla
from Costa.iot import PhysicalDevice, IotConfig
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2
import daria as da
import numpy as np
import skimage
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from daria.corrections.color.colorchecker import ColorCorrection
import time


Parameters = Dict[str, Union[float, List[float]]]
Vector = np.ndarray


def correction(img):
    """Standard curvature and color correction. Return ROI."""

    # Preprocessing. Transform to RGB space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Curvature correction
    img = da.curvature_correction(img)

    # Color correction
    roi_cc = (slice(0, 600), slice(0, 700))
    colorcorrection = ColorCorrection()
    img = colorcorrection.adjust(img, roi_cc, verbosity=False, whitebalancing=True)

    # Extract relevant ROI
    img = img[849:4466, 167:7831]

    # TODO calibration

    return img


def determine_tracer(img, base):
    """Extract tracer based on a reference image"""
    # Take (unsigned) difference
    diff = skimage.util.compare_images(img, base, method="diff")

    # Apply smoothing filter
    # diff = skimage.filters.rank.median(diff, skimage.morphology.disk(20))
    diff = skimage.filters.median(diff)

    return diff


def postprocessing(img):
    """Apply simple postprocessing"""
    # Make images smaller
    img = skimage.transform.rescale(img, 0.2, anti_aliasing=True)

    # Transform to BGR space
    return img
    # return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


class ImageProcessor(FileSystemEventHandler):
    # https://stackoverflow.com/questions/57840072/how-to-check-for-new-files-in-a-folder-in-python

    device: PhysicalDevice

    def __init__(self) -> None:
        super().__init__()
        self._target_dir = "/tmp"

    def process_background_image(self, img: np.ndarray) -> bool:
        """Returns True if successful"""
        print(f"Processing background image")
        tic = time.time()
        self._processed_background = correction(img)

        cv2.imwrite(
            self._target_dir + "/background_img.JPG",
            skimage.util.img_as_ubyte(self._processed_background),
        )

        print(f"Done. Elapsed time: {time.time() - tic}.")
        return True

    def on_created(self, event):

        source = Path(event.src_path)
        file_name = source.name

        full_target = Path(self._target_dir) / file_name

        # Get the time stamp for the new file.
        # This can be complexified, caring about operating systems, time zones etc.
        # but the current version gives some information.
        timestamp = datetime.fromtimestamp(os.path.getmtime(source)).strftime(
            "%Y-%m-%d_%H:%M:%S"
        )

        shutil.move(source, full_target)

        self._process_image(full_target, timestamp)

    def _process_image(self, source: Path, time_stamp: str) -> None:
        # Poor man's image processing: Load the image data,
        # convert to numpy array, do some manipulations.

        tic = time.time()

        img = cv2.imread(str(source))
        proc_img = correction(img)
        tracer = determine_tracer(proc_img, self._processed_background)
        proc_tracer = postprocessing(tracer)

        cv2.imwrite(f"/tmp/{time_stamp}.jpg", skimage.util.img_as_ubyte(proc_tracer))

        print(f"Done. Elapsed time: {time.time() - tic}")
        # Finally send information to
        tic = time.time()
        self.device.emit_state({"t": time_stamp}, "image_data", proc_tracer)
        print(f"Time to send image to Azure: {time.time() - tic}")

    def on_any_event(self, event):
        print(event.event_type, event.src_path)


class IotConfigJson(IotConfig):
    def __init__(self, config: dict) -> None:

        self.registry = config.get("COSTA_RSTR")
        self.hub = config.get("COSTA_HSTR")
        self.storage = config.get("COSTA_SSTR")
        self.container = config.get("COSTA_CONTAINER")

        self.devices = {"image_processor": config["COSTA_PHYSICAL_CSTR"]}


if __name__ == "__main__":

    # Folder where the image data will be uploaded.

    with open("image_config.json", "r") as f:
        img_cfg = json.load(f)

    # Watchdog observer which will monitor the image data directory
    observer = Observer()

    processor = ImageProcessor()

    processor.process_background_image(cv2.imread(img_cfg["background_image"]))

    # COSTA Azure configuration: Load connection strings from a
    # file, create a Iot configuration object (COSTA style) which
    # stores the information.
    connection_data_file = "config.json"
    with open(connection_data_file, "r") as f:
        cfg = json.load(f)
    iot_config = IotConfigJson(cfg)

    # Start the monitoring process with a Costa PhysicalDecive that will
    # communicate with Azure.
    with PhysicalDevice("image_processor", iot_config) as device:
        # This seems always to be necessary
        device.emit_clean()

        # Create an image processor which will move the image file
        # away from the data folder and do image processing.
        # The size argument is a temporary construct.
        processor.device = device

        # Let the observer invoke the processor upon evens and let it
        # monitor the data folder.
        observer.schedule(processor, img_cfg["image_source_folder"])
        # Start the process.
        observer.start()

        # Run until interrupted by keyboard
        while True:
            try:
                now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(now)
                time.sleep(60)

            except KeyboardInterrupt:
                observer.stop()

        # Terminate observer.
        observer.join()
