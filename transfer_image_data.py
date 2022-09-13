import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import daria
import numpy as np
import skimage
from Costa.iot import IotConfig, PhysicalDevice
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

Parameters = Dict[str, Union[float, List[float]]]
Vector = np.ndarray


def preprocessing(img, **kwargs):
    """Coarsening, standard curvature and color correction. Return ROI."""

    # NOTE requires some parameter tuning. Need to choose:
    # * "resize_factor"
    # * "roi_color_checker"
    # * "roi"
    # * "curvature_config"

    # Resize factor in percent
    resize_factor = kwargs.pop("resize_factor", 0.1) * 100

    # Coarsen imager
    img = resize(img, resize_factor)

    # Curvature correction
    curvature_config = kwargs.pop("curvature_config", None)
    # TODO choose proper config file and uncomment
    #img = daria.curvature_correction(img) if curvature_config is None else daria.curvature_correction(img, curvature_config)

    # Transform to RGB space and apply color correction
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    roi_cc = kwargs.pop("roi_color_checker", (slice(0, img.shape[0]), slice(0, img.shape[1])))
    colorcorrection = daria.ColorCorrection()
    img = colorcorrection.adjust(img, roi_cc)

    # Extract relevant ROI
    roi = kwargs.pop("roi", (slice(0, img.shape[0]), slice(0, img.shape[1])))
    img = img[roi]

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img

def determine_tracer(img, base, **kwargs):
    """Extract tracer based on a reference image"""

    # NOTE: User input required:
    # * "resize_factor"
    # * "thresh_min"
    # * "thresh_max"

    # Take (unsigned) difference
    diff = skimage.util.compare_images(img, base, method="diff")

    # Same scaling factor as in preprocessing routine
    resize_factor = kwargs.pop("resize_factor", 0.1)

    # Apply smoothing filter
    diff = skimage.filters.rank.median(diff, skimage.morphology.disk(20 * resize_factor))

    # Calibrate color-concentration map

    # Require threshold values for identifying both 0 and 1 concentrations.
    thresh_min = kwargs.pop("thresh_min", np.min(diff))
    thresh_max = kwargs.pop("thresh_max", np.max(diff))

    # Calibrated thresholding
    tracer_min_mask = diff <= thresh_min
    diff[tracer_min_mask] = 0
    diff[~tracer_min_mask] -= thresh_min
    tracer_max_mask = diff >= thresh_max - thresh_min
    diff[tracer_max_mask] = thresh_max - thresh_min

    # Rescale image to range full_range
    diff = skimage.exposure.rescale_intensity(diff)

    return diff



class ImageProcessor(FileSystemEventHandler):
    # https://stackoverflow.com/questions/57840072/how-to-check-for-new-files-in-a-folder-in-python

    device: PhysicalDevice

    def __init__(self) -> None:
        super().__init__()
        self._target_dir = "/tmp"

        # TODO define config for image analysis
        #self.config = {
        #    "resize_factor": 0.4,
        #    #"curvature_config": {...},
        #    "roi_color_checker": (slice(0,300), slice(0,300)),
        #    #"roi": (slice(), slice())
        #    "thresh_min": 7,
        #    "thresh_max": 47,
        #}

    def process_background_image(self, img: np.ndarray) -> bool:
        """Returns True if successful"""
        print(f"Processing background image")
        tic = time.time()
        self._processed_background = preprocessing(img, **self.config)

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
        proc_img = preprocessing(img, **self.config)
        tracer = determine_tracer(proc_img, self._processed_background, **self.config)
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
