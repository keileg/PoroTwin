#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:56:09 2022

@author: eke001
"""
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from datetime import datetime
import time
import cv2
import numpy as np

import shutil
import json
import os

from Costa import iot
import skimage



def correction(img):
    """Standard curvature and color correction. Return ROI."""

    # Preprocessing. Transform to RGB space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = skimage.transform.rescale(img, 0.2, anti_aliasing=True)

    # Curvature correction
    # img = da.curvature_correction(img)

    # Color correction
    # roi_cc = (slice(0, 600), slice(0, 700))
    # colorcorrection = ColorCorrection()
    # img = colorcorrection.adjust(img, roi_cc, verbosity=False, whitebalancing=True)

    # Extract relevant ROI
    # img = img[849:4466, 167:7831]

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
    #    img = skimage.transform.rescale(img, 0.2, anti_aliasing=True)

    # Transform to BGR space
    return img
    # return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

class LabDevice:
    def __init__(self, name, iot_config, img_config):
        
        self._img_config = img_config
        
        self.process = True
        
        self._num_received=0
        
        if self.process:
            img = cv2.imread(img_config["background_image"])
            self.device = iot.PhysicalDevice(name=name, config=iot_config)

            self.process_background_image(img)

    def upload(self, event):
        source = Path(event.src_path)
        file_name = source.name

        full_name = Path(self._img_config['image_folder']) / file_name

        if self.process:
            time_stamp = datetime.fromtimestamp(os.path.getmtime(full_name)).strftime(
                "%Y-%m-%d_%H:%M:%S"
            )

            self._process_image(full_name, time_stamp)
            
            self._num_received += 1

    def _process_image(self, source: Path, time_stamp: str) -> None:
        # Poor man's image processing: Load the image data,
        # convert to numpy array, do some manipulations.

        tic = time.time()
        
        print(f'Process new image {str(source)}')

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

    def process_background_image(self, img: np.ndarray) -> bool:
        """Returns True if successful"""
        print("Processing background image")
        tic = time.time()
        self._processed_background = correction(img)

        cv2.imwrite(
            str(Path(self._img_config['image_folder']) / Path("background_img.JPG")),
            skimage.util.img_as_ubyte(self._processed_background),
        )

        print(f"Done. Elapsed time: {time.time() - tic}.")
        return True



class ImageHandler(FileSystemEventHandler):
    def __init__(self, uploader):

        self.uploader = uploader

    def on_created(self, event):
        self.uploader.upload(event)


class IotConfigJson(iot.IotConfig):
    def __init__(self, config: dict) -> None:

        self.registry = config.get("COSTA_RSTR")
        self.hub = config.get("COSTA_HSTR")
        self.storage = config.get("COSTA_SSTR")
        self.container = config.get("COSTA_CONTAINER")

        self.devices = {"lab_device": config["COSTA_PHYSICAL_CSTR"]}

    

if __name__ == "__main__":

    # Folder where the image data will be uploaded.
    with open("lab_image_config.json", "r") as f:
        img_cfg = json.load(f)

    # Watchdog observer which will monitor the image data directory
    observer = Observer()

    # COSTA Azure configuration: Load connection strings from a
    # file, create a Iot configuration object (COSTA style) which
    # stores the information.
    connection_data_file = "config.json"
    with open(connection_data_file, "r") as f:
        cfg = json.load(f)
    iot_config = IotConfigJson(cfg)

    device= LabDevice(
        name="lab_device", iot_config=iot_config, img_config=img_cfg
    ) 

    handler = ImageHandler(device)

    observer.schedule(handler, img_cfg["image_folder"])
    # Start the process.
    observer.start()

    for fn in ['DSC04407.JPG', 'DSC04408.JPG', 'DSC04425.JPG']:
        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        full_name = Path(img_cfg["image_folder"]) / Path(fn)
        try:
            os.remove(full_name)
        except:
            pass
        shutil.copy(Path("C:\\Users\\eke001\\Dropbox\\porotwin_images_math\\") / Path(fn),
                    Path(img_cfg['image_folder']))
        
        time.sleep(10)

    observer.stop()
    # Terminate observer.
    observer.join()
