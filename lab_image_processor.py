"""Runscript that detects new image files, assumed to be pictures of a tracer
concentration, post process them to get hold of the 
"""
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from datetime import datetime
import time
import cv2
import numpy as np

from image_analysis import ImageAnalysis


import json
import os

from Costa import iot



class LabDevice:
    def __init__(self, name, iot_config, img_config):
        
        self._img_config = img_config
        
        self.process = True
        
        self._num_received=0
        
        if self.process:
            img = cv2.imread(img_config["background_image"])
            self.device = iot.PhysicalDevice(name=name, config=iot_config)
            
            self.image_analysis = ImageAnalysis(img)

    def upload(self, event):
        source = Path(event.src_path)
        
        # This method will be invoked when the file is created, but the file should not
        # be read before the entire file is written to file. The solution is to wait a
        # few seconds before loading.
        # https://stackoverflow.com/questions/57941401/permissionerror-errno-13-permission-denied-in-watchdog-monitoring-read-text
        time.sleep(2)

        if self.process:
            time_stamp = datetime.fromtimestamp(os.path.getmtime(source)).strftime(
                "%Y-%m-%d_%H:%M:%S"
            )

            self._process_image(source, time_stamp)
            
            self._num_received += 1

    def _process_image(self, source: Path, time_stamp: str) -> None:
        # Poor man's image processing: Load the image data,
        # convert to numpy array, do some manipulations.

        tic = time.time()
        
        print(f'Process new image {str(source)}')
        img = cv2.imread(str(source))
        concentration = self.image_analysis.determine_concentration(img)
        self.image_analysis.store(concentration, Path(self._img_config['storage_folder']) / Path(time_stamp))
        
        print(f"Done. Elapsed time: {time.time() - tic}")
        # Finally send information to
        tic = time.time()
        self.device.emit_state({"t": time_stamp}, "image_data", concentration)
        print(f"Time to send image to Azure: {time.time() - tic}")        


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
    
    print("Processor is online")
    
    while True:
        try:
            now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            print(now)
            time.sleep(60)

        except KeyboardInterrupt:
            observer.stop()
            break
    

    # for fn in ['DSC04407.JPG', 'DSC04408.JPG', 'DSC04425.JPG']:
    #     now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    #     full_name = Path(img_cfg["image_folder"]) / Path(fn)
    #     try:
    #         os.remove(full_name)
    #     except:
    #         pass
    #     shutil.copy(Path("C:\\Users\\eke001\\Dropbox\\porotwin_images_math\\") / Path(fn),
    #                 Path(img_cfg['image_folder']))
        
    #     time.sleep(10)

    observer.stop()
    # Terminate observer.
    observer.join()
