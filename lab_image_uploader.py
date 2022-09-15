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

import json

from Costa import iot


class LabPhysicalDevice(iot.IotServer):
    def __init__(self, name, config, folder):
        super().__init__(name, config)
        self._folder = folder

    def emit_image(self, filename):
        image = self.upload_file(filename)
        print(f"Upload image {image}")
        self.emit("new_image", {"image": image})

    def upload(self, event):

        source = Path(event.src_path)
        file_name = source.name

        full_name = Path(self._folder) / file_name

        self.emit_image(full_name)


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

        self.devices = {"lab_device": config["COSTA_LAB_CSTR"]}


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

    with LabPhysicalDevice(
        name="lab_device", config=iot_config, folder=img_cfg["image_folder"]
    ) as device:

        handler = ImageHandler(device)

        observer.schedule(handler, img_cfg["image_folder"])
        # Start the process.
        observer.start()

        while True:
            try:
                now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(now)
                time.sleep(60)

            except KeyboardInterrupt:
                observer.stop()
                break

        # Terminate observer.
        observer.join()
