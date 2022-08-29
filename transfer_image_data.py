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


Parameters = Dict[str, Union[float, List[float]]]
Vector = np.ndarray


class ImageProcessor(FileSystemEventHandler):
    # https://stackoverflow.com/questions/57840072/how-to-check-for-new-files-in-a-folder-in-python
    def __init__(self, device: PhysicalDevice, sz: np.ndarray) -> None:
        super().__init__()
        self._target_dir = "/tmp"

        self._sz = sz
        print(f"Size of sz {self._sz}")
        self._device = device

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

    def _process_image(self, source: Path, timestamp: str) -> None:
        # Poor man's image processing: Load the image data,
        # convert to numpy array, do some manipulations.
        img = Image.open(source)
        img.load()

        data = np.asarray(img)

        # For now, this is a proxy for the image processing
        if data.ndim > 2:
            data = np.mean(data, axis=-1)
        print(data.shape)

        state = data[: self._sz[0], : self._sz[1]]

        # Finally send information to
        self._device.emit_state({"t": timestamp}, "image_data", state)

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
    path_to_data_folder = Path(".")
    # Watchdog observer which will monitor the image data directory
    observer = Observer()

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
        processor = ImageProcessor(device=device, sz=np.array([5, 5]))

        # Let the observer invoke the processor upon evens and let it
        # monitor the data folder.
        observer.schedule(processor, path_to_data_folder)
        # Start the process.
        observer.start()

        # Run until interrupted by keyboard
        while True:
            try:
                now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                device.emit_state({"t": now}, "image_data", np.ones(2))
                print(now)
                time.sleep(60)
            except KeyboardInterrupt:
                observer.stop()

        # Terminate observer.
        observer.join()
