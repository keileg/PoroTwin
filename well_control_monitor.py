from datetime import datetime
import time
import numpy as np

from Costa.iot import PhysicalDevice, IotConfig
import json


class IotConfigJson(IotConfig):
    def __init__(self, config: dict) -> None:

        self.registry = config.get("COSTA_RSTR")
        self.hub = config.get("COSTA_HSTR")
        self.storage = config.get("COSTA_SSTR")
        self.container = config.get("COSTA_CONTAINER")

        self.devices = {"FlowRig": config["COSTA_PHYSICAL_CSTR"]}


class ManuallyControlledFluidFlower(PhysicalDevice):
    def __init__(self, name: str, config: IotConfig):
        super().__init__(name, config)

        self._num_wells = 5

        self._well_rates = np.zeros(self.num_wells)

    def on_control(self, new_rates: np.ndarray) -> dict[str, bool]:

        tol = 1e-8

        changed = np.logical_not(np.allclose(self._well_rates, new_rates, tol))

        s = ""
        for ind, rate in enumerate(new_rates):
            s += f"Rate in well {ind}: {rate}"

            if changed[ind]:
                s += "   <<<<< CHANGED"
            s += "\n"

        print(s)

        self._well_rates = new_rates

        return {"success": True}


if __name__ == "__main__":

    # COSTA Azure configuration: Load connection strings from a
    # file, create a Iot configuration object (COSTA style) which
    # stores the information.
    connection_data_file = "config.json"
    with open(connection_data_file, "r") as f:
        cfg = json.load(f)
    iot_config = IotConfigJson(cfg)

    with ManuallyControlledFluidFlower("FlowRig", iot_config) as device:
        # This seems always to be necessary
        device.emit_clean()

        # Run until interrupted by keyboard
        while True:
            try:
                now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(now)
                time.sleep(60)
            except KeyboardInterrupt:
                break
