import json
from pathlib import Path
import numpy as np
from Costa.iot import PhysicalDevice, IotConfig
from parabolic_model import ParabolicSolver
from linear_advection_model import (
    LinearAdvection,
    LinearAdvectionWithGravity,
    LinearAdvectionTwoWells,
)


import time


with open("config.json", "r") as f:
    config = json.load(f)
    cstr = config["COSTA_PHYSICAL_CSTR"]


class IotConfigJson(IotConfig):
    def __init__(self, config):

        self.registry = config.get("COSTA_RSTR")
        self.hub = config.get("COSTA_HSTR")
        self.storage = config.get("COSTA_SSTR")
        self.container = config.get("COSTA_CONTAINER")

        self.devices = {"porepy_physical": config["COSTA_PHYSICAL_CSTR"]}

        """
        for key, value in env.items():
            if key.startswith("COSTA_") and key.endswith("_CSTR"):
                try:
                    _, device = next(
                        k for k in value.split(";") if k.startswith("DeviceId")
                    ).split("=")
                    self.devices[device] = value
                except (StopIteration, TypeError, ValueError):
                    pass
        """


def main_advection_two_wells():
    timesteps = 1000
    final = 0.5
    pbm = LinearAdvectionTwoWells(
        {
            "Nx": [150, 150],
            "phys_dims": [1, 1],
            "injection": [1 / 4, 1 / 3],
            "production": [3 / 4, 1 / 3],
        }
    )
    U = pbm._initial_condition(pbm.gb.grids_of_dimension(2)[0])

    with open("config.json", "r") as f:
        config = json.load(f)

    iot_config = IotConfigJson(config)
    with PhysicalDevice("porepy_physical", iot_config) as device:
        device.emit_clean()
        device.emit_state(
            {"t": 0, "dt": final / timesteps, "variable_name": "pressure"}, pbm.pressure
        )
        device.emit_state(
            {"t": 0, "dt": final / timesteps, "variable_name": "concentration"}, U
        )
        for step in range(timesteps + 1):
            t = step * final / timesteps
            params = {"t": t, "dt": final / timesteps}
            Up = U.copy()
            U = pbm.predict(params, uprev=Up)
            device.emit_state(params, U)

            if step % 10 == 0:
                print(f"Time step {step}")

        print("done")


if __name__ == "__main__":
    main_advection_two_wells()
