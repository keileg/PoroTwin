import json
from pathlib import Path
import numpy as np
from Costa.iot import PhysicalDevice, IotConfig
from parabolic_model import ParabolicSolver
from linear_advection_model import LinearAdvection, LinearAdvectionWithGravity

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


def main_parabolic():
    source_alphas = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.6,
        1.7,
        1.8,
        1.8,
        2.0,
    ]
    timesteps = 10
    final = 1
    pbm = ParabolicSolver(known_solution=True)
    with PhysicalDevice(cstr) as device:
        for alpha in source_alphas:
            device.emit_clean()
            for step in range(timesteps + 1):
                t = step * final / timesteps
                params = {"ALPHA": alpha, "t": t, "dt": final / timesteps}
                state = pbm.anasol(params)["primary"]
                device.emit_state(params, state)
                time.sleep(1.0)


def main_advection():
    timesteps = 1000
    final = np.pi / 2
    pbm = ParabolicSolver(known_solution=True)
    pbm = LinearAdvection()
    U = pbm._initial_condition(pbm.gb.grids_of_dimension(2)[0])

    with PhysicalDevice(cstr) as device:
        device.emit_clean()
        device.emit_state({"t": 0, "dt": final / timesteps}, U)
        for step in range(timesteps + 1):
            t = step * final / timesteps
            params = {"t": t, "dt": final / timesteps}
            Up = U.copy()
            U = pbm.predict(params, uprev=Up)
            device.emit_state(params, U)

            if step % 100 == 0:
                print(f"Time step {step}")

        print("done")


def main_advection_with_gravity():
    timesteps = 1000
    final = np.pi / 2
    pbm = ParabolicSolver(known_solution=True)
    pbm = LinearAdvectionWithGravity(params={"rho_1": 1, "rho_2": 1.03})
    U = pbm._initial_condition(pbm.gb.grids_of_dimension(2)[0])

    with PhysicalDevice(cstr) as device:
        device.emit_clean()
        device.emit_state({"t": 0, "dt": final / timesteps}, U)
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
    main_advection_with_gravity()
