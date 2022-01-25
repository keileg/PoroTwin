import json
from pathlib import Path
import numpy as np
from Costa.iot import PhysicalDevice
from parabolic_model import ParabolicSolver
from linear_advection_model import LinearAdvection

import time


with open("config.json", "r") as f:
    config = json.load(f)
    cstr = config["COSTA_PHYSICAL_CSTR"]


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
    timesteps = 3
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


if __name__ == "__main__":
    main_advection()
