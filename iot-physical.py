import json
from pathlib import Path

from Costa.iot import PhysicalDevice
from parabolic_model import ParabolicSolver

import time


with open("config.json", "r") as f:
    config = json.load(f)
    cstr = config["COSTA_PHYSICAL_CSTR"]

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


def main():
    timesteps = 10
    final = 10
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


if __name__ == "__main__":
    main()
