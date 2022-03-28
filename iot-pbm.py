import json

from Costa.iot import PbmServer
from parabolic_model import ParabolicSolver


def main():
    pbm = ParabolicSolver(known_solution=False)

    name = "parabolic_porepy"

    with open("config.json", "r") as f:
        config = json.load(f)
        cstr = config["COSTA_PBM_CSTR"]

    with PbmServer(name, cstr, pbm) as server:
        server.wait()


if __name__ == "__main__":
    main()
