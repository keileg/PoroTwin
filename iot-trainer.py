import json
import click
import os
from pathlib import Path

from Costa.ddm import KerasTrainer
from Costa.iot import DdmTrainer, PbmClient


Problem = click.Path(file_okay=False, path_type=Path)

with open("config.json", "r") as f:
    config = json.load(f)
    cstr = config["COSTA_DDM_CSTR"]
    rstr = config["COSTA_RSTR"]
    hstr = config["COSTA_HSTR"]


def main():
    interval = 20
    kwargs = {
        "retrain_frequency": interval,
        "filename": "ddm.h5",
    }
    pbm = PbmClient(rstr, "TestPbm")
    assert pbm.ping_remote()
    trainer = KerasTrainer(pbm)
    with DdmTrainer(trainer, hstr, cstr, **kwargs) as trainer_client:
        trainer_client.listen()


if __name__ == "__main__":
    main()
