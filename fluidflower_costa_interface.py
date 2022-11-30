"""

"""
import numpy as np
from typing import Union
from porepy_fluidflower import fluidflower_well_test
from porepy_fluidflower.geometry.utils.projections import flatten_image
import json
from datetime import datetime
import time
from Costa.iot import IotConfig, PbmServer
from Costa.api import PhysicsModel


Parameters = dict[str, Union[float, list[float]]]
Vector = np.ndarray
Matrix = np.ndarray

VectorData = Union[Vector, dict[str, Vector]]

from porepy_physical_model import PorePyPhysicsModel, PorePyCostaModel


class FluidFlowerCostaInterface(PhysicsModel):
    def __init__(self, params: dict) -> None:

        self.model = fluidflower_well_test.FluidFlowerWellTest(params)
        self.model.prepare_simulation()

    @property
    def ndof(self) -> int:
        """Return the number of degrees of freedom."""
        return self.model.sd.num_cells

    def dirichlet_dofs(self) -> Vector:
        return []

    def _image_to_sim(self, img: np.ndarray) -> np.ndarray:
        """Transform an array from image format (0-offset )


        Args:
            img (np.ndarray): DESCRIPTION.

        Returns:
            TYPE: DESCRIPTION.

        """
        img_vec = flatten_image(img)

        return self.model.project_to_simulation_vector(img_vec)

    def _sim_to_image(self, sim):
        """ """
        proj_sim = self.model.proj_simulation_2_image.dot(sim)
        return np.flip(
            proj_sim.reshape((self.model.num_pixels[1], self.model.num_pixels[0])),
            0,
        )

    def initial_condition(self, params: Parameters) -> VectorData:
        """Return the configured initial condition for a set of parameters."""
        return np.zeros(self.ndof)

    def predict(self, params: Parameters, uprev: VectorData) -> VectorData:
        """Make an uncorrected prediction of the next timestep given the
        previous timestep.  This is nothing more than a standard discrete
        timestep method.

        :param params: Dictionary of parameters.
            By convention the timestep is named 'dt'.
        :param uprev: Previous timestep.  May be ignored by a stationary solver.
        :return: Prediction of next timestep.
        """
        self._parse_params(params)

        current_concentration = self._image_to_sim(uprev)
        # Perform a predictor step
        tic = time.time()
        predicted_concentration = self.model.predict(params, current_concentration)
        print(f"Time for prediction: {time.time() - tic}")
        return self._sim_to_image(predicted_concentration)

    def residual(
        self, params: Parameters, uprev: VectorData, unext: VectorData
    ) -> VectorData:
        """Calculate the residual Au - b given the assumed solution unext.

        :param params: Dictionary of parameters.
            By convention the timestep is named 'dt'.
        :param uprev: Previous timestep.  May be ignored by a stationary solver.
        :param unext: The purported exact or experimental solution.
        :return: The residual Au - b."""
        self._parse_params(params)

        uprev_sim = self._image_to_sim(uprev)
        unext_sim = self._image_to_sim(unext)

        residual = self.model.residual(params, uprev_sim, unext_sim)

        return self._sim_to_image(residual)

    def correct(
        self, params: Parameters, uprev: VectorData, sigma: VectorData
    ) -> VectorData:
        """Calculate a corrected prediction of the next timestep given
        the previous timestep and a right-hand side perturbation.

        :param params: Dictionary of parameters.
            By convention the timestep is named 'dt'.
        :param uprev: Previous timestep.  May be ignored by a stationary solver.
        :param sigma: Right-hand-side perturbation. If equal to zero, this method
            should be equivalent to predict(params, uprev).
        :return: Corrected prediction of next timestep.
        """
        self._parse_params(params)

        current_concentration = self._image_to_sim(uprev)
        extra_source = self._image_to_sim(sigma)
        # Perform a predictor step
        tic = time.time()
        predicted_concentration = self.model.correct(
            params, current_concentration, extra_source
        )
        print(f"Time for correction: {time.time() - tic}")
        return self._sim_to_image(predicted_concentration)

    def qi(self, params: Parameters, u: VectorData, name: str) -> float:
        """Calculate a named quantity of interest.

        :param params: Dictionary of parameters.
            By convention the timestep is named 'dt'.
        :param u: Coefficient data.
        :param name: Name of quantity to compute.
        :return: Computed quantity.
        """
        raise NotImplementedError()

    def _parse_params(self, params: dict) -> dict:
        if "time" in params and "time_prev" not in params:
            params["time_prev"] = params["time"]
        return params


if __name__ == "__main__":

    class IotConfigJson(IotConfig):
        def __init__(self, config: dict) -> None:

            self.registry = config.get("COSTA_RSTR")
            self.hub = config.get("COSTA_HSTR")
            self.storage = config.get("COSTA_SSTR")
            self.container = config.get("COSTA_CONTAINER")

            self.devices = {"TestPbm": config["COSTA_PBM_CSTR"]}

        # COSTA Azure configuration: Load connection strings from a

    # file, create a Iot configuration object (COSTA style) which
    # stores the information.
    connection_data_file = "config.json"
    with open(connection_data_file, "r") as f:
        cfg = json.load(f)
    iot_config = IotConfigJson(cfg)

    model_params = {
        "export_to_file": True,
        "file_name": "fluidflower_well_test",
        "folder_name": "out",
        "concentration_folder_name": "out/images",
        "concentration_file_name": "concentration",
        "num_pixels": [280, 150],
    }

    model = FluidFlowerCostaInterface(model_params)

    # Start the monitoring process with a Costa PhysicalDecive that will
    # communicate with Azure.
    with PbmServer("TestPbm", iot_config, model) as device:
        print("Server is online")

        print(device.on_ping({}))

        while True:
            try:
                now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(now)
                time.sleep(60)

            except KeyboardInterrupt:
                break
