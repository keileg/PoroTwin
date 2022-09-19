"""The module contains a setup for a two well system, with one injector and one
producer.
"""
from typing import Union
import numpy as np
from datetime import datetime
import porepy as pp
import time
from Costa.iot import IotConfig
import json
from porepy_physical_model import PorePyPhysicsModel, PorePyPhysicalDevice

from linear_advection_model import LinearAdvectionModel


def create_grid(
    config: dict,
) -> tuple[pp.MixedDimensionalGrid, dict[str, Union[float, int]]]:
    """Create a Cartesian grid for domain of specified size and

    Args:
        config (dict): Configuration. Should contain fields:
            physdims: Array-like domain size in x and y direction.
            Nx: Array-like number of cells in each direction.

    Returns:
        mdg (pp.MixedDimensionalGrid): PorePy grid. Wrapped as a mixed-dimensional grid
            to be compatible with PorePy's automatic differentiation framework.
        box (dict): Domain, specified by dictionary keys xmin, xmax, ymin, ymax.

    """

    # Grid size and resolution
    Nx = np.asarray(config["Nx"])
    phys_dims = config.get("phys_dims", np.ones(Nx.size))

    g = pp.CartGrid(Nx, phys_dims)
    g.compute_geometry()

    # Convert to mixed-dimensional grid (this is a bit of an overkill
    # for this problem, but it works).
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains([g])

    box = pp.UnitSquareDomain()

    return mdg, box


class LinearAdvectionInjectionProduction(LinearAdvectionModel):
    """Full setup of a linear advection problem for a system with injection and
    production wells.

    The flow field is calculated by calling an incompressible flow system.
    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)

        self.params = params

        self.prepare_simulation()

    def create_grid(self):
        mdg, box = create_grid(self.params)
        self.mdg = mdg
        self.box = box
        self.dim = 2

        self.g = mdg.subdomains(dim=2)[0]

        # Store the location of the wells. Coordinates will be converted
        # to cell indices when we have a grid to work with.
        self._well_indices = self.g.closest_cell(self.params["well_coordinates"])
        self._well_rates = np.zeros(self._well_indices.size)

    def _set_parameters(self) -> None:

        g = self.g
        data = self.mdg.subdomain_data(g)

        bc = self._bc_type(g)
        bc_values = self._bc_values(g)

        mass_weight = self._mass_weight(g)

        pp.initialize_data(
            g,
            data,
            self.parameter_key,
            {
                "bc": bc,
                "bc_values": bc_values,
                "mass_weight": mass_weight,
            },
        )

        # Solve flow problem and update the Darcy field.
        self.update_flow_field()

    def _bc_type(self, g):
        # Define boundary condition on faces
        xf = g.face_centers
        top = np.where(np.abs(xf[1] - np.max(xf[1])) < 1e-5)[0]
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, top, "dir")

    def _mass_weight(self, g: pp.Grid) -> np.ndarray:
        porosity = self.params["porosity"]
        return porosity * np.ones(g.num_cells)

    def update_flow_field(self) -> None:
        """Calculate the flow field, calling an incompressible flow model.

        The method can be called from external clients to update the flow field, e.g.,
        if the rates have changed.

        """
        print("Update flow field to new well controls")
        params = self.params.copy()
        params["file_name"] = "flow_solution"
        params["folder_name"] = "tmp_flow"
        params["well_rates"] = self._well_rates

        model = FlowModel(params=params)

        # Set up and solve flow problem. This will create a new grid,

        pp.run_stationary_model(model, {})

        # Post process flux field and store it.
        g = model.mdg.subdomains(dim=model.mdg.dim_max())[0]
        flux_eq = model._flux([g])

        data = self.mdg.subdomain_data(self.g)

        data[pp.PARAMETERS][self.parameter_key]["darcy_flux"] = flux_eq.evaluate(
            model.dof_manager
        ).val

        self._eq_manager.discretize(self.mdg)

    def control(self, payload: dict) -> dict:
        self.set_well_rates(payload)
        self.update_flow_field()
        return {"success": True}

    def initial_condition(self):

        g = self.mdg.subdomains(dim=self.mdg.dim_max())[0]

        return np.zeros(g.num_cells)


class FlowModel(pp.IncompressibleFlow):
    """Incompressible flow model for the two well system. Used to calculate the flow
    field.
    """

    def __init__(self, params: dict) -> None:
        super().__init__(params=params)

    def create_grid(self):
        mdg, box = create_grid(self.params)
        self.mdg = mdg
        self.box = box
        self.dim = 2

    def _permeability(self, sd: pp.Grid):
        return np.ones(sd.num_cells) * self.params["permeability"]

    def _bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on the top boundary."""
        xf = g.face_centers
        top = np.where(np.abs(xf[1] - np.max(xf[1])) < 1e-5)[0]
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, top, "dir")

    def _source(self, g: pp.Grid) -> np.ndarray:
        """Zero source term.
        Units: m^3 / s
        """
        src = np.zeros(g.num_cells)
        cells = g.closest_cell(self.params["well_coordinates"])
        src[cells] = self.params["well_rates"]

        return src


if __name__ == "__main__":
    offline = False

    problem_setup = {
        "Nx": [50, 50],
        "phys_dims": [920, 575],
        "well_coordinates": np.array(
            [[235, 235], [660, 160], [120, 420], [170, 290], [500, 375], [760, 460]]
        ).T,
        "well_rates": [1, -1, 0, 0, 0, 0],
        "permeability": 666.34 * pp.DARCY,
        "porosity": 0.375,
    }

    if offline:
        #
        model = LinearAdvectionInjectionProduction(problem_setup)
        exp = pp.Exporter(model.mdg, file_name="two_wells", folder_name="tmp_viz")

        time_steps = [0]

        T = 1
        n_steps = 20
        params = {"dt": T / n_steps}

        U = model.initial_condition()
        g = model.g
        model.mdg.subdomain_data(g)[pp.STATE][model.variable] = U

        exp.write_vtu([model.variable], time_dependent=True, time_step=0)

        well_controls = [[1, -1], [1, 0], [-1, 1]]

        tot_step = 0

        for control in well_controls:

            model.control({"well_rates": control})

            for i in range(n_steps):
                Up = U.copy()
                # The AD formulation of PorePy solves for the update of U, thus interpret
                # the result as an update.
                U += model.solve(params, uprev=Up)

                model.dof_manager.distribute_variable(U)

                if i > 0 and i % 5 == 0:
                    print(f"Time step {i + tot_step}")
                    mdg = model.mdg
                    g = model.mdg.subdomains(dim=2)[0]
                    state = mdg.subdomain_data(g)[pp.STATE]
                    state[model.variable] = U
                    exp.write_vtu(
                        [model.variable], time_dependent=True, time_step=i + tot_step
                    )

                    time_steps.append(i + tot_step)

            exp.write_vtu([model.variable], time_dependent=True, time_step=i + tot_step)

            time_steps.append(i + 1 + tot_step)
            tot_step += n_steps

        exp.write_pvd(time_steps)
    else:

        class IotConfigJson(IotConfig):
            def __init__(self, config: dict) -> None:

                self.registry = config.get("COSTA_RSTR")
                self.hub = config.get("COSTA_HSTR")
                self.storage = config.get("COSTA_SSTR")
                self.container = config.get("COSTA_CONTAINER")

                self.devices = {"porepy_lab": config["COSTA_PHYSICAL_CSTR"]}

            # COSTA Azure configuration: Load connection strings from a

        # file, create a Iot configuration object (COSTA style) which
        # stores the information.
        connection_data_file = "config.json"
        with open(connection_data_file, "r") as f:
            cfg = json.load(f)
        iot_config = IotConfigJson(cfg)

        model = LinearAdvectionInjectionProduction(problem_setup)

        # Start the monitoring process with a Costa PhysicalDecive that will
        # communicate with Azure.
        with PorePyPhysicalDevice("porepy_lab", iot_config, model) as device:
            # This seems always to be necessary
            device.emit_clean()
            while True:
                try:
                    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                    print(now)
                    time.sleep(60)

                except KeyboardInterrupt:
                    break
