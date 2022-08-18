"""The module contains a setup for a two well system, with one injector and one
producer.
"""
from typing import Union
import numpy as np
import porepy as pp

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


class LinearAdvectionTwoWells(LinearAdvectionModel):
    """Full setup of a linear advection problem for a two-well system.

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

    def _injection(self, g: pp.Grid) -> np.ndarray:
        """Source term of injection cell
        Units: m^3 / s
        """
        injection_cell = g.closest_cell(self.params["injection"])
        src = np.zeros(g.num_cells)
        src[injection_cell] = self.params[
            "injection_rate"
        ]  # / g.cell_volumes[injection_cell]
        return src

    def _production(self, g: pp.Grid) -> np.ndarray:
        src = np.zeros(g.num_cells)
        production_cell = g.closest_cell(self.params["production"])
        src[production_cell] = self.params[
            "production_rate"
        ]  # / g.cell_volumes[production_cell]
        return src

    def update_flow_field(self) -> None:
        """Calculate the flow field, calling an incompressible flow model.

        The method can be called from external clients to update the flow field, e.g.,
        if the rates have changed.

        """

        params = self.params.copy()
        params["file_name"] = "flow_solution"
        params["folder_name"] = "tmp_flow"

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

    def initial_condition(self):

        g = self.mdg.subdomains(dim=self.mdg.dim_max())[0]

        return np.zeros(g.num_cells)

    def _create_dof_and_eq_manager(self) -> None:
        """Create a dof_manager and eq_manager based on a mixed-dimensional grid"""
        self.dof_manager = pp.DofManager(self.mdg)
        self._eq_manager = pp.ad.EquationManager(self.mdg, self.dof_manager)


class FlowModel(pp.IncompressibleFlow):
    """Incompressible flow model for the two well system. Used to calculate the flow
    field.
    """

    def __init__(self, params: dict) -> None:
        super().__init__(params=params)
        self.params = params

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
        injection_cell = g.closest_cell(self.params["injection"])
        production_cell = g.closest_cell(self.params["production"])

        src = np.zeros(g.num_cells)
        src[injection_cell] = self.params["injection_rate"]
        src[production_cell] = self.params["production_rate"]

        return src


if __name__ == "__main__":
    #
    model = LinearAdvectionTwoWells(
        {
            "Nx": [50, 50],
            "phys_dims": [1, 1],
            "injection": np.reshape([1 / 4, 1 / 3], (-1, 1)),
            "production": np.reshape([3 / 4, 1 / 3], (-1, 1)),
            "injection_rate": 1,
            "production_rate": -1,
            "permeability": 1,
        }
    )
    exp = pp.Exporter(model.mdg, file_name="two_wells", folder_name="tmp_viz")

    time_steps = [0]

    T = 1
    n_steps = 100
    params = {"dt": T / n_steps}

    U = model.initial_condition()
    g = model.g
    model.mdg.subdomain_data(g)[pp.STATE][model.variable] = U

    exp.write_vtu([model.variable], time_dependent=True, time_step=0)

    for i in range(n_steps):
        Up = U.copy()
        # The AD formulation of PorePy solves for the update of U, thus interpret
        # the result as an update.
        U += model.solve(params, uprev=Up)

        model.dof_manager.distribute_variable(U)

        if i > 0 and i % 5 == 0:
            print(f"Time step {i}")
            mdg = model.mdg
            g = model.mdg.subdomains(dim=2)[0]
            state = mdg.subdomain_data(g)[pp.STATE]
            state[model.variable] = U
            exp.write_vtu([model.variable], time_step=i)

            time_steps.append(i)

    exp.write_pvd(time_steps)
