"""Implementation of a linear advection model, wrapped as a Costa PhysicsModel.
"""
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import porepy as pp

import scipy.sparse.linalg as spla
import scipy.sparse as sps
from Costa.api import PhysicsModel

from parabolic_model import ParabolicSolver


Parameters = Dict[str, Union[float, List[float]]]
Vector = np.ndarray


class LinearAdvection(PhysicsModel):
    """Implementation of the linear advection physics model.

    The class is set to solve a problem defined in Wang et al (SIAM, around 2000),
    with circular motion of a Gaussian blob. To override this behavior, change the
    methods
        create_grid()
        _set_parameters()
        _initial_condition()

    To solve a time step of linear advection, use the method predict, as illustrated
    in the bottom of this file.

    """

    def __init__(self, config: Dict) -> None:
        self.variable = "u"

        self.parameter_key = "transport"
        self._config = config

        self._is_discretized = False
        self._set_model()

    def _set_model(self) -> None:
        self.create_grid()
        self._set_parameters()
        self._set_equations()

    def create_grid(self):
        # Set the simulation model, including parameters.
        # In the future, this information should be read from an input file

        # Grid size and resolution
        dim = 2

        Nx = self._config["Nx"]
        phys_dims = self._config.get("phys_dims", np.ones(2))

        g = pp.CartGrid(Nx, phys_dims)
        g.compute_geometry()

        # Convert to mixed-dimensional grid (this is a bit of an overkill
        # for this problem, but it works).
        gb = pp.GridBucket()
        gb.add_nodes([g])

        self.dim = dim
        self.gb = gb

    def _set_parameters(self) -> None:
        for g, d in self.gb:

            dir_faces = np.where(g.tags["domain_boundary_faces"])[0]
            dir_faces = np.array([])

            bc_type = dir_faces.size * ["dir"]
            bc = pp.BoundaryCondition(g, faces=dir_faces, cond=bc_type)
            bc_values = np.zeros(g.num_faces)

            mass_weight = np.ones(g.num_cells)

            xf = g.face_centers
            velocity = np.vstack((-4 * xf[1], 4 * xf[0]))

            fluxes = np.sum(g.face_normals[:2] * velocity, axis=0)
            fluxes[:] = 0

            injection = np.zeros(g.num_cells)
            production = np.zeros(g.num_cells)
            if "injection" in self._config:

                injection_cell = g.closest_cell(self._config["injection"])
                injection_1d = np.ravel_multi_index(
                    (injection_cell), self._config["Nx"]
                )
                injection[injection_1d] = 1

            if "production" in self._config:
                production_cell = g.closest_cell(self._config["production"])
                production_1d = np.ravel_multi_index(
                    (production_cell), self._config["Nx"]
                )
                production[production_1d] = -1

            d[pp.PRIMARY_VARIABLES] = {self.variable: {"cells": 1}}

            d[pp.PARAMETERS] = {
                self.parameter_key: {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight,
                    "darcy_flux": fluxes,
                    "injection_source": injection,
                    "production_source": production,
                }
            }
            # Initial values - these will be overridden by the Costa-related methods
            # if necessary.
            initial_condition = self._initial_condition(g)
            d[pp.STATE] = {
                self.variable: initial_condition,
                pp.ITERATE: {self.variable: initial_condition.copy()},
            }
            d[pp.DISCRETIZATION_MATRICES] = {"transport": {}}

    def _initial_condition(self, g):

        sigma = 0.0447

        xc = g.cell_centers

        x_center, y_center = -0.25, 0

        # Equation (6.3) in Wang et al
        nominator = -(np.power(xc[0] - x_center, 2) + np.power(xc[1] - y_center, 2))
        denominator = 2 * sigma ** 2

        return np.exp(nominator / denominator)

    def _set_equations(self) -> None:

        gb = self.gb
        dof_manager = pp.DofManager(gb)
        eq_manager = pp.ad.EquationManager(gb, dof_manager)

        # Assume there is a single grids
        g = gb.grids_of_dimension(self.dim)[0]
        grids = [g]
        U = eq_manager.variable(g, self.variable)

        advection = pp.ad.UpwindAd(self.parameter_key, grids)
        div = pp.ad.Divergence(grids)
        mass = pp.ad.MassMatrixAd(self.parameter_key, grids)

        bc = pp.ad.BoundaryCondition(self.parameter_key, grids=grids)

        flux = gb.node_props(g, pp.PARAMETERS)[self.parameter_key]["darcy_flux"]
        flux_mat = sps.dia_matrix((flux, 0), (g.num_faces, g.num_faces))
        flux_ad = pp.ad.Matrix(flux_mat, name="flux_scaling")

        injection = gb.node_props(g, pp.PARAMETERS)[self.parameter_key][
            "injection_source"
        ]
        production = gb.node_props(g, pp.PARAMETERS)[self.parameter_key][
            "production_source"
        ]

        self._eq_comp = {
            "mass": mass.mass,
            "div": div,
            "advection": advection,
            "flux": flux_ad,
            "bc": bc,
            "injection": injection,
            "production": production,
            self.variable: U,
        }
        self._dof_manager = dof_manager
        self._eq_manager = eq_manager

    def _assemble(
        self,
        params: Parameters,
        p_prev: Optional[np.ndarray] = None,
        p_now: Optional[np.ndarray] = None,
        source_given=None,
    ) -> Tuple[sps.spmatrix, np.ndarray]:
        dt = params["dt"]

        g = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        state = self.gb.node_props(g, pp.STATE)

        if p_prev is None:
            p_prev = np.zeros(self.gb.num_cells())

        vec = p_prev if p_now is None else p_now
        state[self.variable] = p_prev
        state[pp.ITERATE][self.variable] = vec

        if source_given is None:
            source_given = np.zeros(self.gb.num_cells())

        mass = self._eq_comp["mass"]
        adv = self._eq_comp["advection"]
        flux = self._eq_comp["flux"]
        div = self._eq_comp["div"]
        bc = self._eq_comp["bc"]
        U = self._eq_comp[self.variable]
        U_prev = U.previous_timestep()

        source_ad = pp.ad.Array(source_given)

        injection = pp.ad.Array(self._eq_comp["injection"])
        production = pp.ad.Array(self._eq_comp["production"]) * U

        eq = (
            div * (flux * (adv.upwind * U) + adv.bound_transport_dir * bc)
            + source_ad
            + injection
            + production
        )

        eq += mass * (U - U_prev) / dt

        self._eq_manager.equations = {"eq": eq}

        # This is a linear model, we need only discretize once
        if not self._is_discretized:
            self._eq_manager.discretize(self.gb)
            self._is_discretized = True

        A, b = self._eq_manager.assemble()
        return A, b

    @property
    def ndof(self) -> int:
        return self.gb.num_cells()

    def dirichlet_dofs(self) -> List[int]:
        # No Dirichlet dofs for a FV method
        return []

    def initial_condition(self, params: Parameters) -> Vector:
        """Return the configured initial condition for a set of parameters."""
        gb = self.gb
        g = gb.grids_of_dimension(gb.dim_max())[0]
        state = gb.node_props(g, pp.STATE)
        return state[self._var_name]

    def predict(self, params: Parameters, uprev: Vector) -> Vector:
        """Make an uncorrected prediction of the next timestep given the
        previous timestep.  This is nothing more than a standard discrete
        timestep method.
        :param params: List of parameters.  By convention the first parameter is
            the timestep.
        :param uprev: Previous timestep.  May be ignored by a stationary solver.
        :return: Prediction of next timestep.
        """
        A, b = self._assemble(params, p_prev=uprev)
        dx = spla.spsolve(A, b)
        return uprev + dx

    def residual(self, params: Parameters, uprev: Vector, unext: Vector) -> Vector:
        """Calculate the residual b - Au given the assumed solution unext.
        :param params: Dictionary of parameters.
            By convention the timestep is named 'dt'.
        :param uprev: Previous timestep.  May be ignored by a stationary solver.
        :param unext: The purported exact or experimental solution.
        :return: The residual b - Au."""
        # Assemble linearized system, return residual vector.
        _, b = self._assemble(params, p_prev=uprev, p_now=unext)
        return b

    def correct(self, params, uprev: Vector, sigma: Vector) -> Vector:
        """Calculate a corrected prediction of the next timestep given
        the previous timestep and a right-hand side perturbation.
        :param params: List of parameters.  By convention the first parameter is
            the timestep.
        :param uprev: Previous timestep.  May be ignored by a stationary solver.
        :param sigma: Right-hand-side perturbation. If equal to zero, this method
            should be equivalent to predict(params, uprev).
        :return: Corrected prediction of next timestep.
        """

        # Assemble, solve and return
        A, b = self._assemble(params=params, p_prev=uprev, source_given=sigma)
        return uprev + spla.spsolve(A, b)


class LinearAdvectionWithGravity(LinearAdvection):
    """Linear advection model, but with a tracer which is heavier than the ambient fluid."""

    def __init__(self, config: Dict = None):
        self.gravity_parameter_key = "gravity"
        super().__init__(config)

        self._config["rho_1"] = config.get("rho_1", 1)
        self._config["rho_2"] = config.get("rho_2", 1)

    def _set_parameters(self) -> None:
        super()._set_parameters()

        for g, d in self.gb:
            nf = g.num_faces
            gravity = np.vstack((np.zeros(nf), np.ones(nf), np.zeros(nf)))

            dir_faces = np.where(g.tags["domain_boundary_faces"])[0]
            bc_type = dir_faces.size * ["dir"]
            bc = pp.BoundaryCondition(g, faces=dir_faces, cond=bc_type)
            bc_values = np.zeros(g.num_faces)

            ng = -np.sum(g.face_normals * gravity, axis=0)
            d[pp.PARAMETERS][self.gravity_parameter_key] = {
                "darcy_flux": ng,
                "bc": bc,
                "bc_values": bc_values,
            }
            d[pp.DISCRETIZATION_MATRICES][self.gravity_parameter_key] = {}

    def _assemble(
        self,
        params: Parameters,
        p_prev: np.ndarray,
        p_now: Optional[np.ndarray] = None,
        source_given: Optional[np.ndarray] = None,
    ) -> Tuple[sps.spmatrix, np.ndarray]:
        dt = params["dt"]

        g = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        state = self.gb.node_props(g, pp.STATE)

        vec = p_prev.copy() if p_now is None else p_now
        state[self.variable] = p_prev
        state[pp.ITERATE][self.variable] = vec

        if source_given is None:
            source_given = np.zeros(self.gb.num_cells())

        mass = self._eq_comp["mass"]
        adv = self._eq_comp["advection"]
        flux = self._eq_comp["flux"]
        div = self._eq_comp["div"]
        bc = self._eq_comp["bc"]
        U = self._eq_comp[self.variable]
        U_prev = U.previous_timestep()

        source_ad = pp.ad.Array(source_given)

        # Advective flux, for the circular field
        advective_flux = flux * (adv.upwind * U)  # + adv.bound_transport_dir * bc

        # Gravity term, which will drag the plume downwards
        rho_1 = self._config["rho_1"]
        rho_2 = self._config["rho_2"]

        grav_adv = pp.ad.UpwindAd(self.gravity_parameter_key, grids=[g])
        non_linear_upwind = grav_adv.upwind * (U * U - U)
        vertical_face_components = g.face_normals[1]
        gravity_term = (
            non_linear_upwind
            * (rho_2 - rho_1)
            * pp.GRAVITY_ACCELERATION
            * vertical_face_components
            * 0
        )

        injection = pp.ad.Array(self._eq_comp["injection"])
        production = pp.ad.Array(self._eq_comp["production"]) * U

        eq = div * (advective_flux) + source_ad - (injection + production)

        eq += mass * (U - U_prev) / dt

        self._eq_manager.equations = {"eq": eq}

        # This is a linear model, we need only discretize once
        if not self._is_discretized:
            self._eq_manager.discretize(self.gb)
            self._is_discretized = True

        A, b = self._eq_manager.assemble()
        return A, b

    def _solve(
        self,
        params: Dict,
        uprev: np.ndarray,
        unext: Optional[np.ndarray] = None,
        source: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        # If unext is provided, make this the new iterate state. If not, pick unext
        # from the previous iterate state.
        if unext is None:
            unext = self._dof_manager.assemble_variable(from_iterate=True)
        else:
            self._dof_manager.distribute_variable(unext, to_iterate=True)

        if source is None:
            source = np.zeros_like(unext)

        # Assemble with the current and previous states
        A, b = self._assemble(params, p_prev=uprev, p_now=unext, source_given=source)
        nrm_b_init = np.linalg.norm(b)
        nrm_b = nrm_b_init

        num_iters = 0
        while nrm_b_init > 1e-20 and nrm_b > nrm_b_init * 1e-12:
            dx = spla.spsolve(A, b)
            self._dof_manager.distribute_variable(dx, additive=True, to_iterate=True)
            unext = self._dof_manager.assemble_variable(from_iterate=True)
            A, b = self._assemble(
                params, p_prev=uprev, p_now=unext, source_given=source
            )
            nrm_b = np.linalg.norm(b)

            num_iters += 1
            if num_iters > 100:
                raise ValueError("Maximum number of non-linear iterations exceeded")

        return unext

    def predict(self, params: Parameters, uprev: Vector) -> Vector:
        """Make an uncorrected prediction of the next timestep given the
        previous timestep.  This is nothing more than a standard discrete
        timestep method.
        :param params: List of parameters.  By convention the first parameter is
            the timestep.
        :param uprev: Previous timestep.  May be ignored by a stationary solver.
        :return: Prediction of next timestep.
        """
        return self._solve(params=params, uprev=uprev)


class LinearAdvectionTwoWells(LinearAdvectionWithGravity):
    def __init__(self, config):

        config.update({"rho_1": 1, "rho_2": 1})

        super().__init__(config)

        self._set_model()

    def _set_model(self) -> None:
        self.create_grid()

        self._set_transport_parameters()

        g = self.gb.grids_of_dimension(self.dim)[0]

        bound_faces = g.tags["domain_boundary_faces"]

        dir_faces = np.where(
            np.abs(g.face_centers[1, bound_faces] - self._config["phys_dims"][1]) < 1e-5
        )[0]
        self._config["dir_faces"] = np.array(dir_faces)

        injection_cell = g.closest_cell(self._config["injection"])
        production_cell = g.closest_cell(self._config["production"])

        injection_1d = np.ravel_multi_index((injection_cell), self._config["Nx"])
        production_1d = np.ravel_multi_index((production_cell), self._config["Nx"])

        source = np.zeros(g.num_cells)
        source[injection_1d] = 1
        source[production_1d] = -1

        prev_sol = np.zeros_like(source)

        # Construct Parabolic solver, which will override the set advection field
        flow_model = ParabolicSolver(False, self._config)
        flow_model.correct(
            {"dt": 1.0, "t": 0.0, "ALPHA": 0.0}, uprev=prev_sol, sigma=source
        )

        g_flow = flow_model.gb.grids_of_dimension(2)[0]
        flux = flow_model.gb.node_props(g_flow, pp.PARAMETERS)[
            flow_model.flow_parameter_key
        ]["darcy_flux"].copy()
        self.pressure = flow_model.gb.node_props(g_flow, pp.STATE)[flow_model.variable]
        self.gb.node_props(g, pp.PARAMETERS)[self.parameter_key]["darcy_flux"] = flux
        self._set_transport_equation()

    def _set_transport_parameters(self):
        super()._set_parameters()

    def _set_transport_equation(self):
        super()._set_equations()

    def _initial_condition(self, g):

        return np.zeros(g.num_cells)


if __name__ == "__main__":

    if True:
        model = LinearAdvectionTwoWells(
            {
                "Nx": [50, 50],
                "phys_dims": [1, 1],
                "injection": [1 / 4, 1 / 3],
                "production": [3 / 4, 1 / 3],
            }
        )

        exp = pp.Exporter(model.gb, "two_wells", folder_name="tmp_viz")

        time_steps = [0]

        T = 0.5
        n_steps = 100
        params = {"dt": T / n_steps}

        U = model._initial_condition(model.gb.grids_of_dimension(2)[0])
        exp.write_vtu(["U"], time_dependent=True, time_step=0)

        for i in range(n_steps):
            Up = U.copy()
            U = model.predict(params, uprev=Up)

            if i > 0 and i % 5 == 0:
                print(f"Time step {i}")
                gb = model.gb
                g = model.gb.grids_of_dimension(2)[0]
                state = gb.node_props(g, pp.STATE)
                state["U"] = U
                exp.write_vtu(["U"], time_dependent=True, time_step=i)

                time_steps.append(i)

        exp.write_pvd(time_steps)

    if False:
        T = 0.1
        n_steps = 100
        params = {"dt": T / n_steps}
        model_no_gravity = LinearAdvectionWithGravity(params={"rho_1": 1, "rho_2": 1})
        model_gravity = LinearAdvectionWithGravity(params={"rho_1": 1, "rho_2": 1.03})

        U_no_gravity = model_no_gravity._initial_condition(
            model_no_gravity.gb.grids_of_dimension(2)[0]
        )
        U_gravity = U_no_gravity.copy()

        model_e = LinearAdvection()

        exp = pp.Exporter(model_e.gb, "gravity", folder_name="tmp_viz")
        time_steps = []

        for i in range(n_steps):
            Up_ng = U_no_gravity.copy()
            U_no_gravity = model_no_gravity.predict(params, uprev=Up_ng)

            Up_g = U_gravity.copy()
            U_gravity = model_gravity.predict(params, uprev=Up_g)

            if i % 10 == 0:
                print(f"Time step {i}")
                gb = model_e.gb
                g = model_e.gb.grids_of_dimension(2)[0]
                state = gb.node_props(g, pp.STATE)
                state["no_grav"] = U_no_gravity
                state["grav"] = U_gravity
                exp.write_vtu(["no_grav", "grav"], time_dependent=True, time_step=i)

                time_steps.append(i)

        exp.write_pvd(time_steps)
