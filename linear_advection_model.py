"""
"""
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import porepy as pp

import scipy.sparse.linalg as spla
import scipy.sparse as sps
from Costa.api import PhysicsModel


Parameters = Dict[str, Union[float, List[float]]]
Vector = np.ndarray


class LinearAdvection(PhysicsModel):
    def __init__(self) -> None:
        self.variable = "u"

        self.parameter_key = "transport"

        self._is_discretized = False
        self._set_model()

    def _set_model(self) -> None:
        self.create_grid()
        self._set_parameters()
        self._set_equations()

    def create_grid(self) -> None:
        # Grid size and resolution
        dim = 2

        Nx = [100, 100]
        phys_dims = np.array([1, 1])
        g = pp.CartGrid(Nx, phys_dims)
        g.nodes[:2] -= 0.5
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

            bc_type = dir_faces.size * ["dir"]
            bc = pp.BoundaryCondition(g, faces=dir_faces, cond=bc_type)
            bc_values = np.zeros(g.num_faces)

            mass_weight = np.ones(g.num_cells)

            xf = g.face_centers
            velocity = np.vstack((-4 * xf[1], 4 * xf[0]))

            fluxes = np.sum(g.face_normals[:2] * velocity, axis=0)

            d[pp.PRIMARY_VARIABLES] = {self.variable: {"cells": 1}}

            d[pp.PARAMETERS] = {
                self.parameter_key: {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight,
                    "darcy_flux": fluxes,
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

        U_prev = U.previous_timestep()

        advection = pp.ad.UpwindAd(self.parameter_key, grids)
        div = pp.ad.Divergence(grids)
        mass = pp.ad.MassMatrixAd(self.parameter_key, grids)

        bc = pp.ad.BoundaryCondition(self.parameter_key, grids=grids)

        flux = gb.node_props(g, pp.PARAMETERS)[self.parameter_key]["darcy_flux"]
        flux_mat = sps.dia_matrix((flux, 0), (g.num_faces, g.num_faces))
        flux_ad = pp.ad.Matrix(flux_mat, name="flux_scaling")

        self._eq_comp = {
            "mass": mass.mass,
            "div": div,
            "advection": advection,
            "flux": flux_ad,
            "bc": bc,
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
        eq = div * (flux * (adv.upwind * U) + adv.bound_transport_dir * bc) + source_ad

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


if True:
    params = {"dt": 0.001}
    model = LinearAdvection()
    U = model._initial_condition(model.gb.grids_of_dimension(2)[0])

    exp = pp.Exporter(model.gb, "advection", folder_name="tmp_viz")
    time_steps = []

    for i in range(1000):
        Up = U.copy()
        U = model.predict(params, uprev=Up)

        if i % 100 == 0:
            exp.write_vtu([model.variable], time_dependent=True, time_step=i)
            time_steps.append(i)

    exp.write_pvd(time_steps)
