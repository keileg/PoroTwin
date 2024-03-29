from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import porepy as pp
import scipy.sparse as sps

import scipy.sparse.linalg as spla
from Costa.api import PhysicsModel
import json


Parameters = Dict[str, Union[float, List[float]]]
Vector = np.ndarray


class ParabolicSolver(PhysicsModel):
    """Implements a Costa PhysicsModel for a parabolic problem. PorePy is used
    as a backend for discretization.

    IMPLEMENTATION NOTE: The problem specification is now hardcoded in setup
    functions. An external problem specification would have been better.

    """

    def __init__(self, known_solution: bool, config: Dict) -> None:
        self.variable = "p"
        self.flow_parameter_key = "flow"

        self._known_solution = known_solution
        self._config = config

        self._set_model()
        self._is_discretized = False

        # Keywords for discretizations

    def _set_model(self):
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

            bound_faces = np.where(g.tags["domain_boundary_faces"])[0]
            bc_type = bound_faces.size * ["neu"]

            dir_faces = self._config.get("dir_faces", [])

            for di in dir_faces:
                bc_type[di] = "dir"
            bc = pp.BoundaryCondition(g, faces=bound_faces, cond=bc_type)
            bc_values = np.zeros(g.num_faces)

            # Permeability specification
            perm = 1
            K = pp.SecondOrderTensor(np.ones(g.num_cells) * perm)

            # Parameters for the elliptic term are permeability and boundary condition
            flow_param = {
                "second_order_tensor": K,
                "bc": bc,
                "bc_values": bc_values,
                "mass_weight": self._config.get("mass_weight", np.zeros(g.num_cells)),
            }

            # Set all data
            d.update(
                {
                    pp.PARAMETERS: {self.flow_parameter_key: flow_param},
                    pp.DISCRETIZATION_MATRICES: {self.flow_parameter_key: {}},
                    pp.PRIMARY_VARIABLES: {self.variable: {"cells": 1}},
                }
            )
            # Initial values - these will be overridden by the Costa-related methods
            # if necessary.
            d[pp.STATE] = {
                self.variable: np.zeros(g.num_cells),
                pp.ITERATE: {self.variable: np.zeros(g.num_cells)},
            }

    def _set_equations(self) -> None:
        # Define equations, Ad style. Again, this is overkill, but it will
        # become useful when we get to non-linear problems
        gb = self.gb
        dof_manager = pp.DofManager(gb)
        eq_manager = pp.ad.EquationManager(gb, dof_manager)

        # Assume there is a single grids
        g = gb.grids_of_dimension(self.dim)[0]
        grids = [g]
        p = eq_manager.variable(g, self.variable)

        mpfa = pp.ad.TpfaAd(self.flow_parameter_key, grids)
        mass = pp.ad.MassMatrixAd(self.flow_parameter_key, grids)

        bc = pp.ad.BoundaryCondition(self.flow_parameter_key, grids=grids)

        div = pp.ad.Divergence(grids=grids)

        # Components of the discretization that will be used later.
        self._eq_comp = {
            "mass": mass.mass,
            "div": div,
            "mpfa": mpfa,
            "bc": bc,
            "p": p,
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
        """Discretize and assemble (non-linear) system.

        Parameters:
            dt: Time step. If not provided, an elliptic system will be solved.
            p_prev: State at the previous time step.

        """
        dt = params["dt"]
        t = params["t"]
        alpha = params["ALPHA"]

        g = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        state = self.gb.node_props(g, pp.STATE)

        if p_prev is None:
            p_prev = np.zeros(self.gb.num_cells())

        vec = p_prev if p_now is None else p_now
        state[self.variable] = p_prev
        state[pp.ITERATE][self.variable] = vec

        if source_given is None:
            source_given = np.zeros(self.gb.num_cells())

        pi = np.pi
        xc = g.cell_centers
        if self._known_solution:
            source_known = (
                -2
                * pi
                * (np.cos(2 * pi * t + alpha) + 2 * pi * np.sin(2 * pi + t + alpha))
                * np.cos(2 * pi * xc[0])
            ) * g.cell_volumes
        else:
            source_known = np.zeros(g.num_cells)

        known_sol = self.anasol(params)

        flow_params = self.gb.node_props(g)[pp.PARAMETERS]["flow"]

        mass = self._eq_comp["mass"]
        mpfa = self._eq_comp["mpfa"]
        div = self._eq_comp["div"]
        bc = self._eq_comp["bc"]
        p = self._eq_comp[self.variable]
        p_prev = p.previous_timestep()

        source_ad = pp.ad.Array(source_given + source_known)
        eq = div * (mpfa.flux * p + mpfa.bound_flux * bc) - source_ad
        if dt is not None:
            eq += mass * (p - p_prev) / dt

        self._eq_manager.equations = {"eq": eq}

        # This is a linear model, we need only discretize once
        if not self._is_discretized:
            self._eq_manager.discretize(self.gb)
            self._is_discretized = True

        A, b = self._eq_manager.assemble()

        return A, b

    def anasol(self, params: Parameters) -> Vector:
        # Hard coded analytical solution.
        t = params["t"]
        alpha = params["ALPHA"]
        assert isinstance(t, float)
        assert isinstance(alpha, float)
        pi = np.pi
        g = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        xc = g.cell_centers
        return {"primary": 1 + np.sin(2 * pi * t + alpha) * np.cos(2 * pi * xc[0])}

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
        return state[self.variable]

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
        u = uprev + dx

        for g, d in self.gb:
            flux = d[pp.DISCRETIZATION_MATRICES][self.flow_parameter_key]["flux"] * u
            d[pp.PARAMETERS][self.flow_parameter_key]["darcy_flux"] = flux

        return u

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
        u = uprev + spla.spsolve(A, b)
        for g, d in self.gb:
            flux = d[pp.DISCRETIZATION_MATRICES][self.flow_parameter_key]["flux"] * u
            d[pp.PARAMETERS][self.flow_parameter_key]["darcy_flux"] = flux
        return u
