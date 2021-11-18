from typing import List, Dict, Union, Optional

import numpy as np
import porepy as pp

import scipy.sparse.linalg as spla
from Costa.api import PhysicsModel


Parameters = Dict[str, Union[float, List[float]]]
Vector = np.ndarray


class ParabolicSolver(PhysicsModel):
    """Implements a Costa PhysicsModel for a parabolic problem. PorePy is used
    as a backend for discretization.

    IMPLEMENTATION NOTE: The problem specification is now hardcoded in setup
    functions. An external problem specification would have been better.

    """

    def __init__(self, known_solution: bool) -> None:
        self._var_name = "p"

        self._known_solution = known_solution

        self._set_model()
        self._is_discretized = False

    def _set_model(self):
        # Set the simulation model, including parameters.
        # In the future, this information should be read from an input file

        # Grid size and resolution
        dim = 1

        if dim == 1:
            Nx = [20]
            phys_dims = np.array([1])
        else:
            Nx = [10, 10]
            phys_dims = np.array([1, 1])
        g = pp.CartGrid(Nx, phys_dims)
        g.compute_geometry()

        # Convert to mixed-dimensional grid (this is a bit of an overkill
        # for this problem, but it works).
        gb = pp.GridBucket()
        gb.add_nodes([g])
        data = gb.node_props(g)

        self._gb = gb

        # Keywords for discretizations
        flow_key = "flow"
        mass_key = "mass"

        tol = 1e-6
        if dim == 1:
            dir_faces = np.where(g.tags["domain_boundary_faces"])[0]
        else:
            dir_faces = np.where(np.abs(g.face_centers[1]) < tol)[0]

        # Boundary conditions are specified by their type and numerical value
        bc_type = dir_faces.size * ["dir"]
        bc = pp.BoundaryCondition(g, faces=dir_faces, cond=bc_type)
        bc_values = np.zeros(g.num_faces)

        # Permeability specification
        perm = 1
        K = pp.SecondOrderTensor(np.ones(g.num_cells) * perm)

        # Parameters for the elliptic term are permeability and boundary condition
        flow_param = {"second_order_tensor": K, "bc": bc, "bc_values": bc_values}
        mass_param = {"mass_weight": 1 * np.ones(g.num_cells)}

        # Set all data
        data.update(
            {
                pp.PARAMETERS: {flow_key: flow_param, mass_key: mass_param},
                pp.DISCRETIZATION_MATRICES: {flow_key: {}, mass_key: {}},
                pp.PRIMARY_VARIABLES: {self._var_name: {"cells": 1}},
            }
        )
        # Initial values - these will be overridden by the Costa-related methods
        # if necessary.
        data[pp.STATE] = {
            self._var_name: np.zeros(g.num_cells),
            pp.ITERATE: {self._var_name: np.zeros(g.num_cells)},
        }

        # Define equations, Ad style. Again, this is overkill, but it will
        # become useful when we get to non-linear problems
        grids = [g]

        dof_manager = pp.DofManager(gb)
        eq_manager = pp.ad.EquationManager(gb, dof_manager)

        p = eq_manager.variable(g, self._var_name)

        mpfa = pp.ad.TpfaAd(flow_key, grids)
        mass = pp.ad.MassMatrixAd(mass_key, grids)

        bc = pp.ad.BoundaryCondition(flow_key, grids=grids)

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
    ):
        """Discretize and assemble (non-linear) system.

        Parameters:
            dt: Time step. If not provided, an elliptic system will be solved.
            p_prev: State at the previous time step.

        """
        dt = params["dt"]
        t = params["t"]
        alpha = params["ALPHA"]

        g = self._gb.grids_of_dimension(self._gb.dim_max())[0]
        state = self._gb.node_props(g, pp.STATE)

        if p_prev is None:
            p_prev = np.zeros(self._gb.num_cells())

        vec = p_prev if p_now is None else p_now
        state[self._var_name] = p_prev
        state[pp.ITERATE][self._var_name] = vec

        if source_given is None:
            source_given = np.zeros(self._gb.num_cells())

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

        flow_params = self._gb.node_props(g)[pp.PARAMETERS]["flow"]
        flow_params["bc_values"][0] = known_sol[0]
        flow_params["bc_values"][-1] = known_sol[-1]

        mass = self._eq_comp["mass"]
        mpfa = self._eq_comp["mpfa"]
        div = self._eq_comp["div"]
        bc = self._eq_comp["bc"]
        p = self._eq_comp[self._var_name]
        p_prev = p.previous_timestep()

        source_ad = pp.ad.Array(source_given + source_known)
        eq = div * (mpfa.flux * p + mpfa.bound_flux * bc) + source_ad
        if dt is not None:
            eq += mass * (p - p_prev) / dt

        self._eq_manager.equations = {"eq": eq}

        # This is a linear model, we need only discretize once
        if not self._is_discretized:
            self._eq_manager.discretize(self._gb)
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
        g = self._gb.grids_of_dimension(self._gb.dim_max())[0]
        xc = g.cell_centers
        return {"primary": 1 + np.sin(2 * pi * t + alpha) * np.cos(2 * pi * xc[0])}

    @property
    def ndof(self) -> int:
        return self._gb.num_cells()

    def dirichlet_dofs(self) -> List[int]:
        # No Dirichlet dofs for a FV method
        return []

    def initial_condition(self, params: Parameters) -> Vector:
        """Return the configured initial condition for a set of parameters."""
        gb = self._gb
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
