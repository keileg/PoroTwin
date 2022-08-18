"""Implementation of a linear advection model, wrapped as a Costa PhysicsModel.
"""
from typing import Union, Optional
import numpy as np
import porepy as pp

import scipy.sparse.linalg as spla
import scipy.sparse as sps
from porepy_physical_model import PorePyCostaModel

Parameters = dict[str, Union[float, list[float]]]
Vector = np.ndarray


class LinearAdvectionModel(PorePyCostaModel):
    """Partial implementation of the linear advection physics model.

    The model implements equation assignment (on PorePy AD form), and the methods
    solve and assemble, as specified by the PorePy/Costa API. Also implemented are
    several helper methods that set parameters etc.

    """

    def __init__(self, config: dict) -> None:
        self.variable = "u"

        self.parameter_key = "transport"
        self._config = config

        self._is_discretized = False

    def _injection(self, g: pp.Grid) -> np.ndarray:
        """Source term of injection cell
        Units: m^3 / s
        """
        return np.zeros(g.num_cells)

    def _production(self, g: pp.Grid) -> np.ndarray:
        return np.zeros(g.num_cells)

    def _bc_type(self, g):
        # Define boundary condition on faces
        all_bf, *_ = self._domain_boundary_sides(g)
        return pp.BoundaryCondition(g, all_bf, "neu")

    def _bc_values(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous boundary values."""
        return np.zeros(g.num_faces)

    def _mass_weight(self, g: pp.Grid) -> np.ndarray:
        return np.ones(g.num_cells)

    def solve(
        self, params: Parameters, uprev: np.ndarray, rhs: Optional[np.ndarray] = None
    ):
        """Solve the linear system for the given previous state. Also allow for
        a rhs correction term in addition to the standard injection and production.

        IMPLEMENTATION NOTE: More advanced models, for instance including
        time varying advection fields or non-linear transport terms, can
        be implemented by overriding this method.

        Args:
            params (dict): Parameters with problem specification.
            uprev (np.ndarray): Vector of solution values at the previous time step.
            rhs (np.ndarray, optional): Vector of rhs correction terms. Defaults to
                a zero vector.

        Returns:
            np.ndarray: Solution based on the current state of the model.

        """
        A, b = self.assemble(params, uprev=uprev, rhs=rhs)
        return spla.spsolve(A, b)

    def assemble(
        self,
        params: Parameters,
        uprev: np.ndarray,
        unext: Optional[np.ndarray] = None,
        rhs: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Assemble the linear problem for the given state vectors and return
        the Jacobian matrix and residual vector.

        Args:
            params (Parameters): Dictionary describing solver and problem parameters.
            uprev (np.ndarray): Problem state at the previous time step.
            unext (np.ndarray, optional): Problem state at the current time step. Will
                be specified to solve non-linear problems, or to assemble the residual
                for the current state.
            rhs (np.ndarray, optional): Costa source term, comes in addition to any
                rhs term in the problem definition.

        Returns:
            sps.spmatrix: Jacobian matrix for the given system state.
            np.ndarray: Residual vector for the given system state.

        """
        dt = params["dt"]

        g = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        state = self.mdg.subdomain_data(g)[pp.STATE]

        # Set the previous state to the previous time step.
        state[self.variable] = uprev

        # If the solution at the next step is provided, assign this as the
        # current state. The assembled right hand side vector will then
        # give the residual of the system. If not provided, the current
        # state is set equal to that in the previous time step, which is
        # what we want if doing a forward time step.
        state[pp.ITERATE][self.variable] = uprev if unext is None else unext

        # Add the right hand side term, as specified by the external client.
        # Note that this comes in addition to the injection and production
        # terms in the problem specification, see self._set_equations()
        if rhs is None:
            # If not provided, the extra right hand side term is zero.
            rhs = np.zeros(self.mdg.num_subdomain_cells())
        source_ad = pp.ad.Array(rhs)

        mdg = self.mdg

        # Assume there is a single grids
        g = self.g
        grids = [g]

        adv = pp.ad.UpwindAd(self.parameter_key, grids)
        div = pp.ad.Divergence(grids)
        mass = pp.ad.MassMatrixAd(self.parameter_key, grids).mass

        bc = pp.ad.BoundaryCondition(self.parameter_key, subdomains=grids)

        darcy_flux = mdg.subdomain_data(g)[pp.PARAMETERS][self.parameter_key][
            "darcy_flux"
        ]
        flux_mat = sps.dia_matrix((darcy_flux, 0), (g.num_faces, g.num_faces))
        flux = pp.ad.Matrix(flux_mat, name="flux_scaling")

        U = self._ad_c
        U_prev = U.previous_timestep()

        injection = pp.ad.Array(self._injection(g))
        production = pp.ad.Array(self._production(g)) * U

        source = source_ad + injection + production

        eq = div * (flux * (adv.upwind * U) + adv.bound_transport_dir * bc) - source

        eq += mass * (U - U_prev) / dt

        self._eq_manager.equations = {"eq": eq}

        # This is a linear model, we need only discretize once
        if not self._is_discretized:
            self._eq_manager.discretize(self.mdg)
            self._is_discretized = True

        A, b = self._eq_manager.assemble()
        return A, b

    def _assign_variables(self) -> None:
        """
        Assign primary variables to the (single) subdomain.
        """
        for _, data in self.mdg.subdomains(return_data=True):
            data[pp.PRIMARY_VARIABLES] = {
                self.variable: {"cells": 1},
            }

    def _create_ad_variables(self) -> None:
        """Create the merged variables for potential and mortar flux"""

        self._ad_c = self._eq_manager.merge_variables(
            [(sd, self.variable) for sd in self.mdg.subdomains()]
        )

    def _discretize(self) -> None:
        self._eq_manager.discretize(self.mdg)

    def prepare_simulation(self) -> None:
        super().prepare_simulation()

        g = self.g
        data = self.mdg.subdomain_data(g)
        data[pp.STATE] = {
            self.variable: self.initial_condition(),
            pp.ITERATE: {self.variable: self.initial_condition()},
        }
