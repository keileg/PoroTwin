"""This model contains two classes intended to make a bridge from a Costa PhysicsModel
to a PorePy 
"""
import abc
from typing import Union, Optional

import numpy as np
import porepy as pp
import scipy.sparse as sps

from Costa.api import PhysicsModel
from Costa.iot import IotConfig, PhysicalDevice

Parameters = dict[str, Union[float, list[float]]]


class PorePyCostaModel(pp.models.abstract_model.AbstractModel):
    """Extension of a PorePy model class which also has methods needed to
    be part of a COSTA implementation.

    Compared to the PorePy AbstractModel, this class prescribes three additional methods
    that should be implemented: initial_condition, assemble an solve.

    In addition, this class also provides implementation of a few methods that are
    expected to be common to all subclasses.

    """

    @abc.abstractmethod
    def initial_contition(self) -> np.ndarray:
        """Get the initial condition.

        Returns:
            np.ndarray: The initial condition for the given problem.

        """

    @abc.abstractmethod
    def assemble(
        self,
        params: Parameters,
        uprev: np.ndarray,
        unext: Optional[np.ndarray],
        rhs: Optional[np.ndarray],
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Assemble the linear(ized) problem for the given state vectors and return
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

    @abc.abstractmethod
    def solve(
        self, params: Parameters, uprev: np.ndarray, rhs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Solve the problem for the given state vectors.

        Args:
            params (Parameters): Dictionary describing solver and problem parameters.
            uprev (np.ndarray): Problem state at the previous time step.
            rhs (np.ndarray, optional): Costa source term, comes in addition to any
                rhs term in the problem definition.

        Returns:
            np.ndarray: The solution to the stated problem. Given as an update to uprev,
                thus the new state is uprev + solution.

                The choice of an incremental form is dictated by PorePy's AD
                formulation.

        """

    @abc.abstractmethod
    def control(self, params: Parameters) -> dict:
        """ """

    def prepare_simulation(self) -> None:
        self.create_grid()

        self._assign_variables()
        self._create_dof_and_eq_manager()
        self._create_ad_variables()
        self.initial_condition()

        self._set_parameters()

        self._discretize()

    def _assign_variables(self) -> None:
        raise NotImplementedError()

    def _create_dof_and_eq_manager(self) -> None:
        """Create a dof_manager and eq_manager based on a mixed-dimensional grid"""
        self.dof_manager = pp.DofManager(self.mdg)
        self._eq_manager = pp.ad.EquationManager(self.mdg, self.dof_manager)

    def _create_ad_variables(self) -> None:
        raise NotImplementedError()


class PorePyPhysicsModel(PhysicsModel):
    """Implements the Costa API for a PhysicsModel.

    The actual problem specification and its discretization is handled
    by a PorePyCostaModel.

    IMPLEMENTATION NOTE: This class act as a very shallow wrapper around
    the PorePyCostaModel, and we could have used one implementation using
    double inheritance. However, experience shows this very soon leads to
    complications, so the choice was made to go for a less intertwined,
    though perhaps structurally less clean, implementation.

    """

    def __init__(
        self, name: str, config: IotConfig, model: PorePyCostaModel, porepy_params: dict
    ) -> None:
        self._model = model
        self._config = porepy_params

        self._model.prepare_simulation()

    @property
    def ndof(self) -> int:
        return self._model.mdg.num_cells()

    def dirichlet_dofs(self) -> list[int]:
        # No Dirichlet dofs for a FV method
        return []

    def predict(self, params: Parameters, uprev: np.ndarray) -> np.ndarray:
        """Make an uncorrected prediction of the next timestep given the
        previous timestep.  This is nothing more than a standard discrete
        timestep method.

        Args:
            params: List of parameters.  By convention the first parameter is
                the timestep.
            uprev: Solution at previous timestep. May be ignored by a stationary solver.

        Returns:
            np.ndarray: The new state, given the previous state.

        """
        return self._model.solve(params, uprev=uprev)

    def residual(
        self, params: Parameters, uprev: np.ndarray, unext: np.ndarray
    ) -> np.ndarray:
        """Calculate the residual b - Au given the assumed solution unext.

        Args:
            params: List of parameters.  By convention the first parameter is
                the timestep.
            uprev: Solution at previous timestep. May be ignored by a stationary solver.
            unext: Solution at current timestep.

        Returns:
            np.ndarray: The residual.

        """
        # Assemble linearized system, return residual vector.
        _, b = self._model.assemble(params, u_prev=uprev, u_now=unext)
        return b

    def correct(
        self, params: Parameters, uprev: np.ndarray, sigma: np.ndarray
    ) -> np.ndarray:
        """Calculate a corrected prediction of the next timestep given
        the previous timestep and a right-hand side perturbation.

        Args:
            params: List of parameters.  By convention the first parameter is
                the timestep.
            uprev: Solution at previous timestep. May be ignored by a stationary solver.
            sigma: Right-hand-side perturbation from Costa. If equal to zero, this
                method should be equivalent to predict(params, uprev).

        """
        return self._model.solve(params, uprev=uprev, rhs=sigma)

    def initial_condition(self, params):
        return self._model.initial_contition()

    def control(self, payload) -> dict[str, bool]:
        return self._model.control(payload)

    def qi(self, params: dict, u: np.ndarray, name: str) -> float:
        return 0


class PorePyPhysicalDevice(PhysicalDevice):
    def __init__(self, name: str, config: IotConfig, model: PorePyCostaModel) -> None:
        super().__init__(name, config)
        self._model = model

        self._model.prepare_simulation()

        self.control_params = {}

    def on_control(self, payload: dict) -> dict:
        self.control_params = payload["params"]
        self.do_timestep()
        return {"success": True}

    def wait_poll(self):
        if self.require_emit:
            self.send_state()

    def do_timestep(self):
        self._model.on_control(self.control_params)
        self.state += self._model.solve({"dt": 1}, self.state)
        self.require_emit = True

    def send_state(self):
        #        self.emit_state({}, "pressure", self.state["pressure"])
        self.emit_state({}, "concentration", self.state)
        self.emit_refreshed()
        self.require_emit = False
