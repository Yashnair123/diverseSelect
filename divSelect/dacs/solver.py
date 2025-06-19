from . import dacs_core
from dataclasses import dataclass
from typing import Union, Tuple
import cvxpy as cp
import numpy as np


@dataclass
class PGDSolver:
    step: float =1.0
    theta: float =1.0
    beta: float =0.5
    max_iters: int =1000
    abs_tol: float =1e-6
    rel_tol: float =1e-6
    debug: bool =False
    solver: Union[dacs_core.PGDSolver, None] =None

    @property
    def n_backtracks(self):
        return self.solver.n_backtracks

    @property
    def iterates(self):
        return self.solver.iterates

    def prox(
        self,
        y: np.ndarray,
        t: float =1.0,
    ):
        out = np.empty(y.size, dtype=y.dtype)
        assert y.size == self.solver.n
        self.solver.prox(y, t, out)
        return out

    def gradient(
        self,
        x: np.ndarray,
    ):
        out = np.empty(x.size, dtype=x.dtype)
        assert x.size == self.solver.n
        self.solver.gradient(x, out)
        return out

    def objective(
        self,
        x: np.ndarray,
        grad_x: Union[np.ndarray, None] =None,
    ):
        if grad_x is None:
            grad_x = self.gradient(x)
        assert x.size == self.solver.n
        assert grad_x.size == self.solver.n
        return self.solver.objective(x, grad_x)


@dataclass
class _SharpePGDSolver:
    S: np.ndarray
    C: float


@dataclass
class SharpePGDSolver(PGDSolver, _SharpePGDSolver):
    def __post_init__(self):
        self.S = np.ascontiguousarray(self.S)
        self.solver = dacs_core.SharpePGDSolver(
            S=self.S,
            C=self.C,
            step=self.step,
            theta=self.theta,
            beta=self.beta,
            max_iters=self.max_iters,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
            debug=self.debug,
        )

    def prox_cvxpy(
        self,
        y: np.ndarray,
        t: float = 1.0,
    ):
        assert y.size == self.solver.n
        x = cp.Variable(self.solver.n)
        expr = cp.sum_squares(x - y)
        constraints = [
            0 <= x,
            x <= self.C,
            cp.sum(x) == 1,
        ]
        prob = cp.Problem(cp.Minimize(expr), constraints)
        prob.solve()
        return x.value

    def solve_cvxpy(self):
        x = cp.Variable(self.solver.n)
        expr = cp.quad_form(x, self.S)
        constraints = [
            0 <= x,
            x <= self.C,
            cp.sum(x) == 1,
        ]
        prob = cp.Problem(cp.Minimize(expr), constraints)
        prob.solve()
        return x.value

    def solve(
        self, 
        warm_start: Union[np.ndarray, None] =None,
    ):
        if warm_start is None:
            x = np.zeros(self.solver.n, dtype=float)
        else:
            x = warm_start
            x = np.copy(x)
        assert x.size == self.solver.n
        self.solver.solve(x)
        return x


@dataclass    
class _MarkowitzPGDSolver:
    S: np.ndarray
    C: float
    gamma: float


@dataclass
class MarkowitzPGDSolver(PGDSolver, _MarkowitzPGDSolver):
    def __post_init__(self):
        self.S = np.ascontiguousarray(self.S)
        self.solver = dacs_core.MarkowitzPGDSolver(
            S=self.S,
            C=self.C,
            gamma=self.gamma,
            step=self.step,
            theta=self.theta,
            beta=self.beta,
            max_iters=self.max_iters,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
            debug=self.debug,
        )

    def prox_cvxpy(
        self,
        y: np.ndarray,
        t: float =1.0,
    ):
        assert y.size == self.solver.n
        x = cp.Variable(self.solver.n)
        expr = cp.sum_squares(x - y)
        constraints = [
            0 <= x,
            x <= 1,
            x <= self.C * cp.sum(x),
        ]
        prob = cp.Problem(cp.Minimize(expr), constraints)
        prob.solve()
        return x.value

    def solve_cvxpy(self):
        x = cp.Variable(self.solver.n)
        expr = (
            0.5 * self.gamma * cp.quad_form(x, self.S, assume_PSD=True)
            - cp.sum(x)
        )
        constraints = [
            0 <= x,
            x <= 1,
            x <= self.C * cp.sum(x),
        ]
        prob = cp.Problem(cp.Minimize(expr), constraints)
        prob.solve()
        return x.value

    def solve(
        self, 
        warm_start: Union[Tuple, None] =None,
    ):
        if warm_start is None:
            x  = np.zeros(self.solver.n, dtype=float)
        else:
            x = warm_start
            x = np.copy(x)
        assert x.size == self.solver.n
        self.solver.solve(x)
        return x 
