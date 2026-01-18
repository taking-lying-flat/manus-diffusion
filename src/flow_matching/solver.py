from abc import ABC, abstractmethod
from contextlib import nullcontext
from math import ceil
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchdiffeq import odeint
from flow_matching.manifolds import geodesic, Manifold
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.utils import categorical, get_nearest_times, gradient, ModelWrapper


class Solver(ABC, nn.Module):
    @abstractmethod
    def sample(self, x_0: Tensor = None) -> Tensor:
        ...


class ODESolver(Solver):
    """ODE solver. Methods: euler, dopri5, midpoint, heun3 (per torchdiffeq)."""

    def __init__(self, velocity_model: Union[ModelWrapper, Callable]):
        super().__init__()
        self.velocity_model = velocity_model
        self._step_size_methods = {"euler", "midpoint", "rk4", "heun3"}

    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:
        time_grid = time_grid.to(x_init.device)

        def ode_func(t, x):
            return self.velocity_model(x=x, t=t, **model_extras)

        ode_opts = {}
        if step_size is not None and method in self._step_size_methods:
            ode_opts["step_size"] = step_size

        with torch.set_grad_enabled(enable_grad):
            sol = odeint(
                ode_func,
                x_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        return sol if return_intermediates else sol[-1]

    def compute_likelihood(
        self,
        x_1: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([1.0, 0.0]),
        return_intermediates: bool = False,
        exact_divergence: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:
        assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
        ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

        if not exact_divergence:
            z = (torch.randn_like(x_1).to(x_1.device) < 0) * 2.0 - 1.0

        def ode_func(x, t):
            return self.velocity_model(x=x, t=t, **model_extras)

        def dynamics_func(t, states):
            xt = states[0]
            with torch.set_grad_enabled(True):
                xt.requires_grad_()
                ut = ode_func(xt, t)

                if exact_divergence:
                    div = 0
                    for i in range(ut.flatten(1).shape[1]):
                        g = gradient(ut[:, i], xt, create_graph=True)[:, i]
                        if not enable_grad:
                            g = g.detach()
                        div += g
                else:
                    ut_dot_z = torch.einsum(
                        "ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1)
                    )
                    grad_ut_dot_z = gradient(ut_dot_z, xt, create_graph=enable_grad)
                    div = torch.einsum(
                        "ij,ij->i",
                        grad_ut_dot_z.flatten(start_dim=1),
                        z.flatten(start_dim=1),
                    )

            if not enable_grad:
                ut = ut.detach()
                div = div.detach()
            return ut, div

        y_init = (x_1, torch.zeros(x_1.shape[0], device=x_1.device))
        ode_opts = {}
        if step_size is not None and method in self._step_size_methods:
            ode_opts["step_size"] = step_size

        with torch.set_grad_enabled(enable_grad):
            sol, log_det = odeint(
                dynamics_func,
                y_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        x_source = sol[-1]
        source_log_p = log_p0(x_source)

        if return_intermediates:
            return sol, source_log_p + log_det[-1]
        return sol[-1], source_log_p + log_det[-1]


class MixtureDiscreteEulerSolver(Solver):
    """Discrete solver (CTMC) using Euler-style updates."""
    def __init__(
        self,
        model: ModelWrapper,
        path: MixtureDiscreteProbPath,
        vocabulary_size: int,
        source_distribution_p: Optional[Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size

        if source_distribution_p is not None:
            assert source_distribution_p.shape == torch.Size(
                [vocabulary_size]
            ), f"Source distribution p dimension must match the vocabulary size {vocabulary_size}. Got {source_distribution_p.shape}."

        self.source_distribution_p = source_distribution_p

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        div_free: Union[float, Callable[[float], float]] = 0.0,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:
        if not div_free == 0.0:
            assert (
                self.source_distribution_p is not None
            ), "Source distribution p must be specified in order to add a divergence-free term to the probability velocity."

        time_grid = time_grid.to(device=x_init.device)

        if step_size is None:
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
        else:
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            assert (
                t_final - t_init
            ) > step_size, f"Time interval [time_grid[0], time_grid[-1]] must be larger than step_size. Got a time interval [{t_init}, {t_final}] and step_size {step_size}."

            n_steps = ceil((t_final - t_init) / step_size)
            t_discretization = torch.tensor(
                [t_init + step_size * i for i in range(n_steps)] + [t_final],
                device=x_init.device,
            )

            if return_intermediates:
                order = torch.argsort(time_grid)
                time_grid = get_nearest_times(
                    time_grid=time_grid, t_discretization=t_discretization
                )

        x_t = x_init.clone()
        steps_counter = 0
        res = []

        if return_intermediates:
            res = [x_init.clone()]

        ctx = nullcontext()

        with ctx:
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                p_1t = self.model(x=x_t, t=t.repeat(x_t.shape[0]), **model_extras)
                x_1 = categorical(p_1t.to(dtype=dtype_categorical))

                if i == n_steps - 1:
                    x_t = x_1
                else:
                    scheduler_output = self.path.scheduler(t=t)

                    k_t = scheduler_output.alpha_t
                    d_k_t = scheduler_output.d_alpha_t

                    delta_1 = F.one_hot(x_1, num_classes=self.vocabulary_size).to(
                        k_t.dtype
                    )
                    u = d_k_t / (1 - k_t) * delta_1

                    div_free_t = div_free(t) if callable(div_free) else div_free

                    if div_free_t > 0:
                        p_0 = self.source_distribution_p[(None,) * x_t.dim()]
                        u = u + div_free_t * d_k_t / (k_t * (1 - k_t)) * (
                            (1 - k_t) * p_0 + k_t * delta_1
                        )

                    delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                    u = torch.where(
                        delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                    )

                    intensity = u.sum(dim=-1)
                    mask_jump = torch.rand(
                        size=x_t.shape, device=x_t.device
                    ) < 1 - torch.exp(-h * intensity)

                    if mask_jump.sum() > 0:
                        x_t[mask_jump] = categorical(
                            u[mask_jump].to(dtype=dtype_categorical)
                        )

                steps_counter += 1
                t = t + h

                if return_intermediates and (t in time_grid):
                    res.append(x_t.clone())


        if return_intermediates:
            if step_size is None:
                return torch.stack(res, dim=0)
            return torch.stack(res, dim=0)[order]
        return x_t


class RiemannianODESolver(Solver):
    """Riemannian ODE solver. Methods: euler, midpoint, rk4."""
    def __init__(self, manifold: Manifold, velocity_model: ModelWrapper):
        super().__init__()
        self.manifold = manifold
        self.velocity_model = velocity_model

    def sample(
        self,
        x_init: Tensor,
        step_size: float,
        projx: bool = True,
        proju: bool = True,
        method: str = "euler",
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Tensor:
        step_fns = {
            "euler": _euler_step,
            "midpoint": _midpoint_step,
            "rk4": _rk4_step,
        }
        assert method in step_fns.keys(), f"Unknown method {method}"
        step_fn = step_fns[method]

        def velocity_func(x, t):
            return self.velocity_model(x=x, t=t, **model_extras)

        time_grid = torch.sort(time_grid.to(device=x_init.device)).values

        if step_size is None:
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
        else:
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            assert (
                t_final - t_init
            ) > step_size, f"Time interval [min(time_grid), max(time_grid)] must be larger than step_size. Got a time interval [{t_init}, {t_final}] and step_size {step_size}."

            n_steps = int(torch.ceil(torch.tensor((t_final - t_init) / step_size)))
            t_discretization = torch.tensor(
                [step_size * i for i in range(n_steps)] + [t_final],
                device=x_init.device,
            )

        t0s = t_discretization[:-1]

        if verbose:
            t0s = t0s

        if return_intermediates:
            xts = []
            i_ret = 0

        with torch.set_grad_enabled(enable_grad):
            xt = x_init
            for t0, t1 in zip(t0s, t_discretization[1:]):
                dt = t1 - t0
                xt_next = step_fn(
                    velocity_func,
                    xt,
                    t0,
                    dt,
                    manifold=self.manifold,
                    projx=projx,
                    proju=proju,
                )
                if return_intermediates:
                    while (
                        i_ret < len(time_grid)
                        and t0 <= time_grid[i_ret]
                        and time_grid[i_ret] <= t1
                    ):
                        xts.append(
                            interp(
                                self.manifold,
                                xt,
                                xt_next,
                                t0,
                                t1,
                                time_grid[i_ret],
                            )
                        )
                        i_ret += 1
                xt = xt_next

        if return_intermediates:
            return torch.stack(xts, dim=0)
        return xt


def interp(manifold, xt, xt_next, t, t_next, t_ret):
    return geodesic(manifold, xt, xt_next)(
        (t_ret - t) / (t_next - t).reshape(1)
    ).reshape_as(xt)


def _euler_step(
    velocity_model: Callable,
    xt: Tensor,
    t0: Tensor,
    dt: Tensor,
    manifold: Manifold,
    projx: bool = True,
    proju: bool = True,
) -> Tensor:
    velocity_fn = lambda x, t: (
        manifold.proju(x, velocity_model(x, t)) if proju else velocity_model(x, t)
    )
    projx_fn = lambda x: manifold.projx(x) if projx else x

    vt = velocity_fn(xt, t0)
    xt = xt + dt * vt
    return projx_fn(xt)


def _midpoint_step(
    velocity_model: Callable,
    xt: Tensor,
    t0: Tensor,
    dt: Tensor,
    manifold: Manifold,
    projx: bool = True,
    proju: bool = True,
) -> Tensor:
    velocity_fn = lambda x, t: (
        manifold.proju(x, velocity_model(x, t)) if proju else velocity_model(x, t)
    )
    projx_fn = lambda x: manifold.projx(x) if projx else x

    half_dt = 0.5 * dt
    vt = velocity_fn(xt, t0)
    x_mid = xt + half_dt * vt
    x_mid = projx_fn(x_mid)

    xt = xt + dt * velocity_fn(x_mid, t0 + half_dt)
    return projx_fn(xt)


def _rk4_step(
    velocity_model: Callable,
    xt: Tensor,
    t0: Tensor,
    dt: Tensor,
    manifold: Manifold,
    projx: bool = True,
    proju: bool = True,
) -> Tensor:
    velocity_fn = lambda x, t: (
        manifold.proju(x, velocity_model(x, t)) if proju else velocity_model(x, t)
    )
    projx_fn = lambda x: manifold.projx(x) if projx else x

    k1 = velocity_fn(xt, t0)
    k2 = velocity_fn(projx_fn(xt + dt * k1 / 3), t0 + dt / 3)
    k3 = velocity_fn(projx_fn(xt + dt * (k2 - k1 / 3)), t0 + dt * 2 / 3)
    k4 = velocity_fn(projx_fn(xt + dt * (k1 - k2 + k3)), t0 + dt)

    return projx_fn(xt + (k1 + 3 * (k2 + k3) + k4) * dt * 0.125)

