from dataclasses import dataclass, field
from typing import Union

import torch
from torch import Tensor
from flow_matching.utils import ModelWrapper


@dataclass
class SchedulerOutput:
    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    sigma_t: Tensor = field(metadata={"help": "sigma_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Tensor = field(metadata={"help": "Derivative of sigma_t."})


class Scheduler:
    def __call__(self, t: Tensor) -> SchedulerOutput:
        raise NotImplementedError("Scheduler must implement __call__().")

    def snr_inverse(self, snr: Tensor) -> Tensor:
        raise NotImplementedError("Scheduler must implement snr_inverse().")


class CondOTScheduler(Scheduler):
    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-torch.ones_like(t),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        kappa_t = snr / (1.0 + snr)
        return kappa_t


class PolynomialConvexScheduler(Scheduler):
    def __init__(self, n: Union[float, int]) -> None:
        assert isinstance(
            n, (float, int)
        ), f"`n` must be a float or int. Got {type(n)=}."
        assert n > 0, f"`n` must be positive. Got {n=}."

        self.n = n

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t**self.n,
            sigma_t=1 - t**self.n,
            d_alpha_t=self.n * (t ** (self.n - 1)),
            d_sigma_t=-self.n * (t ** (self.n - 1)),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        kappa = snr / (1.0 + snr)
        return torch.pow(kappa, 1.0 / self.n)


class VPScheduler(Scheduler):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__()

    def __call__(self, t: Tensor) -> SchedulerOutput:
        b = self.beta_min
        B = self.beta_max
        T = 0.5 * (1 - t) ** 2 * (B - b) + (1 - t) * b
        dT = -(1 - t) * (B - b) - b

        return SchedulerOutput(
            alpha_t=torch.exp(-0.5 * T),
            sigma_t=torch.sqrt(1 - torch.exp(-T)),
            d_alpha_t=-0.5 * dT * torch.exp(-0.5 * T),
            d_sigma_t=0.5 * dT * torch.exp(-T) / torch.sqrt(1 - torch.exp(-T)),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        T = -torch.log(snr**2 / (snr**2 + 1))
        b = self.beta_min
        B = self.beta_max
        t = 1 - ((-b + torch.sqrt(b**2 + 2 * (B - b) * T)) / (B - b))
        return t


class LinearVPScheduler(Scheduler):
    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=(1 - t**2) ** 0.5,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-t / (1 - t**2) ** 0.5,
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        return torch.sqrt(snr**2 / (1 + snr**2))


class CosineScheduler(Scheduler):
    def __call__(self, t: Tensor) -> SchedulerOutput:
        pi = torch.pi
        return SchedulerOutput(
            alpha_t=torch.sin(pi / 2 * t),
            sigma_t=torch.cos(pi / 2 * t),
            d_alpha_t=pi / 2 * torch.cos(pi / 2 * t),
            d_sigma_t=-pi / 2 * torch.sin(pi / 2 * t),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        return 2.0 * torch.atan(snr) / torch.pi


class ScheduleTransformedModel(ModelWrapper):
    def __init__(
        self,
        velocity_model: ModelWrapper,
        original_scheduler: Scheduler,
        new_scheduler: Scheduler,
    ):
        super().__init__(model=velocity_model)
        self.original_scheduler = original_scheduler
        self.new_scheduler = new_scheduler

        assert hasattr(self.original_scheduler, "snr_inverse") and callable(
            getattr(self.original_scheduler, "snr_inverse")
        ), "The original scheduler must have a callable 'snr_inverse' method."

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        r = t

        r_scheduler_output = self.new_scheduler(t=r)

        alpha_r = r_scheduler_output.alpha_t
        sigma_r = r_scheduler_output.sigma_t
        d_alpha_r = r_scheduler_output.d_alpha_t
        d_sigma_r = r_scheduler_output.d_sigma_t

        t = self.original_scheduler.snr_inverse(alpha_r / sigma_r)

        t_scheduler_output = self.original_scheduler(t=t)

        alpha_t = t_scheduler_output.alpha_t
        sigma_t = t_scheduler_output.sigma_t
        d_alpha_t = t_scheduler_output.d_alpha_t
        d_sigma_t = t_scheduler_output.d_sigma_t

        s_r = sigma_r / sigma_t

        dt_r = (
            sigma_t
            * sigma_t
            * (sigma_r * d_alpha_r - alpha_r * d_sigma_r)
            / (sigma_r * sigma_r * (sigma_t * d_alpha_t - alpha_t * d_sigma_t))
        )

        ds_r = (sigma_t * d_sigma_r - sigma_r * d_sigma_t * dt_r) / (sigma_t * sigma_t)

        u_t = self.model(x=x / s_r, t=t, **extras)
        u_r = ds_r * x / s_r + dt_r * s_r * u_t

        return u_r

