from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import jvp, vmap

from flow_matching.manifolds import geodesic, Manifold
from flow_matching.scheduler import CondOTScheduler, Scheduler
from flow_matching.utils import expand_tensor_like, unsqueeze_to_match


@dataclass
class PathSample:
    x_1: Tensor = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: Tensor = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: Tensor = field(metadata={"help": "time samples t (batch_size, ...)."})
    x_t: Tensor = field(
        metadata={"help": "samples x_t ~ p_t(X_t), shape (batch_size, ...)."}
    )
    dx_t: Tensor = field(
        metadata={"help": "conditional target dX_t, shape: (batch_size, ...)."}
    )


@dataclass
class DiscretePathSample:
    x_1: Tensor = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: Tensor = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: Tensor = field(metadata={"help": "time samples t (batch_size, ...)."})
    x_t: Tensor = field(
        metadata={"help": "samples X_t ~ p_t(X_t), shape (batch_size, ...)."}
    )


class ProbPath(ABC):
    """Abstract class, representing a probability path."""

    @abstractmethod
    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        ...

    def assert_sample_shape(self, x_0: Tensor, x_1: Tensor, t: Tensor):
        assert (
            t.ndim == 1
        ), f"The time vector t must have shape [batch_size]. Got {t.shape}."
        assert (
            t.shape[0] == x_0.shape[0] == x_1.shape[0]
        ), f"Time t dimension must match the batch size [{x_1.shape[0]}]. Got {t.shape}"


class AffineProbPath(ProbPath):
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        scheduler_output = self.scheduler(t)

        alpha_t = expand_tensor_like(
            input_tensor=scheduler_output.alpha_t, expand_to=x_1
        )
        sigma_t = expand_tensor_like(
            input_tensor=scheduler_output.sigma_t, expand_to=x_1
        )
        d_alpha_t = expand_tensor_like(
            input_tensor=scheduler_output.d_alpha_t, expand_to=x_1
        )
        d_sigma_t = expand_tensor_like(
            input_tensor=scheduler_output.d_sigma_t, expand_to=x_1
        )

        x_t = sigma_t * x_0 + alpha_t * x_1
        dx_t = d_sigma_t * x_0 + d_alpha_t * x_1

        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)

    def target_to_velocity(self, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = d_sigma_t / sigma_t
        b_t = (d_alpha_t * sigma_t - d_sigma_t * alpha_t) / sigma_t

        return a_t * x_t + b_t * x_1

    def epsilon_to_velocity(self, epsilon: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = d_alpha_t / alpha_t
        b_t = (d_sigma_t * alpha_t - d_alpha_t * sigma_t) / alpha_t

        return a_t * x_t + b_t * epsilon

    def velocity_to_target(self, velocity: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = -d_sigma_t / (d_alpha_t * sigma_t - d_sigma_t * alpha_t)
        b_t = sigma_t / (d_alpha_t * sigma_t - d_sigma_t * alpha_t)

        return a_t * x_t + b_t * velocity

    def epsilon_to_target(self, epsilon: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        sigma_t = scheduler_output.sigma_t

        a_t = 1 / alpha_t
        b_t = -sigma_t / alpha_t

        return a_t * x_t + b_t * epsilon

    def velocity_to_epsilon(self, velocity: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = -d_alpha_t / (d_sigma_t * alpha_t - d_alpha_t * sigma_t)
        b_t = alpha_t / (d_sigma_t * alpha_t - d_alpha_t * sigma_t)

        return a_t * x_t + b_t * velocity

    def target_to_epsilon(self, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        sigma_t = scheduler_output.sigma_t

        a_t = 1 / sigma_t
        b_t = -alpha_t / sigma_t

        return a_t * x_t + b_t * x_1


class CondOTProbPath(AffineProbPath):
    def __init__(self):
        self.scheduler = CondOTScheduler()


class MixtureDiscreteProbPath(ProbPath):
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> DiscretePathSample:
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        sigma_t = self.scheduler(t).sigma_t
        sigma_t = expand_tensor_like(input_tensor=sigma_t, expand_to=x_1)

        source_indices = torch.rand(size=x_1.shape, device=x_1.device) < sigma_t
        x_t = torch.where(condition=source_indices, input=x_0, other=x_1)

        return DiscretePathSample(x_t=x_t, x_1=x_1, x_0=x_0, t=t)

    def posterior_to_velocity(
        self, posterior_logits: Tensor, x_t: Tensor, t: Tensor
    ) -> Tensor:
        posterior = torch.softmax(posterior_logits, dim=-1)
        vocabulary_size = posterior.shape[-1]
        x_t = F.one_hot(x_t, num_classes=vocabulary_size)
        t = unsqueeze_to_match(source=t, target=x_t)

        scheduler_output = self.scheduler(t)

        kappa_t = scheduler_output.alpha_t
        d_kappa_t = scheduler_output.d_alpha_t

        return (d_kappa_t / (1 - kappa_t)) * (posterior - x_t)


class GeodesicProbPath(ProbPath):
    def __init__(self, scheduler: Scheduler, manifold: Manifold):
        self.scheduler = scheduler
        self.manifold = manifold

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)
        t = expand_tensor_like(input_tensor=t, expand_to=x_1[..., 0:1]).clone()

        def cond_u(x_0, x_1, t):
            path = geodesic(self.manifold, x_0, x_1)
            x_t, dx_t = jvp(
                lambda t: path(self.scheduler(t).alpha_t),
                (t,),
                (torch.ones_like(t).to(t),),
            )
            return x_t, dx_t

        x_t, dx_t = vmap(cond_u)(x_0, x_1, t)
        x_t = x_t.reshape_as(x_1)
        dx_t = dx_t.reshape_as(x_1)

        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)


