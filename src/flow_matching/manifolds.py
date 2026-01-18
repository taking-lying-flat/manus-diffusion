import abc
import math
from typing import Callable
import torch
from torch import Tensor
import torch.nn as nn


class Manifold(nn.Module, metaclass=abc.ABCMeta):
    """A manifold class that contains projection operations and logarithm and exponential maps."""

    @abc.abstractmethod
    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        r"""Computes exponential map :math:`\exp_x(u)`."""
        raise NotImplementedError

    @abc.abstractmethod
    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Computes logarithmic map :math:`\log_x(y)`."""
        raise NotImplementedError

    @abc.abstractmethod
    def projx(self, x: Tensor) -> Tensor:
        """Project point :math:`x` on the manifold."""
        raise NotImplementedError

    @abc.abstractmethod
    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Project vector :math:`u` on a tangent space for :math:`x`."""
        raise NotImplementedError


class Euclidean(Manifold):
    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        return x + u

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        return y - x

    def projx(self, x: Tensor) -> Tensor:
        return x

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        return u


class Sphere(Manifold):
    EPS = {torch.float32: 1e-4, torch.float64: 1e-7}

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        norm_u = u.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
        retr = self.projx(x + u)
        cond = norm_u > self.EPS[norm_u.dtype]

        return torch.where(cond, exp, retr)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        u = self.proju(x, y - x)
        dist = self.dist(x, y, keepdim=True)
        cond = dist.gt(self.EPS[x.dtype])
        result = torch.where(
            cond,
            u * dist / u.norm(dim=-1, keepdim=True).clamp_min(self.EPS[x.dtype]),
            u,
        )
        return result

    def projx(self, x: Tensor) -> Tensor:
        return x / x.norm(dim=-1, keepdim=True)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        return u - (x * u).sum(dim=-1, keepdim=True) * x

    def dist(self, x: Tensor, y: Tensor, *, keepdim: bool = False) -> Tensor:
        inner = (x * y).sum(-1, keepdim=keepdim)
        return torch.acos(inner)


class FlatTorus(Manifold):
    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        return (x + u) % (2 * math.pi)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.atan2(torch.sin(y - x), torch.cos(y - x))

    def projx(self, x: Tensor) -> Tensor:
        return x % (2 * math.pi)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        return u


def geodesic(
    manifold: Manifold, start_point: Tensor, end_point: Tensor
) -> Callable[[Tensor], Tensor]:
    shooting_tangent_vec = manifold.logmap(start_point, end_point)

    def path(t: Tensor) -> Tensor:
        tangent_vecs = torch.einsum("i,...k->...ik", t, shooting_tangent_vec)
        points_at_time_t = manifold.expmap(start_point.unsqueeze(-2), tangent_vecs)
        return points_at_time_t

    return path


