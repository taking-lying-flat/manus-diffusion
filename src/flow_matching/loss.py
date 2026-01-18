import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from flow_matching.path import MixtureDiscreteProbPath


class MixturePathGeneralizedKL(_Loss):
    def __init__(self, path: MixtureDiscreteProbPath, reduction: str = "mean") -> None:
        super().__init__(None, None, reduction)
        self.path = path

    def forward(self, logits: Tensor, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        x_1_shape = x_1.shape

        log_p_1t = torch.log_softmax(logits, dim=-1)
        log_p_1t_x1 = torch.gather(log_p_1t, dim=-1, index=x_1.unsqueeze(-1))
        log_p_1t_x1 = log_p_1t_x1.view(*x_1_shape)

        p_1t = torch.exp(log_p_1t)
        p_1t_xt = torch.gather(p_1t, dim=-1, index=x_t.unsqueeze(-1))
        p_1t_xt = p_1t_xt.view(*x_1_shape)

        scheduler_output = self.path.scheduler(t)

        jump_coefficient = (
            scheduler_output.d_alpha_t / (1 - scheduler_output.alpha_t)
        )[(...,) + (None,) * (x_1.dim() - 1)]
        jump_coefficient = jump_coefficient.repeat(1, *x_1_shape[1:])
        delta_x1_xt = (x_t == x_1).to(log_p_1t.dtype)

        loss = -jump_coefficient * (
            p_1t_xt - delta_x1_xt + (1 - delta_x1_xt) * log_p_1t_x1
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"{self.reduction} is not a valid value for reduction")


