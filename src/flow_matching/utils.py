from abc import ABC
from dataclasses import dataclass
import logging
import os
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import Tensor, nn
from torch.nn import functional as F
import random


def get_nearest_times(time_grid: Tensor, t_discretization: Tensor) -> Tensor:
    distances = torch.cdist(
        time_grid.unsqueeze(1),
        t_discretization.unsqueeze(1),
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    nearest_indices = distances.argmin(dim=1)
    return t_discretization[nearest_indices]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    in_channels: int
    image_size: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "mnist": DatasetConfig("mnist", 1, 28, (0.5,), (0.5,)),
    "fashion-mnist": DatasetConfig("fashion-mnist", 1, 28, (0.5,), (0.5,)),
    "cifar-10": DatasetConfig("cifar-10", 3, 32, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "stl-10": DatasetConfig("stl-10", 3, 96, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def get_dataset_config(name: str) -> DatasetConfig:
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset {name}")
    return DATASET_CONFIGS[name]


def get_dataloader(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    data_root: str = "/root/autodl-tmp/data",
):
    spec = get_dataset_config(dataset_name)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ]
    )

    if dataset_name == "mnist":
        dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    elif dataset_name == "fashion-mnist":
        dataset = datasets.FashionMNIST(
            data_root, train=True, download=True, transform=transform
        )
    elif dataset_name == "cifar-10":
        dataset = datasets.CIFAR10(
            data_root, train=True, download=True, transform=transform
        )
    elif dataset_name == "stl-10":
        dataset = datasets.STL10(
            data_root, split="train", download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, spec


def setup_logger(log_dir: Optional[str], log_file: str = "train.log") -> logging.Logger:
    logger = logging.getLogger("DDPM")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.dirname(project_root)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unsqueeze_to_match(source: Tensor, target: Tensor, how: str = "suffix") -> Tensor:
    assert (
        how == "prefix" or how == "suffix"
    ), f"{how} is not supported, only 'prefix' and 'suffix' are supported."

    dim_diff = target.dim() - source.dim()

    for _ in range(dim_diff):
        if how == "prefix":
            source = source.unsqueeze(0)
        elif how == "suffix":
            source = source.unsqueeze(-1)

    return source


def expand_tensor_like(input_tensor: Tensor, expand_to: Tensor) -> Tensor:
    assert input_tensor.ndim == 1, "Input tensor must be a 1d vector."
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"The first (batch_size) dimension must match. Got shape {input_tensor.shape} and {expand_to.shape}."

    dim_diff = expand_to.ndim - input_tensor.ndim

    t_expanded = input_tensor.clone()
    t_expanded = t_expanded.reshape(-1, *([1] * dim_diff))

    return t_expanded.expand_as(expand_to)


def gradient(
    output: Tensor,
    x: Tensor,
    grad_outputs: Optional[Tensor] = None,
    create_graph: bool = False,
) -> Tensor:
    if grad_outputs is None:
        grad_outputs = torch.ones_like(output).detach()
    grad = torch.autograd.grad(
        output, x, grad_outputs=grad_outputs, create_graph=create_graph
    )[0]
    return grad


def categorical(probs: Tensor) -> Tensor:
    return torch.multinomial(probs.flatten(0, -2), 1, replacement=True).view(
        *probs.shape[:-1]
    )


class ModelWrapper(ABC, nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        return self.model(x, t, **extras)


def compute_psnr(pred: Tensor, target: Tensor, max_val: float = 2.0) -> float:
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
    psnr = 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(
        mse + 1e-8
    )
    return psnr.mean().item()


def compute_ssim(pred: Tensor, target: Tensor) -> float:
    pred = (pred + 1) / 2
    target = (target + 1) / 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean().item()


