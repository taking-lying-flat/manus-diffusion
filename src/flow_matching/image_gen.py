import argparse
import os
import sys
import torch
import torch.nn as nn
from torchvision import utils as tv_utils

if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flow_matching.path import AffineProbPath, CondOTProbPath
from flow_matching.scheduler import (
    CondOTScheduler,
    CosineScheduler,
    LinearVPScheduler,
    PolynomialConvexScheduler,
    VPScheduler,
)
from flow_matching.solver import ODESolver
from flow_matching.unet import UNetModel
from flow_matching.utils import (
    ModelWrapper,
    compute_psnr,
    compute_ssim,
    get_dataloader,
    set_seed,
    setup_logger,
)

UNET_CONFIGS = {
    "mnist": {
        "base_channels": 64,
        "num_res_blocks": 2,
        "attention_resolutions": (4, 8),
        "dropout": 0.0,
        "channel_mult": (1, 2, 4),
        "num_heads": 1,
    },
    "fashion-mnist": {
        "base_channels": 64,
        "num_res_blocks": 2,
        "attention_resolutions": (4, 8),
        "dropout": 0.0,
        "channel_mult": (1, 2, 4),
        "num_heads": 1,
    },
    "cifar-10": {
        "base_channels": 128,
        "num_res_blocks": 2,
        "attention_resolutions": (4, 8),
        "dropout": 0.1,
        "channel_mult": (1, 2, 4, 8),
        "num_heads": 4,
    },
    "stl-10": {
        "base_channels": 128,
        "num_res_blocks": 2,
        "attention_resolutions": (4, 8, 16),
        "dropout": 0.1,
        "channel_mult": (1, 2, 4, 8),
        "num_heads": 4,
    },
}


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    spec,
    device: torch.device,
    steps: int,
    ode_method: str,
    out_path: str,
    num_samples: int = 64,
) -> None:
    model.eval()
    wrapper = ModelWrapper(model)
    solver = ODESolver(velocity_model=wrapper)

    x_init = torch.randn(
        num_samples, spec.in_channels, spec.image_size, spec.image_size, device=device
    )
    time_grid = torch.tensor([0.0, 1.0], device=device)
    samples = solver.sample(
        x_init=x_init, step_size=1.0 / steps, method=ode_method, time_grid=time_grid
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tv_utils.save_image(samples, out_path, nrow=8, normalize=True, value_range=(-1, 1))


@torch.no_grad()
def generate_samples_tensor(
    model: nn.Module,
    spec,
    device: torch.device,
    steps: int,
    ode_method: str,
    num_samples: int,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    wrapper = ModelWrapper(model)
    solver = ODESolver(velocity_model=wrapper)
    time_grid = torch.tensor([0.0, 1.0], device=device)
    batches = []
    remaining = num_samples
    while remaining > 0:
        current = min(batch_size, remaining)
        x_init = torch.randn(
            current, spec.in_channels, spec.image_size, spec.image_size, device=device
        )
        samples = solver.sample(
            x_init=x_init,
            step_size=1.0 / steps,
            method=ode_method,
            time_grid=time_grid,
        )
        batches.append(samples)
        remaining -= current
    return torch.cat(batches, dim=0)


def collect_real_samples(loader, num_samples: int, device: torch.device) -> torch.Tensor:
    batches = []
    remaining = num_samples
    for x_1, _ in loader:
        x_1 = x_1.to(device)
        current = min(x_1.shape[0], remaining)
        batches.append(x_1[:current])
        remaining -= current
        if remaining <= 0:
            break
    return torch.cat(batches, dim=0)


def build_scheduler(name: str, params: dict):
    if name == "condot":
        return CondOTScheduler()
    if name == "polynomial":
        return PolynomialConvexScheduler(n=params.get("n", 1.0))
    if name == "vp":
        return VPScheduler(
            beta_min=params.get("beta_min", 0.1),
            beta_max=params.get("beta_max", 20.0),
        )
    if name == "linear_vp":
        return LinearVPScheduler()
    if name == "cosine":
        return CosineScheduler()
    raise ValueError(f"Unknown scheduler {name}")


def build_path(args: argparse.Namespace):
    scheduler_params = {
        "n": args.scheduler_n,
        "beta_min": args.scheduler_beta_min,
        "beta_max": args.scheduler_beta_max,
    }
    scheduler = build_scheduler(args.scheduler, scheduler_params)
    name = args.path
    if name == "condot":
        return CondOTProbPath()
    if name == "affine":
        return AffineProbPath(scheduler)
    raise ValueError(f"Unknown path {name}")


def train(args: argparse.Namespace) -> None:
    data_root = "/root/autodl-tmp/data"
    num_workers = 16
    seed = 42
    project_root = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(os.path.dirname(project_root), "runs", args.dataset)
    
    device = torch.device(args.device)
    set_seed(seed)
    logger = setup_logger(None)

    loader, spec = get_dataloader(
        args.dataset, args.batch_size, num_workers, data_root
    )
    unet_cfg = UNET_CONFIGS.get(args.dataset)
    if unet_cfg is None:
        raise ValueError(f"Missing UNet config for dataset {args.dataset}")
    model = UNetModel(
        image_size=spec.image_size,
        in_channels=spec.in_channels,
        model_channels=unet_cfg["base_channels"],
        out_channels=spec.in_channels,
        num_res_blocks=unet_cfg["num_res_blocks"],
        attention_resolutions=unet_cfg["attention_resolutions"],
        dropout=unet_cfg["dropout"],
        channel_mult=unet_cfg["channel_mult"],
        num_heads=unet_cfg["num_heads"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0
    )
    loss_fn = nn.MSELoss()
    path = build_path(args)

    os.makedirs(save_dir, exist_ok=True)
    step = 0

    logger.info("🧪 dataset=%s device=%s save_dir=%s", args.dataset, args.device, save_dir)
    logger.info("🧭 path=%s scheduler=%s", args.path, args.scheduler)
    logger.info("🧮 ode_method=%s", args.ode_method)
    logger.info("📈 lr=%.6f lr_scheduler=cosine", args.lr)
    logger.info("🚀 start training")

    for epoch in range(args.epochs):
        model.train()
        for x_1, _ in loader:
            x_1 = x_1.to(device)
            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.shape[0], device=device)

            path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)
            pred = model(path_sample.x_t, t)
            loss = loss_fn(pred, path_sample.dx_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                logger.info("🧪 epoch=%d step=%d loss=%.6f", epoch, step, loss.item())

            if (
                args.num_generate_steps_interval > 0
                and step % args.num_generate_steps_interval == 0
            ):
                out_path = os.path.join(save_dir, f"samples_step_{step}.png")
                generate_samples(
                    model,
                    spec,
                    device,
                    args.num_sampling_ode_steps,
                    args.ode_method,
                    out_path,
                )

            step += 1

        num_metric_samples = 1000
        metric_batch = min(args.batch_size, 100)
        with torch.no_grad():
            pred = generate_samples_tensor(
                model,
                spec,
                device,
                args.num_sampling_ode_steps,
                args.ode_method,
                num_metric_samples,
                metric_batch,
            )
            target = collect_real_samples(loader, num_metric_samples, device)
            psnr = compute_psnr(pred, target)
            ssim = compute_ssim(pred, target)
        logger.info("📊 epoch=%d psnr=%.4f ssim=%.4f", epoch, psnr, ssim)

        lr_scheduler.step()

        if args.num_save_ckpt_epochs > 0 and (epoch + 1) % args.num_save_ckpt_epochs == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)

    final_ckpt = os.path.join(save_dir, "checkpoint_final.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, final_ckpt)

    if args.generate_after:
        out_path = os.path.join(save_dir, "samples_final.png")
        generate_samples(
            model,
            spec,
            device,
            args.num_sampling_ode_steps,
            args.ode_method,
            out_path,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flow Matching Training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_save_ckpt_epochs", type=int, default=5)
    parser.add_argument("--num_generate_steps_interval", type=int, default=1000)
    parser.add_argument("--num_sampling_ode_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--ode_method",
        type=str,
        default="euler",
        choices=["euler", "dopri5", "midpoint", "heun3"],
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion-mnist", "cifar-10", "stl-10"],
        default="mnist",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="condot",
        choices=["condot", "affine"],
        help="Path type for image generation.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="condot",
        choices=["condot", "polynomial", "vp", "linear_vp", "cosine"],
        help="Scheduler type for affine path.",
    )
    parser.add_argument(
        "--scheduler_n",
        type=float,
        default=1.0,
        help="Polynomial scheduler exponent (for scheduler=polynomial).",
    )
    parser.add_argument(
        "--scheduler_beta_min",
        type=float,
        default=0.1,
        help="VP scheduler beta_min (for scheduler=vp).",
    )
    parser.add_argument(
        "--scheduler_beta_max",
        type=float,
        default=20.0,
        help="VP scheduler beta_max (for scheduler=vp).",
    )
    parser.add_argument(
        "--generate_after",
        action="store_true",
        help="Generate samples after training.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)

