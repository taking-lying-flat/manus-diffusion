import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import argparse
from tqdm import tqdm

from unet import UNetModel
from ddpm import GaussianDiffusion, get_named_beta_schedule
from ema import EMA
from utils import visualize_denoising, scale_to_01


def create_model(image_size, num_channels, num_res_blocks, channel_mult, attention_resolutions, dropout, num_heads):
    if image_size == 32:
        channel_mult = channel_mult or (1, 2, 2, 2)
    elif image_size == 64:
        channel_mult = channel_mult or (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=3,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_heads=num_heads,
    )


def create_diffusion(diffusion_steps, noise_schedule):
    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    return GaussianDiffusion(
        betas=betas,
        model_mean_type="epsilon",
        model_var_type="fixed_small",
    )


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./output")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_channels", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attention_resolutions", type=str, default="16,8")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_scheduler", action="store_true", default=True, help="use cosine learning rate scheduler")
    parser.add_argument("--no_scheduler", action="store_false", dest="use_scheduler", help="disable learning rate scheduler")
    parser.add_argument("--use_ema", action="store_true", default=True, help="enable EMA weight tracking")
    parser.add_argument("--no_ema", action="store_false", dest="use_ema", help="disable EMA weight tracking")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--sample_interval", type=int, default=10)
    parser.add_argument("--sample_batch_size", type=int, default=64, help="number of images to sample when saving")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(
        image_size=args.image_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        channel_mult=None,
        attention_resolutions=args.attention_resolutions,
        dropout=args.dropout,
        num_heads=args.num_heads,
    ).to(device)

    diffusion = create_diffusion(
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # Learning rate scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if ema and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        print(f"Resumed from epoch {start_epoch}")

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"DDPM Training Configuration")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model parameters: {num_params:,}")
    print(f"Image size: {args.image_size}")
    print(f"Diffusion steps: {args.diffusion_steps}")
    print(f"Noise schedule: {args.noise_schedule}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"LR Scheduler: {args.use_scheduler}")
    print(f"EMA: {args.use_ema} (decay={args.ema_decay})")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    # fixed small batch for visualization/metrics
    vis_batch, _ = next(iter(dataloader))
    vis_batch_size = min(8, vis_batch.shape[0])
    vis_batch = vis_batch[:vis_batch_size].to(device)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, (x, _) in enumerate(pbar):
            x = x.to(device)
            batch_size = x.shape[0]
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)

            optimizer.zero_grad()
            loss = diffusion.training_losses(model, x, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update EMA before optimizer.step() to use pre-update parameters
            if ema:
                ema.update(model)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if step % args.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        avg_loss = total_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch + 1}] Average Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
        
        # Per-epoch validation metrics on a fixed mini-batch (no image saving here)
        model.eval()
        if ema:
            ema.store(model)
            ema.copy_to(model)
        t_vis = torch.full((vis_batch.shape[0],), diffusion.num_timesteps // 2, device=device, dtype=torch.long)
        val_metrics = visualize_denoising(
            diffusion, model, vis_batch, t_vis, save_path=None
        )
        print(f"[Epoch {epoch + 1}] Val (PSNR/SSIM): {val_metrics['psnr']:.3f} / {val_metrics['ssim']:.3f}")
        if ema:
            ema.restore(model)
        model.train()

        if (epoch + 1) % args.sample_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            if ema:
                ema.store(model)
                ema.copy_to(model)
            print("Generating samples...")
            samples, start_noise = diffusion.sample(
                model,
                num_samples=args.sample_batch_size,
                batch_size=args.sample_batch_size,
                sample_size=(3, args.image_size, args.image_size),
                device=device,
                progress=False,
            )
            samples_vis = (samples + 1) / 2
            samples_vis = samples_vis.clamp(0, 1)
            save_image(samples_vis, os.path.join(args.save_dir, f"samples_epoch_{epoch + 1}.png"), nrow=8)
            print(f"Saved samples to {args.save_dir}/samples_epoch_{epoch + 1}.png")

            # before/after comparison (initial noise vs final sample)
            noise_vis = start_noise[: samples_vis.shape[0]]
            noise_vis = scale_to_01(noise_vis)
            comp = torch.cat([noise_vis, samples_vis], dim=0)[:16]
            save_image(comp, os.path.join(args.save_dir, f"compare_noise_epoch_{epoch + 1}.png"), nrow=8)

            # denoising visualization + metrics on a fixed mini-batch
            t_vis = torch.full((vis_batch.shape[0],), diffusion.num_timesteps // 2, device=device, dtype=torch.long)
            metrics = visualize_denoising(
                diffusion, model, vis_batch, t_vis, save_path=os.path.join(args.save_dir, f"denoise_epoch_{epoch + 1}.png")
            )
            print(f"Val metrics (PSNR/SSIM): {metrics['psnr']:.3f} / {metrics['ssim']:.3f}")
            
            if ema:
                ema.restore(model)
            
            # Restore training mode after sampling/evaluation
            model.train()

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "ema_state_dict": ema.state_dict() if ema else None,
            }, os.path.join(args.save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))
        
        # Step learning rate scheduler
        if scheduler:
            scheduler.step()

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "ema_state_dict": ema.state_dict() if ema else None,
    }, os.path.join(args.save_dir, "checkpoint_final.pth"))

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    

if __name__ == "__main__":
    train()
