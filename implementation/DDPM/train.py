import os
import yaml
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

from unet import UNetModel
from ddpm import GaussianDiffusion
from ema import EMA
from utils import visualize_denoising


def setup_logger(log_dir, log_file="train.log"):
    logger = logging.getLogger("DDPM")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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


def create_diffusion(model, diffusion_steps, device, beta_schedule="linear"):
    return GaussianDiffusion(model=model, n_steps=diffusion_steps, device=device, beta_schedule=beta_schedule)


def train(config_path="config.yaml", resume=None):
    config = load_config(config_path)
    torch.manual_seed(config['training']['seed'])
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    logger = setup_logger(log_dir=config['logging']['save_dir'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(
        image_size=config['data']['image_size'],
        num_channels=config['model']['num_channels'],
        num_res_blocks=config['model']['num_res_blocks'],
        channel_mult=None,
        attention_resolutions=config['model']['attention_resolutions'],
        dropout=config['model']['dropout'],
        num_heads=config['model']['num_heads'],
    ).to(device)

    diffusion = create_diffusion(
        model=model,
        diffusion_steps=config['diffusion']['diffusion_steps'],
        device=device,
        beta_schedule=config['diffusion'].get('beta_schedule', 'linear'),
    )

    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'], betas=(0.9, 0.999), weight_decay=0.0)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['training']['num_epochs'], 
        eta_min=1e-6
    )

    ema = EMA(model, decay=config['training']['ema_decay']) if config['training']['use_ema'] else None

    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if ema and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        logger.info(f"♻️  Resumed from epoch {start_epoch}")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=config['data']['data_dir'], train=True, download=True, transform=transform
    )
    dataloader = DataLoader(
        dataset, batch_size=config['training']['batch_size'], shuffle=True,
        num_workers=config['data']['num_workers'], drop_last=True, pin_memory=True
    )

    eval_loader = DataLoader(
        dataset, batch_size=config['training']['batch_size'], shuffle=False,
        num_workers=config['data']['num_workers'], drop_last=False, pin_memory=True
    )
    eval_batches = []
    for idx, (x_eval, _) in enumerate(eval_loader):
        eval_batches.append(x_eval)
        if idx >= 0:
            break
    eval_real = torch.cat(eval_batches, dim=0)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("=" * 60)
    logger.info("🚀 DDPM Training Configuration")
    logger.info("=" * 60)
    logger.info(f"💻 Device: {device}")
    logger.info(f"🔢 Model parameters: {num_params:,}")
    logger.info(f"📐 Image size: {config['data']['image_size']}")
    logger.info(f"⏱️  Diffusion steps: {config['diffusion']['diffusion_steps']}")
    logger.info(f"📊 Beta schedule: {config['diffusion'].get('beta_schedule', 'linear')}")
    logger.info(f"📦 Batch size: {config['training']['batch_size']}")
    logger.info(f"🎯 Learning rate: {config['training']['lr']}")
    logger.info(f"🔄 EMA: {config['training']['use_ema']} (decay={config['training']['ema_decay']})")
    logger.info(f"🏃 Total epochs: {config['training']['num_epochs']}")
    logger.info("=" * 60)

    vis_batch_size = min(8, eval_real.shape[0])
    vis_batch = eval_real[:vis_batch_size].to(device)

    loss_accum = 0.0
    loss_count = 0

    logger.info(f"🎬 Starting training from epoch {start_epoch}...")
    for epoch in range(start_epoch, config['training']['num_epochs']):
        model.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        
        for batch_idx, (x, _) in pbar:
            x = x.to(device)

            optimizer.zero_grad()
            loss = diffusion.loss(x)
            loss.backward()
            optimizer.step()
            if ema:
                ema.update(model)
            loss_accum += loss.item()
            loss_count += 1
            
            avg_loss = loss_accum / loss_count
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        epoch_avg_loss = loss_accum / loss_count
        loss_accum = 0.0
        loss_count = 0

        model.eval()
        if ema:
            ema.store(model)
            ema.copy_to(model)
        
        t_vis = diffusion.num_timesteps // 2
        quick_metrics = visualize_denoising(
            diffusion, vis_batch, t_vis, save_path=None
        )
        logger.info(f"📊 Epoch {epoch + 1}/{config['training']['num_epochs']} | 🎯 Loss: {epoch_avg_loss:.4f} | 📈 PSNR: {quick_metrics['psnr']:.2f} dB | ✨ SSIM: {quick_metrics['ssim']:.4f} | 🔥 LR: {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % config['logging']['sample_interval'] == 0:
            samples = diffusion.sample(
                n_samples=config['logging']['sample_batch_size'],
                channels=3,
                img_size=config['data']['image_size'],
            )
            samples_vis = (samples + 1) / 2
            samples_vis = samples_vis.clamp(0, 1)

            save_image(samples_vis, os.path.join(config['logging']['save_dir'], f"samples_epoch_{epoch + 1}.png"), nrow=8)
            logger.info(f"🖼️  Saved samples to samples_epoch_{epoch + 1}.png")

            detailed_metrics = visualize_denoising(
                diffusion, vis_batch, t_vis, save_path=os.path.join(config['logging']['save_dir'], f"denoise_epoch_{epoch + 1}.png")
            )
            logger.info(f"📸 Saved denoising visualization to denoise_epoch_{epoch + 1}.png")
            logger.info(f"🎨 Detailed Metrics - PSNR: {detailed_metrics['psnr']:.3f} dB | SSIM: {detailed_metrics['ssim']:.4f}")

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_state_dict": ema.state_dict() if ema else None,
            }, os.path.join(config['logging']['save_dir'], f"checkpoint_epoch_{epoch + 1}.pth"))
            logger.info(f"💾 Checkpoint saved: checkpoint_epoch_{epoch + 1}.pth")

        if ema:
            ema.restore(model)
        
        scheduler.step()

    torch.save({
        "epoch": config['training']['num_epochs'],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict() if ema else None,
    }, os.path.join(config['logging']['save_dir'], "checkpoint_final.pth"))

    logger.info("=" * 60)
    logger.info("🎉 Training completed successfully!")
    logger.info("=" * 60)
    

if __name__ == "__main__":
    train(config_path="config.yaml")
