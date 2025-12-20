import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
import logging

from models import Generator, Discriminator
from utils import get_dataset_config, load_data


def setup_logger(log_dir, log_file="train.log"):
    logger = logging.getLogger("GAN")
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


def train(dataset_name, n_epochs=100, sample_epoch_interval=10, num_workers=4, n_gen=2):
    dataset_config = get_dataset_config(dataset_name)
    
    if dataset_config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: mnist, fashion-mnist")
    
    output_dir = f"./output/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger(output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    generator = Generator(
        latent_dim=dataset_config['latent_dim'],
        color_channels=dataset_config['color_channels'],
        pooling_kernels=dataset_config['pooling_kernels'],
        decoder_input_size=dataset_config['decoder_input_size']
    ).to(device)
    
    discriminator = Discriminator(
        color_channels=dataset_config['color_channels'],
        pooling_kernels=dataset_config['pooling_kernels'],
        encoder_output_size=dataset_config['encoder_output_size']
    ).to(device)
    
    # 定义损失函数：使用带 Logits 的二元交叉熵损失
    adversarial_loss = nn.BCEWithLogitsLoss()
    
    # Optimizers - Generator learns faster, Discriminator slower for balance
    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=dataset_config['lr_g'],
        betas=(0.5, 0.999)
    )
    
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=dataset_config['lr_d'],
        betas=(0.5, 0.999)
    )
    
    # Load data
    dataloader = load_data(
        dataset_name,
        batch_size=dataset_config['batch_size'],
        num_workers=num_workers
    )
    
    num_params_g = sum(p.numel() for p in generator.parameters())
    num_params_d = sum(p.numel() for p in discriminator.parameters())
    
    logger.info("=" * 60)
    logger.info("🚀 GAN Training Configuration")
    logger.info("=" * 60)
    logger.info(f"💻 Device: {device}")
    logger.info(f"📊 Dataset: {dataset_name.upper()}")
    logger.info(f"📐 Image size: {dataset_config['img_size']}")
    logger.info(f"🎨 Channels: {dataset_config['color_channels']}")
    logger.info(f"🔢 Latent dim: {dataset_config['latent_dim']}")
    logger.info(f"🧠 Generator params: {num_params_g:,}")
    logger.info(f"🎯 Discriminator params: {num_params_d:,}")
    logger.info(f"📦 Batch size: {dataset_config['batch_size']}")
    logger.info(f"🗂️  Train batches: {len(dataloader)}")
    logger.info(f"🏃 Total epochs: {n_epochs}")
    logger.info(f"📈 Learning rate - G: {dataset_config['lr_g']:.0e} | D: {dataset_config['lr_d']:.0e} (G faster)")
    logger.info(f"⚙️  Optimizer: Adam (betas=(0.5, 0.999))")
    logger.info(f"🔒 Stability: Label smoothing for D (0.9), hard labels for G (1.0)")
    logger.info(f"📉 Loss: BCEWithLogitsLoss (D outputs logits)")
    logger.info(f"🔄 Training ratio: G updates {n_gen}x per D update")
    logger.info(f"💾 Output dir: {output_dir}")
    logger.info(f"📸 Sample interval: every {sample_epoch_interval} epochs")
    logger.info("=" * 60)
    
    fixed_z = torch.randn(64, dataset_config['latent_dim'], device=device)
    
    for epoch in range(n_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_d_real = 0.0
        epoch_d_fake = 0.0
        num_batches = 0
        
        for i, (imgs, _) in enumerate(pbar):
            batch_size = imgs.shape[0]

            # 给真图 0.9 的标签（而不是 1.0），可以防止判别器过于自信，从而让训练更稳定
            valid_smooth = torch.full((batch_size, 1), 0.9, device=device)
            valid_hard = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            
            real_imgs = imgs.to(device)
            
            # ---------------------
            #  Train Discriminator (once per batch)
            # ---------------------
            optimizer_D.zero_grad()

            # 生成随机噪声并产生假图
            z = torch.randn(batch_size, dataset_config['latent_dim'], device=device)
            gen_imgs = generator(z)
            
            real_loss = adversarial_loss(discriminator(real_imgs), valid_smooth)    # 计算真图损失
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)    # 计算假图损失（注意使用 .detach()，防止此处更新生成器）
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator (n_gen times per batch)
            # -----------------
            g_loss_accum = 0.0
            for _ in range(n_gen):
                optimizer_G.zero_grad()

                # 重新采样噪声并生成
                z = torch.randn(batch_size, dataset_config['latent_dim'], device=device)
                gen_imgs = generator(z)
                # 生成器的目标是让判别器认为这张图是 "1" (真图)
                g_loss = adversarial_loss(discriminator(gen_imgs), valid_hard)
                
                g_loss.backward()
                optimizer_G.step()
                g_loss_accum += g_loss.item()
            
            g_loss_avg = g_loss_accum / n_gen
            
            # Monitor D(x) and D(G(z)) for stability tracking
            with torch.no_grad():
                # 计算 D 对真图和假图的平均预测值（经过 Sigmoid 后的概率）
                real_pred = torch.sigmoid(discriminator(real_imgs)).mean().item()
                fake_pred = torch.sigmoid(discriminator(gen_imgs.detach())).mean().item()
            
            epoch_g_loss += g_loss_avg
            epoch_d_loss += d_loss.item()
            epoch_d_real += real_pred
            epoch_d_fake += fake_pred
            num_batches += 1
            
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss_avg:.4f}'
            })
        
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_d_real = epoch_d_real / num_batches
        avg_d_fake = epoch_d_fake / num_batches
        
        logger.info(f"📊 Epoch [{epoch+1}/{n_epochs}] "
                   f"D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f} | "
                   f"D(x): {avg_d_real:.3f} | D(G(z)): {avg_d_fake:.3f}")
        
        if (epoch + 1) % sample_epoch_interval == 0 or epoch == 0:
            with torch.no_grad():
                gen_imgs = generator(fixed_z)
                save_image(
                    gen_imgs.data,
                    f"{output_dir}/epoch_{epoch+1}.png",
                    nrow=8,
                    normalize=True
                )
            logger.info(f"💾 Generated samples saved: epoch_{epoch+1}.png")
    
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
    }, f"{output_dir}/checkpoint_final.pth")
    
    logger.info("=" * 60)
    logger.info("✅ Training completed!")
    logger.info(f"💾 Checkpoint saved: {output_dir}/checkpoint_final.pth")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN on different datasets')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['mnist', 'fashion-mnist'],
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--sample-interval', type=int, default=10, 
                        help='Interval (in epochs) for saving samples')
    parser.add_argument('--num-workers', type=int, default=16, help='Number of data loading workers')
    parser.add_argument('--n-gen', type=int, default=5, 
                        help='Number of Generator updates per Discriminator update')
    args = parser.parse_args()
    
    train(args.dataset, args.epochs, args.sample_interval, args.num_workers, args.n_gen)
