import os
import argparse
import torch
from torchvision.utils import save_image
from train import create_model, load_config
from ddim import GaussianDiffusion
from ema import EMA


def main():
    parser = argparse.ArgumentParser(description='DDIM Sampling for Diffusion Models')
    parser.add_argument("--checkpoint", type=str, required=True, help='Path to model checkpoint')
    parser.add_argument("--config", type=str, default="config.yaml", help='Path to config file')
    parser.add_argument("--output_dir", type=str, default=None, help='Output directory for samples')
    parser.add_argument("--num_samples", type=int, default=64, help='Number of samples to generate')
    parser.add_argument("--batch_size", type=int, default=None, help='Batch size for sampling')
    parser.add_argument("--nrow", type=int, default=8, help='Number of images per row in output grid')
    parser.add_argument("--use_ema", action="store_true", default=True, help='Use EMA weights')
    parser.add_argument("--no_ema", action="store_false", dest="use_ema", help='Do not use EMA weights')
    parser.add_argument("--progress", action="store_true", default=True, help='Show progress bar')
    parser.add_argument("--no_progress", action="store_false", dest="progress", help='Hide progress bar')
    parser.add_argument("--seed", type=int, default=1234, help='Random seed')
    parser.add_argument("--sampling_steps", type=int, default=None, help='DDIM sampling steps (default from config)')
    parser.add_argument("--eta", type=float, default=None, help='DDIM eta parameter (default from config)')
    args = parser.parse_args()

    config = load_config(args.config)
    
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_name = config['data'].get('dataset', 'cifar10').lower()
    if dataset_name not in config.get('datasets', {}):
        raise ValueError(f"Dataset '{dataset_name}' not configured in config file")
    
    dataset_config = config['datasets'][dataset_name]
    image_size = dataset_config['image_size']
    attention_resolutions = dataset_config['attention_resolutions']
    
    output_dir = args.output_dir or os.path.join(config['logging']['save_dir'], dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    model = create_model(
        image_size=image_size,
        num_channels=config['model']['num_channels'],
        num_res_blocks=config['model']['num_res_blocks'],
        channel_mult=None,
        attention_resolutions=attention_resolutions,
        dropout=config['model']['dropout'],
        num_heads=config['model']['num_heads'],
    ).to(device)

    sampling_steps = args.sampling_steps if args.sampling_steps is not None else config['diffusion'].get('sampling_steps', 50)
    eta = args.eta if args.eta is not None else config['diffusion'].get('eta', 0.0)

    diffusion = GaussianDiffusion(
        model=model,
        n_steps=config['diffusion']['diffusion_steps'],
        device=device,
        beta_schedule=config['diffusion'].get('beta_schedule', 'linear'),
        ddim_sampling_steps=sampling_steps,
        ddim_eta=eta,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    ema_state = checkpoint.get("ema_state_dict", None)

    if args.use_ema and ema_state:
        ema = EMA(model, decay=ema_state["decay"])
        ema.load_state_dict(ema_state)
        ema.copy_to(model)
        print("Loaded EMA weights for sampling.")
    elif args.use_ema and not ema_state:
        print("EMA requested but not found in checkpoint, falling back to raw weights.")

    model.eval()

    print("=" * 60)
    print("DDIM Sampling Configuration")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Image size: {image_size}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Sampling steps: {sampling_steps}")
    print(f"Eta: {eta}")
    print(f"Seed: {args.seed}")
    print(f"Using EMA: {args.use_ema and ema_state is not None}")
    print("=" * 60)

    print(f"\nGenerating {args.num_samples} samples using DDIM...")
    samples = diffusion.sample(
        n_samples=args.num_samples,
        channels=3,
        img_size=image_size,
        progress=args.progress,
    )

    samples_vis = (samples + 1) / 2
    samples_vis = samples_vis.clamp(0, 1)
    out_path = os.path.join(output_dir, f"samples_ddim_{args.num_samples}_steps{sampling_steps}_eta{eta}.png")
    save_image(samples_vis, out_path, nrow=args.nrow)
    print(f"\n✅ Saved samples to {out_path}")


if __name__ == "__main__":
    main()
