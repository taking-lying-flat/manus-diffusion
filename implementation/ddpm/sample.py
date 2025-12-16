import os
import argparse
import torch
from torchvision.utils import save_image
from train import create_model, load_config
from ddpm import GaussianDiffusion
from ema import EMA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", action="store_false", dest="use_ema")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no_progress", action="store_false", dest="progress")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    config = load_config(args.config)
    
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = args.output_dir or config['logging']['save_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    model = create_model(
        image_size=config['data']['image_size'],
        num_channels=config['model']['num_channels'],
        num_res_blocks=config['model']['num_res_blocks'],
        channel_mult=None,
        attention_resolutions=config['model']['attention_resolutions'],
        dropout=config['model']['dropout'],
        num_heads=config['model']['num_heads'],
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        n_steps=config['diffusion']['diffusion_steps'],
        device=device,
        beta_schedule=config['diffusion'].get('beta_schedule', 'linear'),
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

    print(f"Sampling {args.num_samples} images...")
    samples = diffusion.sample(
        n_samples=args.num_samples,
        channels=3,
        img_size=config['data']['image_size'],
    )

    samples_vis = (samples + 1) / 2
    samples_vis = samples_vis.clamp(0, 1)
    out_path = os.path.join(output_dir, f"samples_{args.num_samples}.png")
    save_image(samples_vis, out_path, nrow=args.nrow)
    print(f"Saved samples to {out_path}")


if __name__ == "__main__":
    main()
