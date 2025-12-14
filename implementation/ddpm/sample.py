import os
import argparse
import torch
from torchvision.utils import save_image

from train import create_model, create_diffusion
from ema import EMA
from utils import scale_to_01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_channels", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attention_resolutions", type=str, default="16,8")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--use_ema", action="store_true", default=True, help="use EMA weights if present in checkpoint")
    parser.add_argument("--no_ema", action="store_false", dest="use_ema", help="skip EMA weights even if available")
    parser.add_argument("--progress", action="store_true", default=True, help="show sampling progress bar")
    parser.add_argument("--no_progress", action="store_false", dest="progress")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

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
    samples, start_noise = diffusion.sample(
        model,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        sample_size=(3, args.image_size, args.image_size),
        device=device,
        progress=args.progress,
    )

    samples_vis = (samples + 1) / 2
    samples_vis = samples_vis.clamp(0, 1)
    out_path = os.path.join(args.output_dir, f"samples_{args.num_samples}.png")
    save_image(samples_vis, out_path, nrow=args.nrow)
    print(f"Saved samples to {out_path}")

    # noise vs. generated comparison grid
    noise_vis = scale_to_01(start_noise)[: samples_vis.shape[0]]
    compare = torch.cat([noise_vis[: args.nrow], samples_vis[: args.nrow]], dim=0)
    compare_path = os.path.join(args.output_dir, f"compare_noise_{args.num_samples}.png")
    save_image(compare, compare_path, nrow=args.nrow)
    print(f"Saved noise vs. sample grid to {compare_path}")


if __name__ == "__main__":
    main()
