import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid


def scale_to_01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def compute_psnr(pred, target, max_val=2.0):
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
    psnr = 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse + 1e-8)
    return psnr.mean().item()


def compute_ssim(pred, target):
    pred = (pred + 1) / 2
    target = (target + 1) / 2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim_map.mean().item()


def visualize_denoising(diffusion, model, x_start, t, save_path=None):
    model.eval()
    with torch.no_grad():
        x_t = diffusion.q_sample(x_start, t)
        out = diffusion.p_sample(model, x_t, t)
        pred_x0 = out["pred_xstart"].clamp(-1, 1)

    psnr = compute_psnr(pred_x0, x_start)
    ssim = compute_ssim(pred_x0, x_start)

    if save_path:
        vis = torch.cat([x_start, x_t, pred_x0], dim=0)
        grid = make_grid((vis + 1) / 2, nrow=x_start.shape[0])
        save_image(grid, save_path)

    return {"psnr": psnr, "ssim": ssim}
