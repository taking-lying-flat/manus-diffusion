import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    def __init__(
        self,
        betas,
        model_mean_type="epsilon",
        model_var_type="fixed_small",
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        
        # Pre-convert all numpy arrays to torch tensors for faster access
        # These will be moved to device on first use
        self._tensor_cache = {}

    def _extract(self, arr, timesteps, broadcast_shape):
        arr_id = id(arr)
        device = timesteps.device
        
        # Check cache
        cache_key = (arr_id, device)
        if cache_key not in self._tensor_cache:
            # Convert and cache
            if isinstance(arr, np.ndarray):
                tensor = torch.from_numpy(arr).float().to(device)
            else:
                tensor = arr.to(device)
            self._tensor_cache[cache_key] = tensor
        
        arr_tensor = self._tensor_cache[cache_key]
        
        # Index and reshape
        res = arr_tensor[timesteps]
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res.expand(broadcast_shape)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_mean_variance(self, model, x, t, clip_denoised=True):
        B, C = x.shape[:2]
        model_output = model(x, t)

        if self.model_var_type == "fixed_small":
            model_variance = self._extract(self.posterior_variance, t, x.shape)
            model_log_variance = self._extract(self.posterior_log_variance_clipped, t, x.shape)
        elif self.model_var_type == "fixed_large":
            model_variance = self._extract(np.append(self.posterior_variance[1], self.betas[1:]), t, x.shape)
            model_log_variance = torch.log(model_variance)

        if self.model_mean_type == "epsilon":
            pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
        else:
            pred_xstart = model_output

        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(self, model, x, t, clip_denoised=True):
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, model, shape, device, progress=True):
        # Note: caller should set model.eval()
        img = torch.randn(shape, device=device)
        start_img = img.clone()
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            indices = tqdm(indices, desc="Sampling")

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device, dtype=torch.long)
            with torch.no_grad():
                out = self.p_sample(model, img, t)
                img = out["sample"]
        
        return img, start_img

    def sample(self, model, num_samples, batch_size, sample_size, device, progress=True):
        """Generate samples in mini-batches to avoid OOM."""
        samples = []
        starts = []
        for start in range(0, num_samples, batch_size):
            bs = min(batch_size, num_samples - start)
            img, start_img = self.p_sample_loop(
                model, shape=(bs, *sample_size), device=device, progress=progress
            )
            samples.append(img.detach().cpu())
            starts.append(start_img.detach().cpu())
        return torch.cat(samples, dim=0), torch.cat(starts, dim=0)

    def training_losses(self, model, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        model_output = model(x_t, t)

        if self.model_mean_type == "epsilon":
            target = noise
        else:
            target = x_start

        loss = F.mse_loss(model_output, target, reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        return loss.mean()
