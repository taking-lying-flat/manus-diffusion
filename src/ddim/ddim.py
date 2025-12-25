from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def betas_for_alpha_bar(
    num_diffusion_timesteps: int, alpha_bar, max_beta: float = 0.999
) -> np.ndarray:
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


def get_named_beta_schedule(
    schedule_name: str, num_diffusion_timesteps: int
) -> np.ndarray:
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


class GaussianDiffusion:
    def __init__(
        self,
        model: nn.Module,
        n_steps: int,
        device: torch.device,
        beta_schedule: str = "linear",
        ddim_sampling_steps: int = 50,
        ddim_eta: float = 0.0,
    ):
        self.model = model
        self.num_timesteps = n_steps
        self.device = device
        self.ddim_sampling_steps = ddim_sampling_steps
        self.ddim_eta = ddim_eta
        
        betas = get_named_beta_schedule(beta_schedule, n_steps)
        assert (betas > 0).all() and (betas <= 1).all()
        
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        
        self._tensor_cache = {}
        self._setup_ddim(alphas_cumprod)

    def _setup_ddim(self, alphas_cumprod):
        if self.ddim_sampling_steps > self.num_timesteps:
            raise ValueError("ddim_sampling_steps must be <= num_timesteps")
        
        self.ddim_timesteps = np.linspace(
            0, self.num_timesteps - 1, self.ddim_sampling_steps, dtype=np.int64
        )
        
        alpha = alphas_cumprod[self.ddim_timesteps]
        alpha_prev = np.concatenate([np.array([1.0], dtype=alpha.dtype), alpha[:-1]])
        
        self.ddim_alpha = alpha
        self.ddim_alpha_prev = alpha_prev
        
        self.ddim_sigma = (
            self.ddim_eta
            * np.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
        )
        
        self.ddim_alpha_t = torch.from_numpy(self.ddim_alpha).float().to(self.device)
        self.ddim_alpha_prev_t = torch.from_numpy(self.ddim_alpha_prev).float().to(self.device)
        self.ddim_sigma_t = torch.from_numpy(self.ddim_sigma).float().to(self.device)

    def _extract(self, arr, timesteps, broadcast_shape, dtype=None):
        if dtype is None:
            dtype = torch.float32
        
        arr_id = id(arr)
        device = timesteps.device
        
        cache_key = (arr_id, device, dtype)
        if cache_key not in self._tensor_cache:
            if isinstance(arr, np.ndarray):
                tensor = torch.from_numpy(arr).to(device=device, dtype=dtype)
            else:
                tensor = arr.to(device=device, dtype=dtype)
            self._tensor_cache[cache_key] = tensor
        
        arr_tensor = self._tensor_cache[cache_key]
        
        res = arr_tensor[timesteps]
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res.expand(broadcast_shape)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape, x_start.dtype) * x_start
        var = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape, x_start.dtype)
        return mean + var * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape, x_t.dtype)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape, x_t.dtype)
        return (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha

    def p_sample(self, x_t: torch.Tensor, index: int, clip_denoised: bool = True):
        batch_size = x_t.shape[0]
        device = x_t.device
        dtype = x_t.dtype
        
        t_val = int(self.ddim_timesteps[index])
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
        
        eps_pred = self.model(x_t, t)
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)
        
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1, 1)
            sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape, dtype)
            sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape, dtype)
            eps_pred = (x_t - sqrt_alpha * x0_pred) / torch.clamp(sqrt_one_minus_alpha, min=1e-12)
        
        alpha_prev = self.ddim_alpha_prev_t[index].to(device=device, dtype=dtype)
        sigma = self.ddim_sigma_t[index].to(device=device, dtype=dtype)
        
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        sqrt_one_minus = torch.sqrt(torch.clamp(1 - alpha_prev - sigma * sigma, min=0.0))
        
        if float(sigma.item()) == 0.0:
            noise = 0.0
        else:
            noise = torch.randn_like(x_t)
        
        x_prev = sqrt_alpha_prev * x0_pred + sqrt_one_minus * eps_pred + sigma * noise
        
        return x_prev

    def loss(self, x_start: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device, dtype=torch.long)
        
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise)
        eps_pred = self.model(x_t, t)
        
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, n_samples: int, channels: int, img_size: int, progress: bool = False):
        was_training = self.model.training
        self.model.eval()
        
        p = next(self.model.parameters())
        x_t = torch.randn(n_samples, channels, img_size, img_size, device=p.device, dtype=p.dtype)
        
        indices = range(len(self.ddim_timesteps) - 1, -1, -1)
        if progress:
            from tqdm import tqdm
            indices = tqdm(indices, desc="Sampling")
        
        for index in indices:
            x_t = self.p_sample(x_t, index)
        
        if was_training:
            self.model.train()
        
        return x_t

    @torch.no_grad()
    def denoise_step_visualization(self, x_start: torch.Tensor, t_vis: int):
        batch_size = x_start.shape[0]
        device = x_start.device
        
        idx0 = int(np.searchsorted(self.ddim_timesteps, t_vis, side="right") - 1)
        idx0 = max(idx0, 0)
        t0 = int(self.ddim_timesteps[idx0])
        
        t = torch.full((batch_size,), t0, device=device, dtype=torch.long)
        x_t = self.q_sample(x_start, t)
        x_recon = x_t
        
        for index in range(idx0, -1, -1):
            x_recon = self.p_sample(x_recon, index)
        
        return x_start, x_t, x_recon
