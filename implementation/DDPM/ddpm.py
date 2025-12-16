from typing import Optional, Tuple
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
        model_mean_type: str = "epsilon",
        model_var_type: str = "fixed_small",
    ):
        self.model = model
        self.num_timesteps = n_steps
        self.device = device
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        
        # 1. 获取并固定 beta 序列
        betas = get_named_beta_schedule(beta_schedule, n_steps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1
        assert (betas > 0).all() and (betas <= 1).all()
        
        # 2. 计算 Alpha 相关参数：alpha = 1 - beta
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        
        # 3. 预计算前向扩散过程 q(x_t | x_0) 需要的系数
        # x_t = sqrt(bar_alpha) * x_0 + sqrt(1 - bar_alpha) * epsilon
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        
        # 4. 预计算反向去噪过程需要的系数，用于从 x_t 和 epsilon 还原 x_0
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        
        # 5. 后验分布 q(x_{t-1} | x_t, x_0) 的方差 (variance)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        
        # 6. 后验分布均值 (mean) 的系数 mean = coef1 * x_0 + coef2 * x_t
        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        
        self._tensor_cache = {}

    def _extract(self, arr, timesteps, broadcast_shape):
        arr_id = id(arr)
        device = timesteps.device
        
        cache_key = (arr_id, device)
        if cache_key not in self._tensor_cache:
            if isinstance(arr, np.ndarray):
                tensor = torch.from_numpy(arr).float().to(device)
            else:
                tensor = arr.to(device)
            self._tensor_cache[cache_key] = tensor
        
        arr_tensor = self._tensor_cache[cache_key]
        
        res = arr_tensor[timesteps]
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res.expand(broadcast_shape)

    def q_xt_x0(self, x_0: torch.Tensor, t: torch.Tensor):
        """ 返回 q(x_t | x_0) 的均值和标准差 """
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
        var = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return mean, var

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x_start)
        mean, var = self.q_xt_x0(x_start, t)
        return mean + var * noise

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """ 使用初始化时计算好的系数，根据 x_0 和 x_t 计算后验均值 """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        """ 已知 x_t 和预测的噪声 eps，反推 x_0 """
        # x_0 = (x_t - sqrt(1-bar_alpha) * eps) / sqrt(bar_alpha)
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True):
        # 1. 使用神经网络预测噪声
        eps_pred = self.model(x_t, t)  
        
        # 2. 根据预测的噪声，估计 x_0
        if self.model_mean_type == "epsilon":
            x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)
        else:
            x0_pred = eps_pred
        
        # 3. 截断像素值到 [-1, 1]，防止数值溢出，保证生成质量
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1, 1)
        
        # 4. 计算这一步的后验均值和方差
        mean, _, _ = self.q_posterior_mean_variance(x0_pred, x_t, t)
        
        if self.model_var_type == "fixed_small":
            log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        elif self.model_var_type == "fixed_large":
            var = self._extract(np.append(self.posterior_variance[1], self.betas[1:]), t, x_t.shape)
            log_var = torch.log(var)
        
        # 6. x_{t-1} = mean + exp(0.5 * log_var) * z
        # Reverse-step mean:
        #   mu_theta(x_t, t) is set to the posterior mean μ̃_t(x_t, x0),
        #   but since x0 is unknown during sampling, we plug in x0_hat (predicted x0):
        #   mu_theta(x_t, t) ≈ μ̃_t(x_t, x0_hat)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        x_t_minus_1 = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
        
        return x_t_minus_1

    def loss(self, x_start: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device, dtype=torch.long)
        
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise)
        model_output = self.model(x_t, t)
        
        if self.model_mean_type == "epsilon":
            target = noise
        else:
            target = x_start
        
        loss = F.mse_loss(model_output, target, reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        return loss.mean()

    @torch.no_grad()
    def sample(self, n_samples: int, channels: int, img_size: int, progress: bool = False):
        self.model.eval()
        x_t = torch.randn(n_samples, channels, img_size, img_size, device=self.device)
        
        timesteps = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Denoising")
        
        for t_val in timesteps:
            t = torch.full((n_samples,), t_val, device=self.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t)
        
        return x_t

    @torch.no_grad()
    def denoise_step_visualization(self, x_start: torch.Tensor, t_vis: int):
        batch_size = x_start.shape[0]
        t = torch.full((batch_size,), t_vis, device=self.device, dtype=torch.long)
        
        x_t = self.q_sample(x_start, t)
        
        x_recon = x_t
        for t_idx in reversed(range(t_vis + 1)):
            t_current = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            x_recon = self.p_sample(x_recon, t_current)
        
        return x_start, x_t, x_recon
