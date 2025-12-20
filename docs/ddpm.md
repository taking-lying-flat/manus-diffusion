# Denoising Diffusion Probabilistic Models (DDPM)

## 1. 论文链接

* **DDPM**: *Denoising Diffusion Probabilistic Models* (Ho, Jain, Abbeel, **NeurIPS 2020**)
  * Proceedings 页面: [https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)
  * 官方 PDF: [https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)



## 2. 一句话总结

DDPM 将生成建模写成一个 **显式的马尔可夫链潜变量模型**：
* **前向过程 (Forward Process)**：把数据逐步加噪，直到变为近似高斯先验
* **反向过程 (Reverse Process)**：学习逐步去噪，恢复数据分布
* **训练目标**：本质上是最大化变分下界 (ELBO)，最终可化简为 **“预测噪声”的均方误差 (MSE)**



## 3. 前向扩散过程 (Forward Process)

给定数据样本 $x_0 \sim p_{\text{data}}(x)$，定义一个固定的加噪马尔可夫链：

$$
q(\mathbf{x}_{1:T}|\mathbf{x}_0) := \prod_{t=1}^{T} q(\mathbf{x}_t|\mathbf{x}_{t-1}), \quad q(\mathbf{x}_t|\mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})
$$

其中：
* $\beta_t \in (0,1)$ 是噪声日程 (Schedule)
* $\alpha_t = 1 - \beta_t$
* $\bar\alpha_t = \prod_{s=1}^t \alpha_s$ (累计乘积)

**关键性质：任意时刻的一步采样公式**
我们可以直接从 $x_0$ 采样出 $x_t$，而不需要一步步跑循环：

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

**重参数化采样 (Reparameterization Trick) 写法**：

$$
x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
$$



## 4. 反向生成过程 (Reverse Process)

我们希望学习一个反向马尔可夫链，把噪声还原成数据：

$$
p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t)
$$

通常设定：
* **先验**： $p(x_T)=\mathcal{N}(0,I)$
* **反向转移**（假设为高斯分布）：

$$
p_\theta(\mathbf{x}_{0:T}) := p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t), \quad p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) := \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$


## 5. 为什么采样公式总是 "mean + sqrt(var) * z"？

因为我们将反向条件分布建模为高斯分布：

$$
x_{t-1}\mid x_t \sim \mathcal{N}(\mu_\theta, \Sigma_\theta)
$$

根据高斯分布的采样性质：

$$
x_{t-1} = \mu_\theta(x_t,t) + L_\theta(x_t,t)\,z,\quad z\sim\mathcal{N}(0,I),\ \ L_\theta L_\theta^\top=\Sigma_\theta
$$

在 DDPM 中，通常取对角方差 $\Sigma_\theta = \sigma_t^2 I$，因此简化为：

$$
x_{t-1} = \mu_\theta(x_t,t) + \sigma_t z
$$

这对应了代码中常见的写法：`mean + std * noise`


## 6. 后验 $q(x_{t-1}\mid x_t, x_0)$ 的闭式解

虽然我们不知道真实的反向过程 $q(x_{t-1}|x_t)$（因为它依赖于未知的整体数据分布），但如果我们已知 $x_0$，后验是可以算出闭式解的：

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t\mathbf{I}),
$$

其中均值  $\tilde\mu$  和方差  $\tilde\beta_t$  为：

$$
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
$$

$$
\tilde\mu(x_t,x_0,t) = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t
$$

### 噪声预测参数化 (Noise Prediction Parameterization)
由于采样时我们不知道 $x_0$，我们用神经网络 $\epsilon_\theta(x_t, t)$ 来预测噪声 $\epsilon$，从而估计 $x_0$。将 $x_0$ 的估计值代入上面的 $\tilde\mu$ 公式，经过化简得到最终的去噪均值公式：

$$
\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t) \right)
$$

这正是代码中 `pred_mean` 的计算来源。


## 7. 训练目标：从 ELBO 到 "Simple MSE"

DDPM 的理论目标是最大化对数似然的变分下界 (ELBO)：

$$
\log p_\theta(x_0) \ge \mathbb{E}_q\left[\log p_\theta(x_{0:T}) - \log q(x_{1:T}\mid x_0)\right]
$$

经过一系列推导和简化（特别是忽略方差权重的加权项），训练目标最终变为极简的**噪声预测均方误差 (MSE)**：

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,x_0,\epsilon} \left[ \left\| \epsilon - \epsilon_\theta(x_t,t) \right\|_2^2 \right]
$$

其中：
*  $t \sim \text{Uniform}(\{1,\dots,T\})$
*  $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$

---

## 8. 采样算法（生成阶段）

给定 $x_T \sim \mathcal{N}(0,I)$，从 $t=T$ 迭代回 $1$：

1. **预测噪声**：  $\hat\epsilon = \epsilon_\theta(x_t,t)$
2. **计算均值**：  $\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\epsilon\right)$
3. **采样 (Langevin Step)**：$x_{t-1} = \mu_\theta(x_t,t) + \sigma_t z, \quad z \sim \mathcal{N}(0,I)$ (注意：当  $t=1$  时，通常设  $z=0$ ，不加噪声)


## 9. 实践要点（Code Level）

* 噪声日程  $\beta_t$ ：原始论文使用线性 Schedule，但后续改进版（如 Cosine Schedule）效果更好
* **网络预测目标**：最常用的是预测  $\epsilon$。也可以预测  $x_0$ 或  $v$ ，但在标准 DDPM 中  $\epsilon$ 是默认选择
* **Mask $t=0$**：代码中通常会有一个 mask 操作，确保在最后一步$t=1 \to 0$ 不再加入随机噪声，保证输出的一致性
