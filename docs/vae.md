# Variational Autoencoder (VAE) 论文链接与学术总结

## 1. 论文链接（会议版本优先）

### 1.1 VAE 最经典论文（ICLR 2014）
- **Auto-Encoding Variational Bayes** — Diederik P. Kingma, Max Welling  
  会议条目（OpenReview / ICLR 2014）：  
  https://openreview.net/forum?id=33X9fd2-9FyZd

  arXiv（同一工作最常用公开版本）：  
  https://arxiv.org/abs/1312.6114  
  https://arxiv.org/pdf/1312.6114.pdf

### 1.2 并行独立工作（ICML 2014）
- **Stochastic Backpropagation and Approximate Inference in Deep Generative Models** — Danilo J. Rezende, Shakir Mohamed, Daan Wierstra  
  会议版本（PMLR / ICML 2014）：  
  https://proceedings.mlr.press/v32/rezende14.html

  arXiv：  
  https://arxiv.org/abs/1401.4082  
  https://arxiv.org/pdf/1401.4082.pdf


## 2. VAE 学术总结

### 2.1 研究背景与问题
VAE 研究的是 **带连续隐变量的深度生成模型** 如何在后验不可积的情况下可训练。给定观测数据集
$$
X=\{x^{(i)}\}_{i=1}^N,
$$
我们希望学习一个生成模型，使得模型的边缘分布 $p_\theta(x)$ 能够逼近真实数据分布（等价地，让训练数据的 $\log p_\theta(x)$ 尽可能大）。

困难在于：
- $\log p_\theta(x)$ 需要对隐变量积分/求和，通常 **不可计算**；
- 真后验 $p_\theta(z\mid x)$ 依赖 $\log p_\theta(x)$，也通常 **不可计算**。


### 2.2 生成模型与推断模型（Encoder/Decoder 的概率含义）

#### 生成模型（Decoder 对应的概率对象）
VAE 先规定一个联合分布形式：
$$
p_\theta(x,z)=p(z)\,p_\theta(x\mid z),
$$
其中：
- $p(z)$：先验（常用 $\mathcal N(0,I)$）
- $p_\theta(x\mid z)$：条件似然（由 **Decoder 网络参数化**）

注意：Decoder **不是直接输出 $p_\theta(x)$**，而是输出 $p_\theta(x\mid z)$ 的参数（如 Bernoulli 概率或 Gaussian 均值/方差）。

#### 推断模型（Encoder 学的对象）
由于真后验
$$
p_\theta(z\mid x)=\frac{p_\theta(x\mid z)p(z)}{p_\theta(x)}
$$
不可解，VAE 引入一个变分分布（近似后验）：
$$
q_\phi(z\mid x)\approx p_\theta(z\mid x),
$$
并用 **Encoder 网络** 输出其参数（例如对角高斯的 $\mu_\phi(x),\sigma_\phi(x)$）。


### 2.3 ELBO：为什么必然出现 KL(q||p)（把逻辑串起来）

目标：
$$
\max_{\theta}\log p_\theta(x),\quad 
p_\theta(x)=\int p(z)p_\theta(x\mid z)\,dz.
$$
因为积分不可解，引入 $q_\phi(z\mid x)$，并使用 Jensen 不等式得到证据下界（ELBO）：
$$
\log p_\theta(x)\ge 
\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-\mathrm{KL}\big(q_\phi(z\mid x)\|p(z)\big).
$$

这一步直接解释了你之前的疑问：  
**$\mathrm{KL}(q_\phi(z\mid x)\|p(z))$ 不是拍脑袋加的正则项，而是从 ELBO 推导中自然出现的项。**

同时还有一个关键恒等式：
$$
\log p_\theta(x)=\mathrm{ELBO}(\theta,\phi;x)
+\mathrm{KL}\big(q_\phi(z\mid x)\|p_\theta(z\mid x)\big).
$$
因此最大化 ELBO 等价于：
- 提高 $\log p_\theta(x)$（拟合数据边缘分布）
- 同时让 $q_\phi(z\mid x)$ 更接近模型后验 $p_\theta(z\mid x)$


### 2.4 重参数化技巧（Reparameterization Trick）
为了对 $\phi$ 求梯度，VAE 将采样写成可导的确定性变换：
$$
z=g_\phi(\epsilon,x),\quad \epsilon\sim p(\epsilon).
$$
对高斯近似后验常用：
$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot \epsilon,\quad \epsilon\sim \mathcal N(0,I).
$$
这样梯度可以通过 $z$ 回传到 Encoder 参数 $\phi$。


### 2.5 方法学贡献（论文层面的核心点）
- **摊销变分推断（Amortized Inference）**：用 Encoder 一次性学习从 $x$ 到近似后验参数的映射；
- **SGVB / AEVB 训练框架**：用随机梯度优化 ELBO；
- **重参数化技巧**：使得对连续隐变量的变分学习可高效反向传播。


### 2.6 实验与典型现象（简述）
论文在 MNIST、Frey Face 等数据上展示：
- AEVB/SGVB 可稳定提升变分下界；
- 隐变量维度与下界、生成质量之间存在权衡；
- KL 项在训练中起到正则化作用，避免模型退化为“只重建的自编码器”。


### 2.7 影响与局限（后续工作常针对的点）
- 近似后验族（常用对角高斯）可能表达力不足；
- KL 项过强会导致隐变量“被忽略”（posterior collapse）；
- 为提高生成细节与表达力，后续发展出 flows、IWAE、β-VAE、VQ-VAE 等方向。

---

## 3. 一句话总结
VAE 通过定义生成联合分布 $p_\theta(x,z)=p(z)p_\theta(x\mid z)$ 并引入近似后验 $q_\phi(z\mid x)$，最大化 ELBO 来近似最大化 $\log p_\theta(x)$；因此 KL$(q_\phi(z\mid x)\|p(z))$ 是 ELBO 推导中自然出现、并用于对齐“训练时采样的 $z$”与“生成时先验采样的 $z$”的关键项。
