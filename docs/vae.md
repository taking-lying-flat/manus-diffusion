# Auto-Encoding Variational Bayes (VAE)

## 1. 论文链接
**Auto-Encoding Variational Bayes** (Kingma & Welling, ICLR 2014)  
[https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)

---

## 2. 精炼总结

VAE 是一种生成模型，旨在解决复杂数据分布中**后验分布不可计算**的问题。

* **核心思想**：引入一个推断网络（Encoder, $q_\phi$）来逼近真实的后验分布 $p_\theta(z|x)$。
* **优化目标**：最大化数据的对数似然下界——**证据下界 (ELBO)**，而不是直接最大化难以计算的 $\log p(x)$。
* **关键技术**：**重参数化技巧 (Reparameterization Trick)**。将随机采样 $z$ 改写为 $z = \mu + \sigma \odot \epsilon$，使得模型可以通过反向传播端到端地训练。

---

## 3. 核心公式推导

### (1) 对数似然的分解
数据的对数似然等于“近似后验与真后验的 KL 散度”加上“证据下界 (ELBO)”。

$$
\log p_{\theta}(\mathbf{x}^{(i)}) = D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}^{(i)}) || p_{\theta}(\mathbf{z}|\mathbf{x}^{(i)})) + \mathcal{L}(\boldsymbol{\theta}, \phi; \mathbf{x}^{(i)})
$$

### (2) 证据下界 (ELBO) 的定义
由于 KL 散度恒大于等于 0，$\mathcal{L}$ 构成了 $\log p(x)$ 的下界。最大化 $\mathcal{L}$ 即间接优化似然。

$$
\log p_{\theta}(\mathbf{x}^{(i)}) \geq \mathcal{L}(\boldsymbol{\theta}, \phi; \mathbf{x}^{(i)}) = \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})} \left[ -\log q_{\phi}(\mathbf{z}|\mathbf{x}) + \log p_{\theta}(\mathbf{x}, \mathbf{z}) \right]
$$

### (3) 实际训练用的 ELBO 形式
这是 VAE 最终的损失函数形式（取负号最小化）：由**正则项**（KL 散度）和**重建项**（期望对数似然）组成。

$$
\mathcal{L}(\boldsymbol{\theta}, \phi; \mathbf{x}^{(i)}) = -D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}^{(i)}) || p_{\theta}(\mathbf{z})) + \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x}^{(i)})} \left[ \log p_{\theta}(\mathbf{x}^{(i)}|\mathbf{z}) \right]
$$
