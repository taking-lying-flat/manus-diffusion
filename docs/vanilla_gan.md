# Generative Adversarial Nets (GAN)

## 1. 论文链接

**Generative Adversarial Nets** (Goodfellow et al., **NIPS 2014**)

* **NeurIPS/NIPS Proceedings 论文主页**：  
    [https://papers.nips.cc/paper/5423-generative-adversarial-nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets)

* **NeurIPS/NIPS Proceedings 官方 PDF**：  
    [https://papers.neurips.cc/paper/5423-generative-adversarial-nets.pdf](https://papers.neurips.cc/paper/5423-generative-adversarial-nets.pdf)



## 2. 总结

GAN 提出了一种通过 **对抗过程 (Adversarial Process)** 来学习生成模型的新框架，其本质是一个 **两人零和博弈 (Zero-Sum Game)**

### 核心组件
* **生成器 (Generator, $G$)**：
    * **输入**：噪声向量 $z \sim p_z(z)$（通常采样自标准正态分布或均匀分布）
    * **输出**：伪造样本 $G(z)$（与真实数据维度相同，如一张图片）
    * **目标**：欺骗判别器，使其认为 $G(z)$ 是真样本

* **判别器 (Discriminator, $D$)**：
    * **输入**：样本 $x$（可以是真实样本，也可以是生成样本）
    * **输出**：一个标量 $D(x) \in (0, 1)$，代表概率值
        * $D(x) \to 1$：表示 $D$ 认为 $x$ **来自真实数据分布**
        * $D(x) \to 0$：表示 $D$ 认为 $x$ **来自生成器（是假的）**
    * **目标**：正确区分真样本（输出 1）与假样本（输出 0）

### 最终状态
训练中交替更新 $D$ 和 $G$。在理想的 **纳什均衡 (Nash Equilibrium)** 处：
1.  生成分布完全拟合真实分布： $p_g = p_{\text{data}}$
2.  判别器无法区分真假（也就是瞎猜）：对任意输入 $x$，输出恒为 **$D(x) = \frac{1}{2}$**



## 3. 核心公式与关键结论

### (1) 极小极大博弈目标函数 (Minimax Objective)
GAN 的价值函数 $V(D, G)$ 定义如下：
* **$D$ 的目标 (Max)**：最大化识别真图的概率 $\log D(x)$ + 最大化识别假图的概率 $\log(1-D(G(z)))$
* **$G$ 的目标 (Min)**：最小化假图被识别出来的概率 $\log(1-D(G(z)))$

$$
\min_{G}\max_{D}V(D,G) = \mathbb{E}_{x\sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z\sim p_z(z)}[\log(1-D(G(z)))]
$$



### (2) 给定生成器 $G$ 时，判别器 $D$ 的最优解
当我们固定生成器（即固定了 $p_g$）并训练判别器到最优时，判别器的最优形式 $D_G^*(x)$ 为真实数据密度在总密度中的占比：

$$
D_G^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}
$$

**直观含义**：
* 如果在某处 $x$，真实数据概率 $p_{\text{data}}(x)$ 远大于生成概率 $p_g(x)$，则 $D(x) \approx 1$（判真）
* 如果在某处 $x$，生成概率 $p_g(x)$ 远大于真实概率 $p_{\text{data}}(x)$，则 $D(x) \approx 0$（判假）



### (3) 理论核心：等价于最小化 JSD 散度
将最优判别器 $D_G^*(x)$ 代回价值函数，原本的 Minimax 问题在数学上等价于最小化真实分布与生成分布之间的 **Jensen–Shannon Divergence (JSD)**：

$$
V(D_G^*,G) = -\log 4 + 2\cdot \mathrm{JSD}\big(p_{\text{data}} \| p_g\big)
$$

这证明了 GAN 的训练过程本质上是在拉近  $p_g$  和  $p_{\text{data}}$  的距离。当且仅当 $p_g = p_{\text{data}}$ 时，JSD 为 0，达到全局最优解 $-\log 4$


### (4) 生成器的实际训练目标 (Non-saturating Heuristic)
在理论公式中， $G$  应该最小化 $\log(1-D(G(z)))$
**问题**：训练初期  $G$  很弱， $D$  很强，导致 $D(G(z)) \approx 0$，此时 $\log(1-x)$ 函数在 $x=0$ 处的梯度非常平缓（梯度饱和）， $G$  很难学到东西

**解决方案**：论文提出了一种启发式改进，不最小化“被判假”的概率，而是**最大化“被判真”的概率**

$$
\max_G\ \mathbb{E}_{z\sim p_z}[\log D(G(z))] \quad \Longleftrightarrow \quad \min_G\ -\mathbb{E}_{z\sim p_z}[\log D(G(z))]
$$

这被称为 **Non-saturating Loss**，它提供了更强的梯度信号，是实际代码中的标准写法
