
## 0. DDPM 的硬骨架：两条链 + 一个 ELBO

### 生成模型（反向链）
DDPM 把生成分布写成一条从噪声到图像的反向马尔可夫链：

$$
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t),
\qquad p(x_T)=\mathcal N(0,I)
$$

然后我们要优化的边缘分布（对数似然）是：

$$
p_\theta(x_0)=\int p_\theta(x_{0:T})\,dx_{1:T}
$$

这就是标准的“潜变量模型”视角

### 推断/前向链（固定的加噪过程）
DDPM 的前向扩散也是马尔可夫链：

$$
q(x_{1:T}\mid x_0)=\prod_{t=1}^T q(x_t\mid x_{t-1}),
\qquad q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}\,x_{t-1},\, (1-\alpha_t)I)
$$



## 1. 最关键闭式边缘： $q(x_t\mid x_0)$

这一步是整个扩散模型的基石。

从定义出发：

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\,\epsilon_t,\quad \epsilon_t\sim\mathcal N(0,I)
$$

把它连乘展开（或用归纳法）会得到：

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal N(0,I)
$$

因此边缘分布是：

$$
q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I)
$$

**停一下：这条闭式边缘就是后面 DDIM “训练不变”的核心原因。**
因为很多训练目标只用到了它，而没用到联合分布是马尔可夫链这件事

---

## 2. 反向“真后验” $q(x_{t-1}\mid x_t,x_0)$

**为什么它也是高斯？**
你已经有：
1. $q(x_{t-1}\mid x_0)$ 是高斯（用上面闭式边缘）
2. $q(x_t\mid x_{t-1})$ 是高斯（前向一步）

根据贝叶斯公式：
$$
q(x_{t-1}\mid x_t,x_0)\propto q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0)
$$
高斯乘高斯还是高斯 $\Rightarrow q(x_{t-1}\mid x_t,x_0)$ 也是高斯（均值是线性组合，方差可写闭式）。
这一步在 DDPM 推导里用于把 ELBO 拆成一堆 KL 项。

---

## 3. ELBO 拆开后，为什么会变成“预测噪声”的 MSE

ELBO（变分下界）标准形式：
$$
\log p_\theta(x_0)\ \ge\ 
\mathbb E_{q(x_{1:T}\mid x_0)}
\big[\log p_\theta(x_{0:T})-\log q(x_{1:T}\mid x_0)\big]
$$
把两条链代进去、逐步整理，会得到“每个时间步一个 KL”的结构（省略常数项）：
$$
\mathcal L \approx \sum_{t=2}^T 
\mathbb E_q \Big[
\mathrm{KL}\big(q(x_{t-1}\mid x_t,x_0)\ \|\ p_\theta(x_{t-1}\mid x_t)\big)
\Big]
\ +\ \text{(t=1 重建项)}
$$
如果你令反向模型也用高斯、并且方差取某个固定值（或按日程给定）：
$$
p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta(x_t,t),\ \sigma_t^2 I)
$$
那么每个 KL（同方差高斯之间）就等价于“均值差的平方”：
$$
\mathrm{KL}(\mathcal N(\mu_q,\sigma_t^2 I)\|\mathcal N(\mu_\theta,\sigma_t^2 I))
=\frac{1}{2\sigma_t^2}\|\mu_q-\mu_\theta\|^2
$$
接下来关键是：用 $\epsilon$-参数化（噪声预测器）来写均值，让 $\mu_q-\mu_\theta$ 变成一个系数乘 $(\epsilon-\epsilon_\theta)$，于是整体等价于加权 MSE：
$$
\sum_t \gamma_t\ \mathbb E\|\epsilon-\epsilon_\theta(x_t,t)\|^2
$$
这就是你熟悉的 DDPM “simple loss / noise prediction loss”。

---

## 4. 到这里为止，你缺的那条“桥”是什么？

你卡的点其实是：
**为什么 DDIM 能把前向过程改成非马尔可夫，却还“共享训练目标”？**

答案就一句话，但必须用公式说明：
因为上面的噪声 MSE 训练，只用到了闭式边缘 $q(x_t\mid x_0)$，而不需要用到“联合分布 $q(x_{1:T}\mid x_0)$ 是不是马尔可夫链”。

这正是 DDIM 的构造空间：
**只要你构造一个新的联合分布 $q_\sigma(x_{1:T}\mid x_0)$，让所有边缘 $q_\sigma(x_t\mid x_0)$ 仍等于同一个 $\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$，训练那套噪声预测就不需要变。**

---

## 5. DDIM 的核心：构造一族 $q_\sigma$，边缘不变、联合可变

DDIM 定义一族“推断分布”（论文里叫 generalized forward / inference process）：
$$
q_\sigma(x_{1:T}\mid x_0)=q(x_T\mid x_0)\prod_{t=2}^T q_\sigma(x_{t-1}\mid x_t,x_0)
$$
并令每一步条件分布是：
$$
q_\sigma(x_{t-1}\mid x_t,x_0)=\mathcal N(\mu_\sigma(x_t,x_0,t),\ \sigma_t^2 I)
$$
其中均值被刻意设计成：
$$
\mu_\sigma(x_t,x_0,t)
=
\sqrt{\bar\alpha_{t-1}}x_0
+
\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot
\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}
$$

### 5.1 这均值不是“拍脑袋”：它来自一个更直观的重参数化
先用闭式边缘把 $x_t$ 写成：
$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal N(0,I)
$$
于是可以反解出 $\epsilon$：
$$
\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}=\epsilon
$$
然后 DDIM 直接规定（这就是它的定义）：
$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}}x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon+\sigma_t z,
\quad z\sim\mathcal N(0,I)
$$
给定 $x_t,x_0$ 时，$\epsilon$ 就被确定了。
这样你立刻得到上面的 $\mu_\sigma$。

### 5.2 为什么这样做能保证“边缘不变”？
看 $x_{t-1}\mid x_0$：
$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}}x_0 + \underbrace{\big(\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon+\sigma_t z\big)}_{\text{仍然是 } \mathcal N(0,\ (1-\bar\alpha_{t-1})I)}
$$
因为 $\epsilon,z$ 独立高斯，方差相加：
$$
(1-\bar\alpha_{t-1}-\sigma_t^2)+\sigma_t^2=1-\bar\alpha_{t-1}
$$
所以边缘仍是：
$$
q_\sigma(x_{t-1}\mid x_0)=\mathcal N(\sqrt{\bar\alpha_{t-1}}x_0,\ (1-\bar\alpha_{t-1})I)
$$
这就是“边缘完全不变”的严格原因。

---

## 6. 生成（反向）怎么来：把 $x_0$ 用网络估计替换掉

上面 $q_\sigma(x_{t-1}\mid x_t,x_0)$ 需要真实 $x_0$，但采样时你没有 $x_0$。
DDIM/ DDPM 都是同一招：

### 6.1 先从 $\epsilon_\theta$ 得到“预测的干净图” $\hat x_0$
$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
\Rightarrow
x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\bar\alpha_t}}
$$
把 $\epsilon$ 用网络预测 $\epsilon_\theta(x_t,t)$ 替换：
$$
\hat x_0
=
f_\theta(x_t,t)
:=
\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}
$$

### 6.2 再把 $\hat x_0$ 塞回刚才的 $q_\sigma$ 均值里
于是定义反向转移：
$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal N\!\Big(
\sqrt{\bar\alpha_{t-1}}\hat x_0+
\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\ \epsilon_\theta(x_t,t),
\ \sigma_t^2 I
\Big)
$$
对应一次采样更新就是（你要的“顺着公式到底”版本）：
$$
\boxed{
x_{t-1}
=
\sqrt{\bar\alpha_{t-1}}\hat x_0
+
\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\ \epsilon_\theta(x_t,t)
+
\sigma_t z,\quad z\sim\mathcal N(0,I)
}
$$
这条就是很多库里标注的 DDIM 公式（常被称作 Eq.(12) 那条统一式）。

---

## 7. 两个“极限/特例”一下就看懂：DDPM vs DDIM

### 7.1 取特定 $\sigma_t$ $\Rightarrow$ 退化回随机 DDPM（ancestral sampling）
很多实现用一个 $\eta$ 控制随机性，$\eta=1$ 对应 DDPM 的后验方差：
$$
\sigma_t
=
\eta\cdot
\sqrt{
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}
\Big(1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}\Big)
}
$$

### 7.2 取 $\sigma_t=0$ $\Rightarrow$ 确定性 DDIM
把噪声项直接消掉：
$$
\boxed{
x_{t-1}
=
\sqrt{\bar\alpha_{t-1}}\hat x_0
+
\sqrt{1-\bar\alpha_{t-1}}\ \epsilon_\theta(x_t,t)
}
$$
此时 给定同一个初始 $x_T$，整条轨迹是确定的（这就是 “implicit / deterministic” 的含义）。

---

## 8. “训练不变”最后还差一步严谨性：为什么还是噪声 MSE？

你想要的就是这句严格推导（我把关键代数写出来）：

真实推断的均值（来自 $q_\sigma$）：
$$
\mu_q=\sqrt{\bar\alpha_{t-1}}x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\ \epsilon
$$
模型的均值（把 $x_0$ 换成 $\hat x_0$）：
$$
\mu_\theta=\sqrt{\bar\alpha_{t-1}}\hat x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\ \epsilon_\theta
$$
把 $\hat x_0$ 的定义代进去整理，你会发现：
$$
\mu_q-\mu_\theta = c_t\cdot(\epsilon-\epsilon_\theta)
$$
其中 $c_t$ 是只依赖 $\bar\alpha_t,\bar\alpha_{t-1},\sigma_t$ 的系数。
然后每步 KL（同方差高斯）就是：
$$
\mathrm{KL}=\frac{1}{2\sigma_t^2}\|\mu_q-\mu_\theta\|^2
=
\gamma_t\|\epsilon-\epsilon_\theta\|^2
$$
所以整个变分目标就是：
$$
\sum_t \gamma_t\,\mathbb E\|\epsilon-\epsilon_\theta(x_t,t)\|^2 + \text{const}
$$
这就是“不同 $q_\sigma$（不同 $\sigma$）只会改变权重 $\gamma_t$，但损失形式仍然是噪声预测 MSE”，也就是论文说的“共享 surrogate objective”。

---

## 9. 采样为什么能加速：不是“公式更快”，是“你可以跳步”

上面统一更新式默认跑 $t=T,T-1,\dots,1$ 共 $T$ 步。
但 DDIM 允许你选一个子序列（时间索引）：
$$
\tau=(\tau_1<\cdots<\tau_S)\subset\{1,\dots,T\}
$$
采样时只在这些点上更新：
$$
x_{\tau_S}\to x_{\tau_{S-1}}\to \cdots \to x_{\tau_1}\to x_0
$$
更新式完全同型，只要把 $(t,t-1)$ 换成 $(\tau_i,\tau_{i-1})$，把 $\bar\alpha_t$ 换成 $\bar\alpha_{\tau_i}$。
于是你把网络调用次数从 $T=1000$ 直接降到 $S=50/20$，这就是速度来源。
