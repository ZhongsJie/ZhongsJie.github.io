---
title: "MLA(Multi-Head Latent Attention)"
date: 2026-06-08T23:39:17+08:00
draft: false
subtitle: ""
author: "ZhongsJie"
tags:
  - deepseek
  - inference
  - attention
categories: []
showToc: true
TocOpen: true
disableShare: true
cover:
    image: ""
    caption: ""
    alt: ""
    relative: false
---

![](/images/2026-06/Pasted%20image%2020260608085830.png)

## MHA（多头注意力）
$$Attention(Q, K, V) = softmax(Q·K^T / \sqrt{d_h}) · V$$
$$q_t = W^Q h_t$$
$$k_t = W^K h_t$$
$$v_t = W^V h_t$$

  - $d$: Embedding维度；
  - $d_h$：每一个头的维度；
  - $n_h$: Attention头数；
  - $h_t ∈ ℝ^{d}$： Attention单层中第$t$ 个token的输入；
  - $W^Q, W^K, W^V ∈ ℝ^{d_hn_h × d}$ ；
  - $q_t,k_t,v_t ∈ ℝ^{d_hn_h}$ ；
$$[q_{𝑡,1};q_{𝑡,2}; ...; q_{𝑡,𝑛_ℎ} ] = q_t$$
$$[k_{𝑡,1};k_{𝑡,2}; ...; k_{𝑡,𝑛_ℎ} ] = k_t$$
$$[v_{𝑡,1};v_{𝑡,2}; ...; v_{𝑡,𝑛_ℎ} ] = v_t$$
$$o_{t,j} = \sum_{j=1}^t{softmax_j(q_{t,i}^T·k_{j,i} / \sqrt{d_h})} · v_{j,i}$$
$$u_𝑡 = 𝑊^𝑂 [o_{𝑡,1}; o_{𝑡,2}; ...; o_{𝑡,𝑛_ℎ} ]$$
- $q_{t,i},k_{t,i},v_{t,i} ∈ ℝ^{d_h}$ : 第$i$个Attention头；
- $W^O ∈ ℝ^{d × d_hn_h}$ ：输出投影矩阵；
- 每个 token 单层需要缓存的 cache 的是 $n_h$ 套 ($k,v$) — 共 $2n_hd_h$


## MQA（多查询注意力）

> 所有 $n_h$ 个 Q head 共享同一套 K、V，只剩 1 套，更快的推理速度：由于内存和缓存需求大幅下降，内存带宽开销也基本消除。虽然训练效率基本保持不变，但与 MHA 相比，共享 K/V 对可能会导致模型质量和输出精度略有下降

$$
q_{t,i} = W_{i}^Q · h_t,
\qquad
k_{t} = W^K h_t,
\qquad
v_{t} = W^V h_t
$$

\[
\operatorname{head}_{(i)}
= \operatorname{softmax}
\left(
\frac{
Q_{i}K^{\top}
}{
\sqrt{d_h}
}
+
M
\right)
V,
\qquad
i = 1, \cdots, n_h
\]

- $W^K, W^V \in \mathbb{R}^{d_h \times d}$ —— 所有 head 共享同一份 K、V 投影。
- cache 收缩到 $2d_h$，比 MHA 小 $n_h$ 倍；但表达能力受限，大模型直接套 MQA 容易掉点 —— 工程上很少单独使用，更多被 GQA 替代。

## GQA（分组注意力）

把 $n_h$ 个 Q head 切成 $n_{kv}$ 组，组内共享 K、V —— 是 MHA 与 MQA 之间的连续插值：

$$
q_{t,i} = W_{i}^Q · h_t,
\qquad
k_{t} = W^K_g h_t,
\qquad
v_{t} = W^V_g h_t
$$

\[
\operatorname{head}_{(h)}
= \operatorname{softmax}
\left(
\frac{
Q_{(h)} K_{(g(h))}^{\top}
}{
\sqrt{d_h}
}
+
M
\right)
V_{(g(h))},
\qquad
g(h) = \left\lfloor \frac{h \cdot n_{kv}}{n_h} \right\rfloor
\]

- $W^K_{(g)}, W^V_{(g)} \in \mathbb{R}^{d_h \times d}$，$g = 1, \cdots, n_{kv}$ —— 每组一份 K、V 投影。
- $g(h)$ —— 第 $h$ 个 Q head 所属的组索引。
- $n_{kv} = n_h$ 退化为 MHA，$n_{kv} = 1$ 退化为 MQA。
- kernel 里只实际算 $n_{kv}$ 套 K、V，分数计算时把 K、V 沿 group 维 broadcast 到 $n_h$，不真的复制张量。
- 每Token需要缓存$2d_hn_{kv}$
- Llama 3 70B：$n_h = 64$，$n_{kv} = 8$，$d_h = 128$，单 token cache，比同样大小的 MHA 小 $8$ 倍。

$$
2 \cdot 8 \cdot 128 \cdot 2\text{B} = 4\text{KB} / \text{layer}
$$

## MLA（多头潜在注意力）
>标准 MHA 中: $k_t = W^K·h_t, v_t = W^V·h_t$，均为从 $h_t ∈ ℝ^{4096}$ 投影而来。虽然 K 和 V 投影后的维度是 $d_h·n_h = 16384$，但它们的秩不会超过 4096 ($h_t$ 的维度)。**这 16384 维中存在大量线性相关的冗余**。而GQA 只是按 head 数线性缩 cache，MLA将完整的 K/V 向量替换为一个低维隐向量的缓存 $c^{KV}$，推理时通过矩阵吸收技巧避免显式恢复完整 K/V。通过优化KV-cache来减少显存占用，从而提升推理性能。

### 低秩 KV 联合压缩

将高维 $h_t$ 压缩到低维隐空间:

$$c_t^{KV} = W^{DKV} · h_t$$
  - $c_t^{KV} ∈ ℝ^{d_c}$  ：压缩后的 KV 联合隐向量, $d_c \ll d_hd_n$
  - $W^{DKV} ∈ ℝ^{d_c \times  d}$： 下投影矩阵 ，低秩变化矩阵
  - $d_c = 512$ ：论文中取$4\times d_h$，远小于 $d_h·n_h$ = 16384 (压缩比 32:1)
$$k_t^{C}=W^{UK} · c_t^{KV}$$
$$v_t^{C}=W^{UV} · c_t^{KV}$$
-  $W^{UK}, W^{UV} ∈ ℝ^{d_hn_h \times  d_c}$： 上投影矩阵

经过上述变化，通过两个低秩变化矩阵先做压缩、再做扩展，最终能降低参数的数量。MLA本质是要做到减少KV-cache的存储，在推理过程中只需要缓存$c_t^{KV}$， $W^{UK}$可以吸收到$W^Q$， $W^{UV}$可以吸收到$W^O$。为了减少激活的内存量，在训练过程中，deepseek也会对$Q$采用类似的低秩化操作（不减少缓存）：
$$c_t^{Q}=W^{DQ} · h_t$$
$$q_t^{Q}=W^{UQ} · c_t^{Q}$$
- $W^{DQ} ∈ ℝ^{d_c^{'} \times  d}$： 下投影矩阵
- $W^{UQ} ∈ ℝ^{d_hn_h \times  d_c^{'}}$： 上投影矩阵
- $d_c^{'}\ll d_hn_h$:  Deepseek中使用1536

在低秩化过程中**可能存在的问题**是：如果真实的数据分布在很多"方向"上都有重要成分，而你选的秩 $r$ 不够大，那么：
- 某些方向的信息会被强行压扁甚至抹掉；
- 在 attention 里体现为：某些 token 间的相关性被低估或高估，注意力图更"粗糙"。

#### 矩阵吸收计算

> "矩阵吸收"本质上就是：把两个线性变换先乘好，合成到一个矩阵里，从而少做一次矩阵乘法或少存一套权重。

$$x_1^{'T}x_2^{'} = (Px_1)^{T}{Qx_2} = x_1^{T}(P^TQ)x_2=x_1^{T}Q^{'}x_2=x_1^{T}x_2^{''}$$
在 MLA 里这么做有两个现实收益：
- 减少一次 matmul 或在 kernel 里直接用 fused 权重，少访存、易优化。
- 既然 key/value 的上投影 $W_{UK}, W_{UV}$ 被吸掉了，推理阶段只需要缓存 latent  $c^{KV}$不用缓存大维 K/V，本身的 KV cache 就能保持极小。
### 解耦RoPE

>In this way, $𝑊^{𝑈𝐾}$ cannot be absorbed into $𝑊^𝑄$ any more during inference, since a RoPE matrix related to the currently generating token will lie between $𝑊^𝑄$ and $𝑊_{𝑈𝐾}$ and matrix multiplication does not obey a commutative law. **RoPE与低秩KV不兼容，没法做矩阵吸收计算**

1) 在不考虑RoPE的情况下：那么 $q, k$ 乘积计算如下，其中 $(i)$ 表示变换矩阵第 $i$ 个 Head 的切片：

$$
\begin{aligned}
q_{t,i}^{T} \times k_{j,i}
&=
\left(W_{(i)}^{UQ} c_t^Q\right)^T
\times
W_{(i)}^{UK} c_j^{KV}  \\
&=
\left(c_t^Q\right)^T
\times
\left(W_{(i)}^{UQ}\right)^T W_{(i)}^{UK}
\times
c_j^{KV}
\end{aligned}
$$
**不加 RoPE**的情况下，可以提前计算好  $\left(W_{(i)}^{UQ}\right)^T W_{(i)}^{UK}$部分的值，  也就是上面说的 $W^{UK}$ 吸收到 $W^{UQ}$ 中，这样在做 $q$ 的变换的时候，也就同时计算 $W^{UK}$ 矩阵的乘法。
因此在推理过程中只需要缓存 $c_j^{KV}$，而不是缓存 $W_{(i)}^{UK} \times c_j^{KV}$ 的结果。  而$c_j^{KV}$ 维度只有 $4d_h$ 的长度， $W_{(i)}^{UK} \times c_j^{KV}$ 是个 $4d_h \to d$ 的变换，也就是完全恢复了隐层的维度：
$$
d = d_h * n_h = 64d_h
$$
DeepSeek-V3 中 $n_h$ 配置为 64，这是 MLA 的**压缩 KV Cache 的核心原理**。

> $\left(W_{(i)}^{UQ}\right)^TW_{(i)}^{UK}$合并成一个矩阵的恒等变换，虽然算子序列在数学上可交换/重排，但浮点加减乘除不满足严格的结合律，理论上只有在无限精度下才成立，实际上如果我们使用单精度尤其是BF16的话，经过变换后的精度损失往往还是挺明显的，经过多层累积后可能放大到比较可观的程度；

---
2) 但是在考虑RoPE的场景下，加上 RoPE 后，计算 $q, k$ 乘积，会在 $\left(W_{(i)}^{UQ}\right)^T$ 和 $W_{(i)}^{UK}$ 之间，增加一个融合了相对位置的变量 $\mathcal{R}_{t-j}$，在标准 MHA 里，RoPE 对 **Query 和 Key** 都要进行旋转$q_{t,i}=RoPE(W_{(i)}^{UQ} c_t^Q)=\mathcal{R}_t W_{(i)}^{UQ} c_t^Q$其中$\mathcal{R}_t$为位置$t$的RoPE 位置矩阵：

$$
\begin{aligned}
q_{t,i}^{T} \times k_{j,i}
&=
\left(\mathcal{R}_t W_{(i)}^{UQ} c_t^Q\right)^T
\times
\mathcal{R}_j W_{(i)}^{UK} c_j^{KV} \\
&=
\left(c_t^Q\right)^T
\times
\left(W_{(i)}^{UQ}\right)^T
\mathcal{R}_t^T \mathcal{R}_j
W_{(i)}^{UK}
\times c_j^{KV} \\
&=
\left(c_t^Q\right)^T
\times
\left(W_{(i)}^{UQ}\right)^T
\mathcal{R}_{t-j}
W_{(i)}^{UK}
\times c_j^{KV}
\end{aligned}
$$

中间这个分量  $\left(W_{(i)}^{UQ}\right)^T \mathcal{R}_{t-j} W_{(i)}^{UK}$  是**随相对位置变化而变化**的，并不是固定的矩阵，并不能提前计算好。  因此 RoPE 与低秩变换不兼容。最简单的方式是放弃RoPE，换用其他基于Attention Bias的位置编码，如ALIBI，但DeepSeek的实验显示它明显不如RoPE（注意，MLA不是不能加RoPE，而是加了RoPE之后无法用恒等变换技巧来减少KV Cache），MLA的本质改进也在于低秩投影之后的RoPE适配工作

---
3) 通过增加一个**很小 $q,k$分量**，引入RoPE

解耦 RoPE 策略。该策略使用额外的多头 Query  $\mathbf{q}_{t,i}^{R} \in \mathbb{R}^{d_h^R}$  以及一个共享的 Key  $\mathbf{k}_{t}^{R} \in \mathbb{R}^{d_h^R}$  来承载 RoPE，其中 $d_h^R$ 表示解耦后的 Query 和 Key 在每个 Head 上的维度。
在引入解耦 RoPE 策略后，MLA 执行如下计算：

$$
[\mathbf{q}_{t,1}^{R}; \mathbf{q}_{t,2}^{R}; \cdots; \mathbf{q}_{t,n_h}^{R}]
= \mathbf{q}_{t}^{R}
= \operatorname{RoPE}(W^{QR}\mathbf{c}_{t}^{Q})
$$

$$
\mathbf{k}_{t}^{R}
= \operatorname{RoPE}(W^{KR}\mathbf{h}_{t})
$$

$$
\textcolor{red}{\mathbf{q}_{t,i}
= [\mathbf{q}_{t,i}^{C}; \mathbf{q}_{t,i}^{R}]}
$$

$$
\textcolor{red}{\mathbf{k}_{t,i}
= [\mathbf{k}_{t,i}^{C}; \mathbf{k}_{t}^{R}]}
$$

$$
\mathbf{o}_{t,i}
= \sum_{j=1}^{t}
\operatorname{Softmax}_{j}
\left(
\frac{
\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}
}{
\sqrt{d_h + d_h^R}
}
\right)
\mathbf{v}_{j,i}^{C}
$$

$$
\mathbf{u}_{t}
= W^{O}
[
\mathbf{o}_{t,1};
\mathbf{o}_{t,2};
\cdots;
\mathbf{o}_{t,n_h}
]
$$

- $W^{QR} \in \mathbb{R}^{d_h^R n_h \times d_c'}$  $W^{KR} \in \mathbb{R}^{d_h^R \times d}$  ：分别是用于生成解耦 Query 和 Key 的矩阵；
- $\operatorname{RoPE}(\cdot)$ 表示应用 RoPE 矩阵的操作；
- $[\cdot;\cdot]$ 表示拼接操作。

MLA采取了一种**混合方法**——每个 Attention Head 的 Q、K 新增 $d_r$ 个维度用来添加 RoPE，其中 K 新增的维度每个 Head 共享，将 RoPE 加在 $c_i$ 之后，这样 **$\mathcal{R}_i$ 就可以吸收到 $c_i$ 中去**，但这样就没有：$\mathcal{R}_t^{\top} \mathcal{R}_j=\mathcal{R}_{t-j}$ 的运算，此时的 RoPE 不再是通过绝对位置实现相对位置，而单纯是在 Q、K 上加绝对位置，让模型自己想办法提炼相对位置信息。在推理过程中，解耦后的 Key 也需要被缓存。

### MLA完整计算推导

![](/images/2026-06/Pasted%20image%2020260608185756.png)

$$
\begin{aligned}
\mathbf{c}_t^Q &= W^{DQ}\mathbf{h}_t, \\[4pt]
[\mathbf{q}_{t,1}^C;\mathbf{q}_{t,2}^C;\cdots;\mathbf{q}_{t,n_h}^C]
&= \mathbf{q}_t^C
= W^{UQ}\mathbf{c}_t^Q, \\[4pt]
[\mathbf{q}_{t,1}^R;\mathbf{q}_{t,2}^R;\cdots;\mathbf{q}_{t,n_h}^R]
&= \mathbf{q}_t^R
= \operatorname{RoPE}(W^{QR}\mathbf{c}_t^Q), \\[4pt]
\mathbf{q}_{t,i}
&= [\mathbf{q}_{t,i}^C;\mathbf{q}_{t,i}^R], \\[8pt]
\textcolor{red}{\mathbf{c}_t^{KV}}
&= W^{DKV}\mathbf{h}_t, \\[4pt]
[\mathbf{k}_{t,1}^C;\mathbf{k}_{t,2}^C;\cdots;\mathbf{k}_{t,n_h}^C]
&= \mathbf{k}_t^C
= W^{UK}\mathbf{c}_t^{KV}, \\[4pt]
\textcolor{red}{\mathbf{k}_t^R}
&= \operatorname{RoPE}(W^{KR}\mathbf{h}_t), \\[4pt]
\mathbf{k}_{t,i}
&= [\mathbf{k}_{t,i}^C;\mathbf{k}_t^R], \\[8pt]
[\mathbf{v}_{t,1}^C;\mathbf{v}_{t,2}^C;\cdots;\mathbf{v}_{t,n_h}^C]
&= \mathbf{v}_t^C
= W^{UV}\mathbf{c}_t^{KV}, \\[8pt]
\mathbf{o}_{t,i}
&=
\sum_{j=1}^{t}
\operatorname{Softmax}_{j}
\left(
\frac{
\mathbf{q}_{t,i}^{T}\mathbf{k}_{j,i}
}{
\sqrt{d_h + d_h^R}
}
\right)
\mathbf{v}_{j,i}^C, \\[8pt]
\mathbf{u}_t
&=
W^O[\mathbf{o}_{t,1};\mathbf{o}_{t,2};\cdots;\mathbf{o}_{t,n_h}].
\end{aligned}
$$

-  $\mathbf{c}_t^{KV}$ 是低秩压缩的向量， 维度是$4\times d_h=512$
- $\mathbf{k}_t^R$ 是引入位置编码的MQA范式计算的共享$k$，维度$d_h/2=64$

$$
\begin{aligned}
q_{t,i}^{T} \times k_{j,i}
= [q_{t,i}^{C}; q_{t,i}^{R}]^{T}
\times
[k_{j,i}^{C}; k_{t}^{R}]
&=
q_{t,i}^{C} k_{j,i}^{C}
+
q_{t,i}^{R} k_{t}^{R} \\
q_{t,i}^{C} k_{j,i}^{C}
&=
(c_t^Q)^T
(W_{(i)}^{UQ})^T
W_{(i)}^{UK}
c_j^{KV} \\
\tilde{q}_{t,i}^{C}
&=
(c_t^Q)^T
(W_{(i)}^{UQ})^T
W_{(i)}^{UK} \\
q_{t,i}^{C} k_{j,i}^{C}
&=
\tilde{q}_{t,i}^{C} c_j^{KV}
\end{aligned}
$$

前一项 $q_{t,i}^{C} k_{j,i}^{C}$ 通过矩阵吸收处理，全 Head 只需要缓存一个 $c_t^{KV}$；后一项 $q_{t,i}^{R} k_t^{R}$ 按正常 MQA 的方式计算，全 Head 只需要缓存一个共享的 $k^R$。
$$
\begin{aligned}
s_{t,j,i}
= \frac{
q_{t,i}^{T} k_{j,i}
}{
\sqrt{d_h + d_h^R}
}
&=
\frac{
q_{t,i}^{C} k_{j,i}^{C}
+
q_{t,i}^{R} k_{j}^{R}
}{
\sqrt{d_h + d_h^R}
} \\
a_{t,j,i}
&=
\operatorname{Softmax}_j
\left(
\frac{
\tilde{q}_{t,i}^{C} c_j^{KV}
+
(q_{t,i}^{R})^T k_j^{R}
}{
\sqrt{d_h + d_h^R}
}
\right) \\
o_{t,i}
&=
\sum_{j=1}^{t}
a_{t,j,i}
v_{j,i}^{C} \\
o_{t,i}
&=
\sum_{j=1}^{t}
\operatorname{Softmax}_j
\left(
\frac{
\tilde{q}_{t,i}^{C} \textcolor{red}{c_j^{KV}}
+
(q_{t,i}^{R})^T \textcolor{red}{k_j^{R}}
}{
\sqrt{d_h + d_h^R}
}
\right)
v_{j,i}^{C}
\end{aligned}
$$
因此，推理阶段实际需要缓存的是：

$$
\{c_j^{KV}, k_j^R\}_{j=1}^{t}
$$

而不是完整的：

$$
\{k_{j,i}, v_{j,i}\}_{i=1}^{n_h}
$$

所以 MLA 的 KV Cache 大小从 MHA 的：$2 n_h d_h$压缩为：$d_c + d_h^R$
其中 $c_j^{KV}$ 负责承载可被低秩压缩的 K/V 信息，$k_j^R$ 负责承载不能被矩阵吸收的 RoPE 位置信息。

### MLA Mode
>MLA模块有两种计算模式，在DeepseekV3.2的报告中，非吸收矩阵用**MHA mode表示**，吸收矩阵用**MQA mode表示**。MHA模式在prefill阶段使用，MQA模式在decode阶段使用。

![](/images/2026-06/Pasted%20image%2020260608192754.png)
- 在推理中可根据阶段的特征来选择形态，因为**序列的变化**会让两种计算形态的**计算量**、**显存占用量**不一样！ 序列的变化包括每次输入MLA的序列长度，以及历史的KV cache长度。

差异点：
- Q:  吸收形态的$q_{nope}$后面多了一个矩阵乘法运算；
- KV: 非吸收形态要经过上采样后再进行attention计算，而吸收形态不用。吸收形态KV的head只有一份(相当于MQA)；
- $O$： 吸收矩阵MLA的O线性层计算前多了一个V转移过来的矩阵乘法；


## Attention 对比：参数量 / KV cache

### 参数量对比（单层）

<style>
.param-table { border-collapse: collapse; width: 100%; }
.param-table th, .param-table td { padding: 6px 20px; text-align: left; white-space: nowrap; }
.param-table tr.sep td { border-bottom: 2px solid #333; }
.param-table tr.sep-top td { border-top: 2px solid #333; }
</style>

<table class="param-table">
  <tr class="sep-top"><td></td><td></td><td></td><td></td><td></td></tr>
  <tr>
    <th>步骤</th>
    <th>MHA</th>
    <th>MQA</th>
    <th>GQA</th>
    <th>MLA</th>
  </tr>
  <tr class="sep"><td></td><td></td><td></td><td></td><td></td></tr>
  <tr>
    <td>Q 投影</td>
    <td>$H⋅n_hd_h$</td>
    <td>$H⋅n_hd_h$</td>
    <td>$H⋅n_hd_h$</td>
    <td>$Hd_h′+d_h′⋅n_hd_h$</td>
  </tr>
  <tr>
    <td>K 投影</td>
    <td>$H⋅n_hd_h$</td>
    <td>$H⋅d_h$</td>
    <td>$H⋅n_{kv}d_h$</td>
    <td>$Hd_c+d_c⋅n_hd_h$</td>
  </tr>
  <tr>
    <td>V 投影</td>
    <td>$H⋅n_hd_h$</td>
    <td>$H⋅d_h$</td>
    <td>$H⋅n_{kv}d_h$</td>
    <td>$d_c⋅n_hd_h$（与 K 共用 $W^{DKV}$）</td>
  </tr>
  <tr>
    <td>RoPE 分支</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>$d_h′⋅n_hd_r+Hd_r$</td>
  </tr>
  <tr>
    <td>$W^O$</td>
    <td>$n_hd_h⋅H$</td>
    <td>$n_hd_h⋅H$</td>
    <td>$n_hd_h⋅H$</td>
    <td>$n_hd_h⋅H$</td>
  </tr>
  <tr class="sep"><td></td><td></td><td></td><td></td><td></td></tr>
</table>

### KV cache对比

![](/images/2026-06/Pasted%20image%2020260608175328.png)

## MLA vllm框架代码实现

vLLM 对 DeepSeek V2/V3/R1 一类 MLA（Multi-head Latent Attention）的支持：
1. 在模型层把 DeepSeek 的普通 attention、未优化 MLA、优化 MLA 分成三条路径。
    - `DeepseekAttention`：普通 MHA，用传统 `Attention`。
    - `DeepseekV2Attention`：保留 DeepSeek MLA 数学结构，但把 latent KV 展开成普通 K/V 后走 `Attention`，不是 cache 优化路径。
    - `DeepseekV2MLAAttention`：真正的 vLLM MLA 优化路径，使用 latent KV cache 和 MLA backend。
2. 用 `MultiHeadLatentAttentionWrapper` 和 `MLAAttention` 把 MLA 的外层投影、RoPE、KV cache 写入和 attention backend 调用解耦。
3. 把 KV cache 从传统 `[K, V]` 双缓存改成每 token 一个 latent 向量：`[kv_lora_rank + qk_rope_head_dim]`。
4. 在推理时按 scheduler metadata 把 batch 拆成 prefill 和 decode：prefill 使用计算友好的 MHA 形式，decode 使用访存友好的 MQA 形式。

### 配置读取
```python
# 从 HuggingFace config 读取 MLA 维度。
# qk_nope_head_dim: 不参与 RoPE 的 Q/K 维度。
# qk_rope_head_dim: 参与 RoPE 的 Q/K 维度。
# v_head_dim: value head dim。
# kv_lora_rank: compressed KV latent rank。
qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
v_head_dim = getattr(config, "v_head_dim", 0)
kv_lora_rank = getattr(config, "kv_lora_rank", 0)

# DeepSeek v1 或没有 MLA 维度时，走普通 MHA。
use_mha = config.model_type == "deepseek" or all(
    dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
)

if use_mha:
    # 普通 MHA：传统 Q/K/V cache。
    attn_cls = DeepseekAttention
elif model_config.use_mla:
    # 优化 MLA：latent KV cache + MLA backend。
    attn_cls = DeepseekV2MLAAttention
else:
    # 数学上仍是 MLA，但会展开成普通 K/V 后走通用 Attention。
    # 这是禁用 MLA 优化时的回退路径。
    attn_cls = DeepseekV2Attention
```

### DeepseekV2MLAAttention初始化

```python
class DeepseekV2MLAAttention(nn.Module):
    """
    核心组件:
    - q_lora_rank: Q 的低秩维度
    - kv_lora_rank: KV 的低秩维度
    - fused_qkv_a_proj: 融合的 QKV 投影
    - q_a_layernorm, q_b_proj: Q 的低秩分解
    - kv_a_layernorm, kv_b_proj: KV 的低秩分解
    - o_proj: 输出投影
    """

    def __init__(self, ...):
        ...
        if self.q_lora_rank is not None:
            # DeepSeek V3/R1 常见路径：把 q_a_proj 和 kv_a_proj 融合成一个 GEMM。
            # 输出维度是 [q_lora_rank, kv_lora_rank + qk_rope_head_dim]。
            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
                proj_input_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
        else:
            # 没有 q_lora_rank 时，KV latent 单独投影。
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                proj_input_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        # KV latent 先 RMSNorm，再通过 kv_b_proj 展开成每个 head 的 k_nope + v。
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        # 输出投影。注意 MLA attention 内部先得到 [heads * v_head_dim]，
        # 最后仍需 o_proj 回到 hidden_size。
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
```

MLA 的核心参数关系：

```text
cache entry dim = kv_lora_rank + qk_rope_head_dim
q head dim      = qk_nope_head_dim + qk_rope_head_dim
value head dim  = v_head_dim
```

DeepSeek R1/V3 常见值：

```text
kv_lora_rank = 512
qk_rope_head_dim = 64
cache entry dim = 576
```

### MLA Wrapper前处理

```python
@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    def forward(self, positions, hidden_states, llama_4_scaling=None):
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            # 一次 fused projection 同时得到 Q 的低秩表示 q_c，
            # 以及 KV cache 所需的 kv_lora。
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]

            # q_c:      [tokens, q_lora_rank]
            # kv_lora:  [tokens, kv_lora_rank + qk_rope_head_dim]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )

            # Q 低秩表示归一化后上投影为完整 Q。
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            # 非 q_lora 路径：Q 和 KV latent 分开算。
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        # kv_c 是 compressed KV latent。
        # k_pe 是 decoupled rope key。
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # 重要：写入 KV cache 的不是原始 kv_c，而是 RMSNorm 后的 kv_c_normed。
        kv_c_normed = self.kv_a_layernorm(kv_c)

        # Q reshape 成 [tokens, local_heads, qk_head_dim]。
        q = q.view(-1, self.num_heads, self.qk_head_dim)

        # k_pe 是 MQA 形态：所有 head 共享一份 rope key，所以 head 维是 1。
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            # 只对 Q 的 rope 部分和 K 的 rope 部分做 RoPE。
            # Q 的 nope 部分不参与 RoPE。
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        # sparse MLA 时，额外算 top-k indices，供 sparse backend 选择历史 token。
        if self.indexer and self.is_sparse:
            _topk_indices = self.indexer(
                hidden_states, q_c, positions, self.indexer_rope_emb
            )

        # 进入通用 MLA attention 层。
        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )

        # attention 输出仍需 o_proj。
        return self.o_proj(attn_out)[0]
```

可以理解为 MLA 的"外层预处理"：
- Q 会形成完整 head 维。
- KV 不会展开成 K/V，而是保留 `kv_c_normed + k_pe`。
- cache 写入的数据就是 `kv_c_normed + k_pe`。

### MLAAttention 初始化：确定 cache entry 形态和 backend
```python
class MLAAttention(nn.Module, AttentionLayerBase):
    def __init__(...):
        ...
        self.num_heads = num_heads
        self.scale = scale
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.kv_b_proj = kv_b_proj

        # MLA cache 每个 token 只有一个向量：
        # [kv_lora_rank, qk_rope_head_dim]
        # 所以 head_size 不是 qk_head_dim，也不是 v_head_dim。
        self.head_size = kv_lora_rank + qk_rope_head_dim

        # MLA 的 KV head 固定为 1，因为 latent KV 是 MQA 风格共享的。
        self.num_kv_heads = 1
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # use_mla=True 是 backend 选择的关键。
        # use_sparse=True 时会选择 sparse MLA backend。
        self.attn_backend = get_attn_backend(
            self.head_size,
            dtype,
            kv_cache_dtype,
            use_mla=True,
            use_sparse=use_sparse,
            num_heads=self.num_heads,
        )

        # backend impl 接收 MLA 专用维度和 kv_b_proj。
        # kv_b_proj 后续用于 prefill 展开 K/V，也用于权重后处理拆 W_UK/W_UV。
        self.impl = impl_cls(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=1,
            kv_cache_dtype=self.kv_cache_dtype,
            attn_type=AttentionType.DECODER,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=kv_b_proj,
            indexer=indexer,
        )
```
- MLA cache 是 `[num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]`。
- backend 只需要处理 latent cache，普通 K/V 展开不是 cache 层的职责。

### Forward过程：先更新KV cache，再计算attention

```python
class MLAAttention(nn.Module, AttentionLayerBase):
    def forward(self, q, kv_c_normed, k_pe, output_shape=None):
        if self.calculate_kv_scales:
            # FP8 / quantized KV cache 场景可动态计算 q/k/v scale。
            torch.ops.vllm.maybe_calc_kv_scales(
                q,
                kv_c_normed,
                k_pe,
                _encode_layer_name(self.layer_name),
            )

        if self.use_direct_call:
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata

            # 多 layer metadata 以 dict 按 layer_name 存储。
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]

            self_kv_cache = self.kv_cache
            slot_mapping = forward_context.slot_mapping

            # slot_mapping 由 scheduler/KV manager 生成，决定当前 token 写到哪个 cache slot。
            # 区分不同的实现（例如Triton。FlashAttn, ...)
            self.impl.do_kv_cache_update(
                kv_c_normed,
                k_pe,
                self_kv_cache,
                slot_mapping.get(self.layer_name),
                self.kv_cache_dtype,
                self._k_scale,
            )

            # cache 已更新，随后执行 attention。
            output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
            self.forward_impl(
                q,
                kv_c_normed,
                k_pe,
                self_kv_cache,
                attn_metadata,
                output=output,
            )
            return output
```


```python
def do_kv_cache_update(
    self,
    kv_c_normed,
    k_pe,
    kv_cache,
    slot_mapping,
    kv_cache_dtype,
    k_scale,
):
    if kv_cache.numel() == 0:
        return

    # concat_and_cache_mla 会把两段拼起来：
    #   kv_c_normed: [tokens, kv_lora_rank]
    #   k_pe:        [tokens, qk_rope_head_dim]
    # 写入：
    #   kv_cache[slot] = [kv_c_normed, k_pe]
    ops.concat_and_cache_mla(
        kv_c_normed,
        k_pe.squeeze(1),
        kv_cache,
        slot_mapping.flatten(),
        kv_cache_dtype=kv_cache_dtype,
        scale=k_scale,
    )
```

### forward_impl：把 batch 拆成 prefill 和 decode

```python
fp8_attention = is_quantized_kv_cache(self.kv_cache_dtype)
num_actual_toks = attn_metadata.num_actual_tokens

# CUDA graph 会 padding 输入输出，这里只截取真实 token。
output = output[:num_actual_toks, ...]
q = q[:num_actual_toks, ...]
k_c_normed = k_c_normed[:num_actual_toks, ...]
k_pe = k_pe[:num_actual_toks, ...]

# sparse MLA backend 只支持 MQA/decode-style attention。
is_sparse_impl = isinstance(self.impl, SparseMLAAttentionImpl)

if is_sparse_impl:
    # sparse 情况下所有 token 都走 MQA。
    num_mqa_tokens = q.size(0)
    num_mha_tokens = 0
else:
    # 普通 MLA 由 metadata 告诉当前 batch 中多少 token 是 decode。
    num_mqa_tokens = attn_metadata.num_decode_tokens
    num_mha_tokens = q.size(0) - num_mqa_tokens

if num_mha_tokens > 0:
    # prefill 部分：compute-friendly MHA。
    self.impl.forward_mha(
        q[num_mqa_tokens:],
        k_c_normed[num_mqa_tokens:],
        k_pe[num_mqa_tokens:],
        kv_cache,
        attn_metadata,
        self._k_scale,
        output=output[num_mqa_tokens:],
    )

if num_mqa_tokens > 0:
    # decode 部分：data-movement-friendly MQA。
    mqa_q = q[:num_mqa_tokens]
    mqa_output_slice = output[:num_mqa_tokens]

    # q_nope: [B, heads, qk_nope_head_dim]
    # q_pe:   [B, heads, qk_rope_head_dim]
    mqa_q_nope, mqa_q_pe = mqa_q.split(
        [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    # 将 q_nope 投到 latent KV 空间：
    # [heads, B, P] x [heads, P, Lkv] -> [heads, B, Lkv]
    mqa_q_nope = mqa_q_nope.transpose(0, 1)
    torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)
    mqa_ql_nope = mqa_ql_nope.transpose(0, 1)

    # backend 的 MQA 输入是 latent query + rope query。
    mqa_q = (mqa_ql_nope, mqa_q_pe)

    # backend 从 paged latent KV cache 中读取历史 KV。
    attn_out, lse = self.impl.forward_mqa(mqa_q, kv_cache, attn_metadata, self)

    # MQA 输出仍在 latent value 空间，需要通过 W_UV 投回 v_head_dim。
    self._v_up_proj(attn_out, out=mqa_output_slice)
```
- Prefill token 多，$S_q/S_{kv}$ 比例更接近 1，展开 K/V 做 MHA 更划算。
- Decode token 少，主要瓶颈是读历史 KV cache，所以直接在 latent cache 上做 MQA 更省带宽。

### Prefill MHA：当前 token 与历史 context 分开算

```python
def forward_mha(self, q, kv_c_normed, k_pe, kv_cache, attn_metadata, k_scale, output):
    prefill_metadata = attn_metadata.prefill

    # 当前 prefill chunk 的新 token，需要从 latent KV 展开出 k_nope 和 v。
    kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
        -1,
        self.num_heads,
        self.qk_nope_head_dim + self.v_head_dim,
    )
    k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    # 完整 K = [k_nope, k_pe]。
    k = self._concat_k_nope_k_pe(k_nope, k_pe)

    # 当前新 token 之间做 causal attention。
    output_prefill = self._run_prefill_new_tokens(
        prefill=prefill_metadata,
        q=q,
        k=k,
        v=v,
        return_softmax_lse=has_context,
    )

    if has_context:
        # chunked prefill 有历史 context。
        # 历史 context 已在 KV cache 中，不能直接用当前 kv_c_normed。
        context_output, context_lse = self._compute_prefill_context(
            q,
            kv_cache,
            attn_metadata,
            k_scale,
        )

        # 当前 suffix 和历史 context 是两次 attention，需要按 LSE 稳定合并。
        merge_attn_states(
            output=output.view(-1, self.num_heads, self.v_head_dim),
            prefix_output=context_output,
            prefix_lse=context_lse,
            suffix_output=suffix_output,
            suffix_lse=suffix_lse,
            prefill_tokens_with_context=(
                prefill_metadata.chunked_context.prefill_tokens_with_context
            ),
        )
    else:
        # 没有历史 context，直接写输出。
        output_prefill = output_prefill[..., : v.shape[-1]].flatten(start_dim=-2)
        output.copy_(output_prefill)
```

- prefill 仍需要较多计算：历史 latent KV 省了 cache 空间，但参与 MHA 前必须上投影为普通 K/V。

### Decode MQA：直接在 latent cache 上算
> 这里以Triton实现为例，vllm/v1/attention/backends/mla/triton_mla.py
```python
def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
    assert kv_c_and_k_pe_cache.numel() > 0
    assert attn_metadata.decode is not None

    # 通用层可能传入 tuple: (ql_nope, q_pe)。
    # Triton backend 这里拼成完整 latent-space query。
    if type(q) is tuple:
        q = torch.cat(q, dim=-1)

    B = q.shape[0]
    q_num_heads = q.shape[1]

    # decode attention 的输出不是 v_head_dim，而是 kv_lora_rank。
    # 后续由通用层 _v_up_proj 投到 v_head_dim。
    o = torch.zeros(
        B,
        q_num_heads,
        self.kv_lora_rank,
        dtype=q.dtype,
        device=q.device,
    )

    # 根据 max_seq_len 和 SM 数估计把 KV 分成几段并行处理。
    ideal_splits = max(1, attn_metadata.max_seq_len // min_work_per_split)
    ideal_splits = triton.next_power_of_2(ideal_splits)
    num_kv_splits = min(ideal_splits, max_splits)

    # cache 原始 shape 是 [num_blocks, block_size, Lkv + R]。
    # 这里增加一个 head 维，方便复用 decode kernel 形态。
    kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)

    # latent value 使用 kv_c 部分。
    kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]

    decode_attention_fwd(
        q,
        kv_c_and_k_pe_cache,
        kv_c_cache,
        o,
        lse,
        attn_metadata.decode.block_table,
        attn_metadata.decode.seq_lens,
        attn_logits,
        num_kv_splits,
        self.scale,
        PAGE_SIZE,
        k_scale=layer._k_scale,
        v_scale=layer._k_scale,
        is_mla=True,
    )

    return o, lse
```
- cache 不展开成 `[heads, qk_nope_head_dim]` 的 K。
- Q 的 nope 部分先乘 $W^{{UK}}$ 投到 latent 维。
- attention value 直接用 `kv_c_cache`。
- attention 输出是 latent value，再乘 $W^{UV}$。


### KV cache

```python
def get_kv_cache_spec(self, vllm_config):
    kv_cache_dtype = kv_cache_dtype_str_to_dtype(
        self.kv_cache_dtype,
        vllm_config.model_config,
    )

    return MLAAttentionSpec(
        block_size=vllm_config.cache_config.block_size,

        # MLA latent KV 是 MQA 风格，所以 KV head 固定为 1。
        num_kv_heads=1,

        # 每个 token 存 [kv_c_normed, k_pe]。
        head_size=self.head_size,

        dtype=kv_cache_dtype,
        cache_dtype_str=vllm_config.cache_config.cache_dtype,
    )
```


```python
@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    cache_dtype_str: str | None = None

    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "fp8_ds_mla":
            # DeepSeek/FlashMLA sparse 的特殊格式：
            # 每 token 固定 656 bytes。
            return self.block_size * 656

        # 普通 MLA 页大小：
        # block_size * 1 * (kv_lora_rank + qk_rope_head_dim) * dtype_size。
        # 注意这里没有传统 attention 的 K/V 两份乘 2。
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )
```
对比普通 full attention：

```python
# FullAttentionSpec.real_page_size_bytes
return (
    2
    * self.block_size
    * self.num_kv_heads
    * self.head_size
    * get_dtype_size(self.dtype)
)
```

MLA cache 省内存的直接原因就是少了传统 K/V 双缓存，并且没有按所有 attention heads 展开。

```python
def get_kv_cache_shape(
    num_blocks,
    block_size,
    num_kv_heads,  # MLA 假设为 1。
    head_size,
    cache_dtype_str="auto",
):
    # MLA cache 逻辑 shape。
    # 不是 [2, num_blocks, block_size, heads, head_dim]。
    return (num_blocks, block_size, head_size)
```

典型 DeepSeek R1/V3：

```text
kv_cache.shape = [num_blocks, block_size, 576]
576 = 512 kv_lora_rank + 64 qk_rope_head_dim
```

Sparse FlashMLA 的 `fp8_ds_mla` 特例：
```python
if cache_dtype_str == "fp8_ds_mla":
    # DeepSeek 特化 packed cache 格式。
    return (num_blocks, block_size, 656)
else:
    return (num_blocks, block_size, head_size)
```

## 参考
1. https://arxiv.org/pdf/2405.04434
2. https://arxiv.org/pdf/2412.19437
3. https://zhuanlan.zhihu.com/p/16730036197
4. https://spaces.ac.cn/archives/10091
5. https://github.com/CalvinXKY/InfraTech/tree/main/models/deepseek_v3
6. https://github.com/vllm-project/vllm
