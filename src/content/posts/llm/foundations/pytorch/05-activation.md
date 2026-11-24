---
title: "Lesson 5 PyTorch-Activation（激活函数与多层网络）"
description: "为什么必须有非线性：没有 ReLU 的多层网络会坍缩成一层，加了 ReLU 深度才有意义。"
pubDate: 2026-11-24T23:12:00
category: "LLM"
tags: ["PyTorch", "Activation", "学习笔记"]
cover: "/images/posts/llm-foundations.jpg"
draft: false
---

## 1. 为什么需要激活函数？

先看一个问题：把两层线性层叠起来，会发生什么？

```python
import torch
import torch.nn as nn

# 两层纯线性网络（没有激活函数）
model = nn.Sequential(
    nn.Linear(1, 1),   # 第一层：y = w1·x + b1
    nn.Linear(1, 1),   # 第二层：y = w2·y + b2
)

# 手动设置权重，验证等效性
with torch.no_grad():
    model[0].weight.fill_(2.0)   # w1 = 2
    model[0].bias.fill_(1.0)     # b1 = 1
    model[1].weight.fill_(3.0)   # w2 = 3
    model[1].bias.fill_(-0.5)    # b2 = -0.5

x = torch.tensor([[4.0]])
print(model(x))
# 输出：tensor([[23.5000]])
```

**核心推导**（不管权重具体是多少，结论都一样）：

```text
第二层输出 = w2·(w1·x + b1) + b2
           = (w2·w1)·x + (w2·b1 + b2)
           = W·x + B        ← 又变回了一层线性！
```

**纯线性层叠 100 层，数学上等价于 1 层。** 这样的网络永远学不会曲线、折线、语言——任何非线性规律。

## 2. 激活函数：给网络注入"弯折"能力

在层与层之间插入非线性函数，叠加就不再坍缩：

```python
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),          # ← 非线性：max(0, x)
    nn.Linear(16, 1),
)
```

ReLU 的定义极其简单：

```text
ReLU(x) = max(0, x)
负数 → 0，正数 → 原样保留
```

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0])
print(F.relu(x))
# 输出：tensor([0., 0., 0., 2., 5.])
```

"负的砍掉、正的保留"这么简单的操作，配上足够多的神经元，就能拼出任意复杂的曲线——这是**万能近似定理**的大白话版本。

下面两张图直观展示了差距（数据都是 `y = sin(x)`，网络结构、初始权重、训练轮数完全相同，唯一区别是有没有 ReLU）：

**不加 ReLU**——不管堆多少层，网络输出永远是一条直线，拟合不到曲线，loss 卡在 0.2 附近下不去：

![不加 ReLU：网络输出是一条直线](/images/posts/llm/pytorch/05-linear-collapse.png)

**加了 ReLU**——每个神经元贡献一个"折点"，折线拼接后逼近 sin 曲线，同样的结构 loss 直接降到 0.0001 级别：

![加了 ReLU：折线拼接逼近曲线](/images/posts/llm/pytorch/05-relu-fits-curve.png)

> 一句话总结：**ReLU 之前，深度是假的；ReLU 之后，深度才有意义。**

## 3. 常见激活函数速览

| 名称 | 公式 | 特点 | 使用场景 |
|---|---|---|---|
| Sigmoid | `1/(1+e⁻ˣ)` | 压到 (0,1) | 二分类输出层（现在内部层很少用） |
| Tanh | 压到 (−1,1) | 零中心化 | RNN 时代常用 |
| **ReLU** | `max(0,x)` | 快、简单、最常用 | CNN、MLP 隐藏层标配 |
| LeakyReLU | 负区间小斜率 | 避免神经元"死亡" | GAN 判别器常见 |
| GELU | 平滑版 ReLU | 概率化软化 | **Transformer 标配**，GPT/BERT 都用它 |
| SiLU/Swish | `x·sigmoid(x)` | 平滑、自门控 | Llama、现代 LLM 的 FFN 用它 |

**和大模型的关联**：Mini-GPT / Llama 里的前馈网络（FFN）用的就是 GELU 或 SiLU。现在只需记住「**隐藏层非线性 = 网络深度有意义**」。

## 4. 多层网络实战：拟合正弦曲线

线性模型永远拟合不出 sin 曲线。用「2 层 + ReLU」试试：

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# ========== 数据：y = sin(x) ==========
x = torch.unsqueeze(torch.linspace(-3.14, 3.14, 100), dim=1)  # (100, 1)
y = torch.sin(x)

# ========== 模型：1 → 32 → 32 → 1，隐藏层用 ReLU ==========
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # 上节课的 Adam

# ========== 训练 ==========
for epoch in range(1000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"epoch {epoch+1}, loss = {loss.item():.6f}")

# ========== 验证：抽几个点看预测 ==========
test_x = torch.tensor([[-3.0], [-1.5], [0.0], [1.5], [3.0]])
pred = model(test_x)
print("真实值:", torch.sin(test_x).squeeze().tolist())
print("预测值:", pred.squeeze().tolist())
```

运行结果：

```text
epoch 200, loss = 0.003120
epoch 400, loss = 0.000210
epoch 600, loss = 0.000032
epoch 800, loss = 0.000008
epoch 1000, loss = 0.000002
真实值: [-0.1411, -0.9975, 0.0000, 0.9975, 0.1411]
预测值: [-0.1425, -0.9959, 0.0012, 0.9949, 0.1445]
```

整个拟合过程如下（截取了 epoch 0 → 2000 的关键帧，前密后疏）：一开始网络输出是一条随机的陡直线，随着训练推进，折线逐渐"弯"出弧度，最终和 sin 曲线重合——

![ReLU 网络拟合 sin 曲线的过程](/images/posts/llm/pytorch/05-relu-fitting.gif)

### 代码拆解：`torch.unsqueeze` 是在干嘛？

```python
x = torch.unsqueeze(torch.linspace(-3.14, 3.14, 100), dim=1)  # (100, 1)
```

这行拆成两步看：

1. `torch.linspace(-3.14, 3.14, 100)`：在区间上**均匀等间隔**取 100 个点，得到一维张量，shape 为 `(100,)`。注意这 100 个 x 本身是均匀的、"线性"的——真正的非线性来自目标函数 `y = sin(x)`，网络要学的弯曲规律在 y 里。
2. `torch.unsqueeze(..., dim=1)`：在 `dim=1` 的位置插入一个新维度，shape 从 `(100,)` 变成 `(100, 1)`，即 100 行 × 1 列的"列向量"。

为什么必须升这一维？因为 `nn.Linear` 要求输入形状为 **`(batch_size, features)`**——"多少个样本、每个样本几个特征"。`(100, 1)` 的含义就是「100 个样本，每个样本 1 个特征」。直接把 `(100,)` 的一维张量喂进模型，矩阵乘法的维度对不上，会直接报错：

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (100x1 and ...)
```

一句话：**linspace 决定"在哪取点"，unsqueeze 决定"摆成什么队形"，sin 决定"学什么目标"。**

### 网络结构长什么样？

把上面 `model` 的结构画出来：1 个输入 → 32 个神经元（隐藏层 1）→ 32 个神经元（隐藏层 2）→ 1 个输出，层与层全连接：

![网络结构：1 → 32 → 32 → 1](/images/posts/llm/pytorch/05-network-structure.png)

对照代码看图：`nn.Linear(1, 32)` 负责"扇形展开"（1 维变 32 维），`nn.Linear(32, 32)` 是中间的"宽对宽"主干（参数量大头，占 88%），`nn.Linear(32, 1)` 负责最后"收拢"成 1 个预测值。两个 `nn.ReLU()` 分别插在隐藏层 1、2 的输出之后——没有它们，这 1153 个参数堆出来的依然只是一条直线。

单层线性网络对此束手无策（最多给出一条直线），加了 ReLU 的多层网络轻松拟合正弦曲线。**这就是"深度"的意义。**（数值为示意，实际因随机初始化略有差异。）

## 5. 本节练习

1. 把上面正弦拟合代码跑通，然后把两个 `nn.ReLU()` 都删掉再跑一次，对比 loss 的最终值（会明显下不去）。
2. 把隐藏层宽度从 32 改成 4，观察拟合效果变差——体会"宽度"和"深度"都是容量来源。
3. 思考题：ReLU 把负数全部砍成 0，为什么不会因此"丢信息"丢到没法学习？（提示：想想 32 个神经元各自学到不同区间。）

## 下一课预告

**Dataset 和 DataLoader：喂批量数据**——真实训练不是一次喂全部数据，而是分批（batch），下一课讲清 batch 的意义和 shuffle 的作用。
