---
title: "Lesson 4 PyTorch-Nn-Module（搭建真正的神经网络）"
description: "用 nn.Module 管参数、nn.Linear 管运算、optimizer 管更新：把上一课手写的训练循环升级成工业级组件。"
pubDate: 2026-11-24T21:37:00
category: "LLM"
tags: ["PyTorch", "nn.Module", "学习笔记"]
cover: "/images/posts/llm-foundations.jpg"
draft: false
---

上一课手写了 `w`、`b` 和更新逻辑。参数一多，手写就乱了。PyTorch 提供了一套官方组件：**`nn.Module` 管参数、`nn.Linear` 管运算、`optimizer` 管更新**。

## 1. nn.Module：所有模型的基类

```python
import torch
import torch.nn as nn

# 定义模型：继承 nn.Module
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(in_features, out_features) 内部自动创建 w 和 b
        # 并自动设置 requires_grad=True
        self.linear = nn.Linear(1, 1)   # 输入 1 维 → 输出 1 维

    def forward(self, x):
        # 前向传播：定义"怎么算"
        return self.linear(x)            # 等价于 w·x + b

model = LinearModel()

# 参数不用自己管理，PyTorch 自动追踪
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)
# 输出：
# linear.weight torch.Size([1, 1]) True
# linear.bias torch.Size([1]) True
```

对比上一课：**w 和 b 的创建、求梯度、命名，全部自动完成**。GPT 也是这样——一个巨大的 `nn.Module`，内部几亿个参数全部被自动追踪。

## 2. 优化器：接管"更新参数"

```python
import torch

# SGD：最基础的优化器，lr 就是学习率
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 上一课手写的这两行：
#   w -= lr * w.grad
#   w.grad.zero_()
# 现在变成：
optimizer.step()        # 更新所有参数
optimizer.zero_grad()   # 清零所有梯度
```

### 常见优化器一览

| 优化器 | 核心思想 | 优点 | 缺点 |
|---|---|---|---|
| **SGD** | `w = w - lr × grad` | 简单、省显存 | 对学习率敏感、易震荡 |
| **SGD + Momentum** | 梯度基础上累加"惯性"（速度项） | 冲过平坦区和鞍点（第 3 课的逃离手段） | 仍需精细调学习率 |
| **Adagrad** | 为每个参数累积历史梯度平方，稀疏参数自动获得更大步长 | 适合稀疏特征 | 累积项只增不减，后期步长趋近 0 |
| **RMSprop** | 用滑动平均代替 Adagrad 的累积，步长不会枯竭 | RNN 时代的经典搭配 | 已逐渐被 Adam 取代 |
| **Adam** | Momentum + RMSprop 合体，逐参数自适应步长 | 收敛快、几乎不用调就有不错表现 | 显存占用大（额外存两份参数状态） |
| **AdamW** | Adam + 解耦的权重衰减（weight decay） | **Transformer 时代的标准答案**，GPT/BERT/Llama 全用它 | 显存占用同 Adam |
| **Adafactor** | 因式分解二阶矩，省显存 | 大模型训练显存紧张时的选择 | 实现细节更微妙 |
| **8-bit Adam / PagedAdamW** | 优化器状态量化成 8bit | 显存再省一大截，LoRA 微调常配 | 依赖 bitsandbytes 库 |
| **Lion** | 只用梯度的符号更新，省一整份状态 | 更省显存、增长快的新秀 | 对学习率更敏感 |

### 不同网络/任务怎么选（惯用组合）

| 任务/网络类型 | 推荐优化器 | 说明 |
|---|---|---|
| MLP 回归（入门） | SGD + Momentum 或 Adam | 都够用，Adam 更省心 |
| CNN 图像分类 | SGD + Momentum（经典配方）或 AdamW | ResNet 时代 SGD 调好后精度略高，但 AdamW 上手快 |
| RNN / LSTM | RMSprop 或 Adam | 处理梯度爆炸/衰减较稳 |
| **Transformer / NLP** | **AdamW** | 行业标准，配学习率 warmup + cosine 衰减 |
| 多模态（CLIP 类） | AdamW | 两塔都是 Transformer，同上 |
| GAN 生成模型 | Adam 或 RMSprop | 经验上 D 和 G 都稳 |
| 深度聚类 | Adam / AdamW | 见下方说明 |
| **大模型预训练** | **AdamW**（显存不够换 Adafactor / 8bit / Lion） | 十亿参数级 AdamW 状态本身就要数 GB |
| **LoRA 微调** | AdamW（或 8bit AdamW） | 微调阶段的事实标准 |

**关于聚类的一个澄清**：聚类本身（如 k-means）是经典算法，没有"训练"也没有优化器；只有"深度聚类"（先训自编码器/对比学习再做聚类）才涉及优化器选择，此时它就是个普通表征学习网络，用 Adam/AdamW 即可。

**最重要的认知**：优化器与任务类型是**正交的**——它只看到梯度，不关心梯度来自回归还是 NLP。上表是"惯用组合"，不是"专属绑定"。现代实践的无脑默认是 **AdamW**；显存告急时换它的省显存变体。我们后面写 Mini-GPT 时用的就是 AdamW。

## 3. 完整训练循环：拟合 y = 3x + 2

```python
import torch
import torch.nn as nn

# ========== 1. 准备数据 ==========
# 20 个样本，y = 3x + 2（模型要学会这两个数）
torch.manual_seed(0)
x = torch.unsqueeze(torch.linspace(-5, 5, 20), dim=1)  # 形状 (20, 1)
y = 3 * x + 2

# ========== 2. 定义模型、损失、优化器 ==========
model = LinearModel()                        # 用上面的 LinearModel
loss_fn = nn.MSELoss()                       # 均方误差：((y-预测)² 求平均)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# ========== 3. 训练循环 ==========
for epoch in range(200):
    y_pred = model(x)              # ① 前向传播
    loss = loss_fn(y_pred, y)      # ② 计算损失

    optimizer.zero_grad()          # ③ 清零梯度（第 3 课学的坑！）
    loss.backward()                # ④ 反向传播
    optimizer.step()               # ⑤ 更新参数

    if (epoch + 1) % 50 == 0:
        print(f"epoch {epoch+1}, loss = {loss.item():.6f}")

# ========== 4. 检查学到什么 ==========
print("学到的 w:", model.linear.weight.item())
print("学到的 b:", model.linear.bias.item())

# 用模型预测
print("预测 x=4 时:", model(torch.tensor([[4.0]])).item())
```

运行结果：

```text
epoch 50, loss = 0.012345
epoch 100, loss = 0.000234
epoch 150, loss = 0.000005
epoch 200, loss = 0.000000
学到的 w: 2.999998
学到的 b: 2.000001
预测 x=4 时: 13.999993
```

模型自己"学"出了 `w≈3, b≈2`，预测 `x=4` 得到 `≈14`——训练闭环完成。（数值为示意，实际会因随机初始化略有不同，但都会收敛到 3 和 2 附近。）

### 拟合过程动图

![梯度下降拟合 y=3x+2 的过程](/images/posts/llm/pytorch/fitting-y3x2.gif)

图中红点是真实数据（y = 3x + 2），蓝线是模型当前的 `y = wx + b`：

- 起始时 w、b 偏差很大，蓝线离红点很远，loss 很大。
- 随着每轮"求梯度 → 更新参数"，蓝线逐渐旋转、平移，贴近红点。
- 训练结束时 `w = 3.0000, b = 2.0000, loss ≈ 0`（本机纯 Python 复现了同样的训练循环并验证过）。

这张动图就是"训练"的直观含义：**一条随梯度不断自我修正的直线，最终贴住数据。**

## 4. 对上一课的升级对照

| 上一课（手写） | 本课（组件化） |
|---|---|
| `w = torch.tensor(2.0, requires_grad=True)` | `nn.Linear(1, 1)` 内部自动创建 |
| `y = w * x + b` | `model(x)` 自动调用 `forward` |
| `loss = (y - target) ** 2` | `nn.MSELoss()` |
| `w -= lr * w.grad` | `optimizer.step()` |
| `w.grad.zero_()` | `optimizer.zero_grad()` |

**关键理解**：训练循环的骨架完全没变，还是第 3 课流程图那 5 步——只是每一步换用了工业级组件。后面写 Mini-GPT 时，变的只是 `forward` 里面的内容。

## 5. zero_grad / backward / step 的顺序要求

### 两条硬约束

**约束 1：`zero_grad()` 必须在 `backward()` 之前**

因为梯度是累加的。如果清零放在 backward 之后、step 之前：

```python
loss.backward()        # 算出梯度
optimizer.zero_grad() # ❌ 刚算出来的梯度被抹掉了！
optimizer.step()      # 拿着 0 梯度更新 → 参数纹丝不动
```

**约束 2：`backward()` 必须在 `step()` 之前**

`step()` 的全部依据就是 `param.grad` 里的值：

```python
optimizer.step()     # ❌ 此时 grad 还是上一轮的旧值（或 None）
loss.backward()      # 算出新梯度，但已经晚了
```

### 一条有弹性的：zero_grad 放哪

放在循环**开头**或**结尾**都可以，效果等价：

```python
# 写法 A：开头清零（最常见）
for data in loader:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 写法 B：结尾清零（同样正确）
for data in loader:
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()   # 为下一轮做准备
```

只要保证「**backward 之前，grad 是干净的**」就行。

### 依赖关系图

```text
zero_grad ──→ backward ──→ step
（清空场地）  （生产梯度）  （消费梯度）

backward 的产出是 step 的输入 → 顺序不能换
zero_grad 的意义是保护 backward → 必须排在它前面（本轮或上一轮末尾）
```

### 顺带一个坑

`zero_grad()` 在新版本 PyTorch 中默认是把梯度设为 `None` 而不是填 0，两者效果一致，但如果手动读取 `param.grad` 判断"梯度是否算过了"，要注意区分 `None`（还没算）和 `tensor(0.)`（算了但为 0）。

`zero_grad → backward → step` 是标准写法，顺序不能随便换，尤其 **backward 和 step 颠倒**是最隐蔽的错误——程序不报错，但参数永远不会被正确更新。

## 6. 本节练习

1. 把上面代码跑通，改 `lr` 为 `0.5` 和 `0.01`，观察收敛速度的差异。
2. 把数据换成 `y = 5x - 1`，看模型能否学出新的 w、b。
3. 思考题：为什么 `optimizer.step()` 要放在 `loss.backward()` 之后？（提示：梯度是什么时候算出来的？）

## 下一课预告

**激活函数与多层网络**——为什么必须有非线性，没有它的多层网络会退化成一层。
