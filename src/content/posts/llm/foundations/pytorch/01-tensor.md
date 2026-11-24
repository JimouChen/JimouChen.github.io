---
title: "Lesson 1 PyTorch-Tensor（张量）"
description: "大模型学习基础笔记：理解 Tensor 的创建、形状与基本运算，建立排查模型问题的第一直觉。"
pubDate: 2026-11-23T23:16:00
category: "LLM"
tags: ["PyTorch", "Tensor", "学习笔记"]
cover: "/images/posts/llm-foundations.jpg"
draft: false
---

## 1. 为什么先学 Tensor？

大模型里的数据基本都以 Tensor 表示：

- 文本经过 Tokenizer 后，变成 token ID 张量。
- 模型权重是张量。
- 注意力计算中的 Q、K、V 是张量。
- 训练过程中的梯度也是张量。

可以暂时把 Tensor 理解为：**支持 GPU 计算和自动求导的多维数组**。

## 2. 创建 Tensor

```python
import torch

# 标量：0 维张量
a = torch.tensor(3.0)

# 向量：1 维张量
b = torch.tensor([1.0, 2.0, 3.0])

# 矩阵：2 维张量
c = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

print("a:", a)
# 输出：a: tensor(3.)

print("b:", b)
# 输出：b: tensor([1., 2., 3.])

print("c:", c)
# 输出：c: tensor([[1., 2., 3.],
#                  [4., 5., 6.]])

print("a 的形状:", a.shape)
# 输出：a 的形状: torch.Size([])

print("b 的形状:", b.shape)
# 输出：b 的形状: torch.Size([3])

print("c 的形状:", c.shape)
# 输出：c 的形状: torch.Size([2, 3])

print("c 的数据类型:", c.dtype)
# 输出：c 的数据类型: torch.float32

print("c 的元素数量:", c.numel())
# 输出：c 的元素数量: 6
```

重点关注：

```python
c.shape
```

结果是 `torch.Size([2, 3])`，表示 `c` 有 **2 行、3 列**。以后分析模型报错时，`shape` 会是最常用的排查信息之一。

其中，`a` 是标量，没有任何维度，因此它的形状为 `torch.Size([])`；`b` 是包含 3 个元素的一维向量，因此形状为 `torch.Size([3])`。

## 3. Tensor 的基本运算

```python
import torch

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

# 每个元素乘以 2
y = x * 2

# 每个元素加 1
z = x + 1

# 所有元素求和
total = x.sum()

# 调整形状：2 行 3 列变成 3 行 2 列
reshaped = x.reshape(3, 2)

print("y:\n", y)
# 输出：y:
# tensor([[ 2.,  4.,  6.],
#         [ 8., 10., 12.]])

print("z:\n", z)
# 输出：z:
# tensor([[2., 3., 4.],
#         [5., 6., 7.]])

print("total:", total)
# 输出：total: tensor(21.)

print("reshaped:\n", reshaped)
# 输出：reshaped:
# tensor([[1., 2.],
#         [3., 4.],
#         [5., 6.]])

print("reshaped 的形状:", reshaped.shape)
# 输出：reshaped 的形状: torch.Size([3, 2])
```

注意：`reshape(3, 2)` 改变的是数据的组织形状，不会改变元素总数：

```text
原元素数量：2 × 3 = 6
新元素数量：3 × 2 = 6
```

## 4. 本节练习

独立完成以下练习：

1. 创建一个形状为 `(2, 4)` 的 Tensor，元素可以自行指定。
2. 打印它的 `shape`、`dtype` 和 `numel()`。
3. 将每个元素乘以 `3`。
4. 将它调整为形状 `(4, 2)`。
5. 求出所有元素的总和。

先不用查更复杂的 API，使用本节出现的内容就可以完成。

## 下一课预告

**自动求导 Autograd 与反向传播**：理解 `requires_grad`、`grad` 和 `backward()`，并手动完成一次最简单的梯度计算。