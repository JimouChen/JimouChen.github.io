---
title: "Lesson 2 PyTorch-Dtypes（数据类型）"
description: "浮点、整型与 bool 的精度和范围速查，弄清 float32、bfloat16、int64 在大模型场景中的分工。"
pubDate: 2026-11-23
category: "LLM"
tags: ["PyTorch", "dtype", "学习笔记"]
cover: "/images/posts/llm-foundations.jpg"
draft: false
---

Tensor 里的每个元素都有固定类型，创建时可以指定，也可以事后转换。大模型学习阶段最常用的类型加粗标出。

## 一、浮点型（存小数，模型权重的类型）

| 类型 | 别名 | 有效数字 | 数据范围 | 说明 |
|---|---|---|---|---|
| **`torch.float32`** | `torch.float` | 约 7 位 | ±1.2 × 10⁻³⁸ ~ ±3.4 × 10³⁸ | 默认类型，深度学习标准精度 |
| **`torch.bfloat16`** | — | 约 3 位 | ±1.2 × 10⁻³⁸ ~ ±3.4 × 10³⁸ | 精度低但范围与 float32 相同，大模型训练/推理的主力类型 |
| `torch.float16` | `torch.half` | 约 3 位 | ±6.1 × 10⁻⁵ ~ ±6.5 × 10⁴ | 半精度，常用于推理和混合精度训练 |
| `torch.float64` | `torch.double` | 约 15~16 位 | ±2.2 × 10⁻³⁰⁸ ~ ±1.8 × 10³⁰⁸ | 双精度，主要用于科学计算 |

浮点范围的理解要点：

- **范围（能存多大的数）** 和 **精度（数有多准）** 是两回事。
- `bfloat16` 和 `float16` 精度差不多（都约 3 位有效数字），但 `bfloat16` 的范围和 `float32` 一样大，所以训练大模型时梯度不容易上下溢出，这是它成为主流的原因。
- `float16` 上限只有约 65504，训练时数值一旦超过就容易变成 `inf`（上溢）。

## 二、整型（存整数，token ID 的类型）

| 类型 | 数据范围 | 说明 |
|---|---|---|
| **`torch.int64`** | 约 ±9.2 × 10¹⁸ | **Tokenizer 输出的 token ID 默认类型**，范围绰绰有余 |
| `torch.int32` | -2,147,483,648 ~ 2,147,483,647（约 ±21.5 亿） | 一般整数 |
| `torch.int16` | -32,768 ~ 32,767 | 低精度整型 |
| `torch.int8` | -128 ~ 127 | 量化推理时会用到 |

## 三、其他

| 类型 | 数据范围 | 说明 |
|---|---|---|
| **`torch.bool`** | 只有 True / False | 用于 mask（注意力掩码、padding 掩码） |
| `torch.uint8` | 0 ~ 255 | 无符号 8 位整型 |

## 四、每种类型的数据长什么样（实际运行结果）

```python
import torch

f32 = torch.tensor([1.2345678, 2.0, 3.14159265])
print(f32, f32.dtype)
# tensor([1.2346, 2.0000, 3.1416]) torch.float32
# 精度：约 7 位有效数字

f64 = f32.to(torch.float64)
print(f64, f64.dtype)
# tensor([1.23456780, 2.00000000, 3.14159265], dtype=torch.float64)
# 精度：约 15~16 位有效数字，数字更长更精确

f16 = f32.to(torch.float16)
print(f16, f16.dtype)
# tensor([1.2346, 2., 3.1406], dtype=torch.float16)
# 精度：约 3 位有效数字，注意 3.14159 被截成 3.1406，尾部精度丢失

bf16 = f32.to(torch.bfloat16)
print(bf16, bf16.dtype)
# tensor([1.2344, 2., 3.1406], dtype=torch.bfloat16)
# 精度与 float16 类似（约 3 位），但能表示的数值范围与 float32 一样大

ids = torch.tensor([5, 101, 233, 9999, 152000])
print(ids, ids.dtype)
# tensor([     5,    101,    233,   9999, 152000]) torch.int64
# 注意：整数没有小数点，且自动推导为 int64，这就是 token ID 的样子

i8 = torch.tensor([127, -128], dtype=torch.int8)
print(i8, i8.dtype)
# tensor([127, -128], dtype=torch.int8)
# 范围 -128 ~ 127，超出就会溢出报错

b = torch.tensor([True, False, True])
print(b, b.dtype)
# tensor([True, False, True]) torch.bool
# 只有 True/False 两种值，就是 mask 的样子
```

## 五、类型转换（后面会反复用到）

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])

# 转 bfloat16（大模型常用）
x_bf16 = x.to(torch.bfloat16)
print(x_bf16.dtype)
# torch.bfloat16

# 转 long（token ID 常用）
ids = x.to(torch.int64)
print(ids.dtype)
# torch.int64
# tensor([1, 2, 3])，小数部分被截掉
```

## 六、大模型场景对照表

| 场景 | 类型 |
|---|---|
| 文本转成 token ID | `int64` |
| 模型权重（训练默认） | `float32` |
| 模型权重（大模型常用） | `bfloat16` |
| 推理量化 | `int8` / `float16` |
| 注意力掩码 / padding 掩码 | `bool` |

## 记忆要点

> **浮点默认 float32，token 是 int64，mask 是 bool。**
>
> 精度越低（float32 → bfloat16/float16 → int8），显存占用越小、速度越快，但数值精度越低。等微调阶段遇到显存不足时，再回来深入半精度细节即可。