---
title: "Lesson 6 PyTorch-DataLoader（Dataset 和 DataLoader）"
description: "真实训练不是一次喂全部数据：讲清 batch 的意义、Dataset 与 DataLoader 的分工，以及 shuffle 为什么必须打开。"
pubDate: 2026-11-25T22:51:00
updatedDate: 2026-11-28
category: "LLM"
tags: ["PyTorch", "DataLoader", "学习笔记"]
cover: "/images/posts/llm-foundations.jpg"
draft: false
---

前几课的训练都是**一次把全部数据喂给模型**：100 个点一起算 loss、一起算梯度、更新一次。数据只有 100 个时没问题，但真实场景是百万张图片、GB 级文本——显存根本装不下。这一课解决"数据怎么喂"。

## 1. 为什么必须分批（batch）？

| 喂法 | 每次 update 用的样本数 | 名称 | 问题 |
|---|---|---|---|
| 全部一起喂 | N（全部） | 全批量梯度下降 | 数据大了显存装不下；一轮只更新 1 次，太慢 |
| 一次喂 1 个 | 1 | 随机梯度下降（SGD） | 梯度抖动大、利用率低、慢 |
| 一次喂一小批 | 16 / 32 / 64 ... | **小批量（mini-batch）** | 折中方案，现代训练的标准做法 |

核心收益：

1. **显存**：只装一个 batch，数据集多大都不怕。
2. **速度**：GPU 最擅长并行，一次算 32 个样本和算 1 个样本耗时几乎一样。矩阵运算远快于 Python 循环。
3. **更新次数**：100 个样本、batch_size=16，一轮（epoch）能更新 **7 次**而不是 1 次。
4. **正则化效果**：batch 的梯度带一点噪声，能帮助跳出不好的区域（第 3 课的鞍点、平坦区）。

> 术语：**epoch** = 把整个数据集完整过一遍；**iteration / step** = 处理一个 batch 并更新一次参数。

## 2. Dataset：定义"单个样本怎么取"

`Dataset` 是一个协议，只需要实现两个方法：

```python
import torch
from torch.utils.data import Dataset

class SineDataset(Dataset):
    """y = sin(x)，100 个样本"""

    def __init__(self, n=100):
        # 数据只在初始化时准备一次（真实场景：读文件路径、加载索引）
        x = torch.linspace(-3.14, 3.14, n).unsqueeze(1)   # (100, 1)
        self.x = x
        self.y = torch.sin(x)

    def __len__(self):
        return len(self.x)          # 告诉外界：一共有多少个样本

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]   # 给一个下标，返回 (样本, 标签)

dataset = SineDataset()
print(len(dataset))     # 100
print(dataset[0])       # 取第 0 个样本
# 输出：(tensor([-3.1400]), tensor([-0.0016]))
```

设计要点：

- **`__getitem__` 只负责"取一个"**，不管 batch——打包是 DataLoader 的事，职责分离。
- **惰性加载**：真实项目里 `__init__` 通常只存文件路径列表，`__getitem__` 被调用时才真正读图/读文本。百万张图的 Dataset 占用内存几乎为 0。

如果数据已经在内存里是 Tensor，可以不手写类，直接用现成的 `TensorDataset`：

```python
from torch.utils.data import TensorDataset

x = torch.linspace(-3.14, 3.14, 100).unsqueeze(1)   # (100, 1)
y = torch.sin(x)
dataset = TensorDataset(x, y)      # 每个样本返回 (x[i], y[i])
```

## 3. DataLoader：负责"打包成 batch"

```python
from torch.utils.data import DataLoader

dataset = SineDataset()

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,      # 每个 epoch 打乱顺序
)

print(len(loader), len(loader.dataset))
# 输出：7 100   ← len(loader) 是 batch 个数，len(loader.dataset) 才是样本数

# 看一个 batch 长什么样
xb, yb = next(iter(loader))         # 取出第一个 batch
print(xb.shape, yb.shape)
# 输出：torch.Size([16, 1]) torch.Size([16, 1])
```

**`shuffle=True` 打乱的到底是什么？**——是**样本的访问顺序**，也就是"哪些样本被分到同一个 batch"，数据本身不变。

DataLoader 内部维护一份样本编号列表 `[0, 1, 2, ..., 99]`。`shuffle=True` 时，每个 epoch 开始会把这个编号列表随机重排，再按 `batch_size` 切成几段，每段就是一个 batch。用 10 个样本、`batch_size=3` 举例：

```text
不打乱（每个 epoch 都一样）：
  epoch 1: [0,1,2] [3,4,5] [6,7,8] [9]
  epoch 2: [0,1,2] [3,4,5] [6,7,8] [9]

打乱（每个 epoch 重新洗一次）：
  epoch 1: [3,0,7] [1,9,4] [2,8,5] [6]
  epoch 2: [5,2,9] [0,7,1] [4,3,8] [6]
```

- **只有顺序变**：每个样本每个 epoch 仍然恰好被用一次，不会重复也不会漏。
- **数据不动**：只是取的编号变了，内存里的样本不会真的被重新排列。
- **批次组合跟着变**：同一个 epoch 内，哪些样本凑成一批也和上次不同。

`100 / 16 = 6.25`，向上取整得到 **7 个 batch**：前 6 个每个 16 个样本，最后一个只有 4 个。

**注意 shape 的变化**——这里接上了第 5 课的 `unsqueeze`：

```text
单个样本   dataset[0]      → (1,)       1 个特征
一个 batch  loader 里的 xb  → (16, 1)    16 个样本 × 1 个特征
                                        ↑
                            DataLoader 自动把 16 个 (1,) 堆叠成 (16, 1)
```

所以之前手写 `(100, 1)` 时，那个 100 其实就是"batch 维"。

整个流程用一张图串起来：Dataset 只管"按下标取单个样本"，DataLoader 负责"打乱编号 → 切批 → 堆叠"，最后交给循环的是一个 `(16, 1)` 的 batch：

![Dataset → DataLoader → batch 堆叠](/images/posts/llm/pytorch/06-dataloader-pipeline.png)

### 常用参数速览

| 参数 | 默认值 | 作用 | 什么时候改 |
|---|---|---|---|
| `batch_size` | 1 | 每批几个样本 | 几乎总要设，常用 16 / 32 / 64，显存允许越大越快 |
| `shuffle` | False | 每个 epoch 是否打乱 | **训练集必须 True**，验证/测试集用 False |
| `drop_last` | False | 最后不满一个 batch 的尾巴要不要丢 | 网络里有 BatchNorm 时设 True（batch_size=1 会报错） |
| `num_workers` | 0 | 用几个子进程加载数据 | 数据要读硬盘/解码图片时调大（如 4、8） |
| `pin_memory` | False | 锁页内存，加速 CPU → GPU 拷贝 | 用 GPU 训练时设 True |

## 4. 完整训练循环：用 DataLoader 重写 sin 拟合

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)

# ========== 1. 数据 ==========
class SineDataset(Dataset):
    def __init__(self, n=100):
        x = torch.linspace(-3.14, 3.14, n).unsqueeze(1)
        self.x, self.y = x, torch.sin(x)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = SineDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ========== 2. 模型、损失、优化器（和上节课一样） ==========
model = nn.Sequential(
    nn.Linear(1, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 1),
)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ========== 3. 训练：双层循环 ==========
for epoch in range(200):                 # 外层：过多少遍数据集
    total_loss = 0.0
    for xb, yb in loader:                 # 内层：每次取一个 batch
        y_pred = model(xb)                # ① 前向
        loss = loss_fn(y_pred, yb)        # ② 算 loss（只针对这 16 个样本）

        optimizer.zero_grad()             # ③ 清零
        loss.backward()                  # ④ 反向
        optimizer.step()                # ⑤ 更新 ← 每个 batch 都更新一次

        total_loss += loss.item() * len(xb)

    if (epoch + 1) % 50 == 0:
        print(f"epoch {epoch+1}, loss = {total_loss / len(dataset):.6f}")

# ========== 4. 验证 ==========
model.eval()                              # 切到评估模式（习惯，后面讲）
test_x = torch.tensor([[-3.0], [-1.5], [0.0], [1.5], [3.0]])
print("预测值:", model(test_x).squeeze().tolist())
```

运行结果：

```text
epoch 50, loss = 0.001890
epoch 100, loss = 0.000210
epoch 150, loss = 0.000041
epoch 200, loss = 0.000008
预测值: [-0.1412, -0.9970, 0.0005, 0.9968, 0.1410]
```

（数值为示意，实际因随机初始化和 shuffle 顺序略有差异。）对比第 5 课全量喂的写法，模型、损失、优化器一行没变，**变的只有"数据从哪来"**——训练骨架依然是那 5 步。

### 和上一课的结构对比

```text
上一课（全量）                     这一课（分批）
for epoch in range(...):           for epoch in range(...):
    y_pred = model(x)       ←┐         for xb, yb in loader:   ← 多了一层
    loss = loss_fn(...)       │ 同样       y_pred = model(xb)     ←┐
    zero_grad / backward      │ 的 5 步    loss / zero_grad          │ 同样的
    step                    ←┘            backward / step          │ 5 步
                                                              ←┘
一轮更新 1 次                       一轮更新 len(loader) = 7 次
```

**整个流程下来，参数一共更新了几次？**——**1400 次**：200 个 epoch × 每个 epoch 7 个 batch。

```text
for epoch in range(200):          ← 外层跑 200 遍
    for xb, yb in loader:         ← 内层每个 epoch 跑 7 遍（100 ÷ 16，向上取整）
        optimizer.step()          ← 每个内层循环执行 1 次
```

对比第 5 课全量喂的写法：同样 200 个 epoch，那边只在每个 epoch 结尾更新 1 次，总共 **200 次**。分批之后更新次数变成 7 倍，这也是 mini-batch 收敛更快的原因之一。

对应到下面的动图：epoch 1 慢放那一段，每个 batch 的第 ⑤ 步红线移动一次，7 个 batch 就是 7 次更新；之后每个 epoch 都是 7 次，直到 200 个 epoch 结束。

另外两个相关的数字：

- **样本被使用的次数**：每个样本每个 epoch 恰好用 1 次，所以每个样本一共被看了 200 次。
- **每次更新看的样本数**：每次 `step()` 只用 16 个样本算梯度（最后一个 batch 是 4 个），而不是全部 100 个。

### 训练过程逐帧动图

![DataLoader 分批训练的逐帧过程](/images/posts/llm/pytorch/06-dataloader-training.gif)

> 想一步一步看？打开[交互版逐帧演示](/demos/pytorch/dataloader/)：点击画面或按 `→` 进入下一帧，每帧配有当前步骤的说明。

画面分三块，对应上面的双层循环：

| 位置 | 看什么 |
|---|---|
| 左边 | 蓝线是 sin 真实曲线，红线是网络当前输出，橙色大点是**当前 batch 的样本**，橙色竖线是这些样本的误差 |
| 中间 | 100 个样本编号，颜色代表被分进了哪个 batch；黑框是正在处理的 batch |
| 右边 | 每个 batch 内的 5 步，橙色高亮是当前走到哪一步；下方实时显示 loss、梯度、total_loss |

播放节奏分三段：

1. **epoch 1（慢速）**：开头先展示打乱后切成的 7 个 batch，然后每个 batch 的 ①~⑤ 各停一帧。看右侧的 grad 那一行：①② 时它还是**上一批的旧梯度**，③ 清零，④ 算出新值，⑤ 红线移动。
2. **epoch 2（中速）**：每个 batch 更新一次后给一帧，能看到中间的黑框逐批移动，红线一段段弯出弧度。
3. **epoch 3 → 200（快速）**：每个关键 epoch 一帧，中间的颜色分布每个 epoch 都重新洗牌（`shuffle=True`），红线逐渐贴合蓝线。

两点说明：

- 动图用 numpy 复现了同样的结构（1→32→32→1 + ReLU + Adam，`lr=0.01`，`batch_size=16`），数值和上面的示意结果不完全一致。
- 各 epoch 的 loss 不是一路下降的，比如中途会比前一个关键帧略高——这是 mini-batch 的正常现象：每个 batch 的梯度只是 16 个样本的局部估计，带噪声。（动图里展示的是每个 epoch 内各 batch loss 的累加平均值。）

## 5. shuffle 的作用：为什么必须打乱？

先想一个问题：我们的数据是 `linspace` 生成的，**按 x 从小到大排好序**。如果 `shuffle=False`，每个 batch 会是什么样？

```text
不打乱：batch 1 = x∈[-3.14, -2.19]   batch 2 = x∈[-2.13, -1.17]  ...
        → 每个 batch 都是曲线上一小段，梯度只反映"局部"，方向偏来偏去

打乱后（示意）：batch 1 = [-0.5, 2.7, -1.3, 0.2, ...]
        → 每个 batch 都近似整个数据集的"缩影"，梯度方向有代表性
```

真实数据集更容易出事：图片按类别整理（前 1 万张全是猫、后 1 万张全是狗）。不打乱的话，模型先学一整轮猫、再学一整轮狗，学到狗就忘了猫。

**结论：训练集 `shuffle=True`，验证/测试集无所谓（不算梯度，不需要随机）。**

## 6. 训练集 / 验证集划分

真实训练必须留一部分数据不参与训练，用来检查"模型是背题了还是真学会了"：

```python
from torch.utils.data import random_split

full_dataset = SineDataset(n=100)

train_ds, val_ds = random_split(
    full_dataset,
    [80, 20],                                  # 80 个训练，20 个验证
    generator=torch.Generator().manual_seed(0) # 固定划分，结果可复现
)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=20, shuffle=False)

print(len(train_ds), len(val_ds))     # 80 20

# 验证：不计算梯度，只看误差（model、loss_fn 沿用第 4 节训练好的）
model.eval()
with torch.no_grad():
    val_loss = 0.0
    for xb, yb in val_loader:
        val_loss += loss_fn(model(xb), yb).item() * len(xb)
    print(f"val loss = {val_loss / len(val_ds):.6f}")
```

两个新东西，先记住用途，细节放到后面的课：

- `model.eval()`：把模型切到"评估模式"，让 Dropout、BatchNorm 这类训练/推理行为不同的层进入推理状态。
- `torch.no_grad()`：关闭梯度追踪，省显存、省计算（验证不需要反向传播）。

## 7. 常见坑

**坑 1：一个 epoch 的 loss 怎么平均？**

最后一个 batch 只有 4 个样本，直接 `sum(loss.item()) / len(loader)` 会给这 4 个样本和小 batch 同等的权重。正确写法是按样本数加权：

```python
total_loss += loss.item() * len(xb)     # loss 是均值 → 乘回去变"总和"
epoch_loss = total_loss / len(dataset)   # 再除以总样本数
```

**坑 2：`len(loader)` 不等于 `len(dataset)`**

`len(loader)` 返回的是 **batch 个数**（7），`len(dataset)` 是样本数（100）。算平均 loss、打印进度时别混用。

**坑 3：忘了 `shuffle=True`**

程序不报错，loss 也在降，但收敛更慢、效果更差。很隐蔽——写 `DataLoader` 时把 `shuffle=True` 当成训练集的默认动作。

**坑 4：把 batch 维度弄丢**

如果 Dataset 返回的是 `(1,)` 形状的样本，DataLoader 会堆成 `(16, 1)`，正好符合 `nn.Linear` 要的 `(batch_size, features)`。但如果 `__getitem__` 里顺手做了 `.squeeze()` 把样本压成 0 维标量，堆出来就是 `(16,)`，进模型会报维度错误。**样本的形状应该是 `(features,)`，batch 维交给 DataLoader 加。**

**坑 5：`num_workers` 在 Mac / Jupyter 上报错**

子进程的启动方式是 spawn，Dataset 必须能被序列化（定义在 notebook 单元格里的类不行）。入门阶段保持默认 `num_workers=0` 最省事。

## 8. 本节练习

1. 把第 4 节代码跑通，把 `batch_size` 依次改成 `4`、`32`、`100`，观察 `len(loader)` 的变化和收敛速度。
2. 把 `shuffle` 改成 `False` 再跑一次，对比 loss 曲线——因为数据是按 x 排序的，差别会比较明显。
3. 设 `drop_last=True`，确认 `len(loader)` 从 7 变成 6。
4. 思考题：batch_size 是不是越大越好？从显存、更新次数、梯度噪声三个角度想想。

## 下一课预告

**损失函数与分类任务**——回归用 MSE，那分类呢？讲清 CrossEntropyLoss 和 softmax 的关系，以及为什么分类网络的最后一层不要加激活函数。
