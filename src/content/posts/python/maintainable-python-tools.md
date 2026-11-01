---
title: "让 Python 工具脚本具备产品级可维护性"
description: "不引入沉重框架，通过入口隔离、类型、配置与错误边界，让一次性脚本变成可靠工具。"
pubDate: 2026-10-18
category: "Python"
tags: ["Python", "Engineering", "CLI"]
cover: "/images/posts/python.jpg"
draft: false
---

很多实用工具都始于几十行 Python 脚本。需求增长之后，问题通常不是“代码不够高级”，而是输入、核心逻辑和副作用混在了一起。

## 把核心逻辑移出入口

命令行参数解析只负责构造输入和展示结果，文件写入、网络请求等副作用则留在边界层。核心函数因此可以脱离终端独立测试。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Report:
    total: int
    succeeded: int

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total else 0.0


def build_report(results: list[bool]) -> Report:
    return Report(total=len(results), succeeded=sum(results))
```

## 类型应该表达业务约束

类型标注不只是给编辑器看的提示。`NewType`、数据类和枚举可以阻止含义不同的值被意外混用，也能让函数契约更直观。

### 配置只在一个地方读取

不要在业务函数中散落环境变量读取。启动时集中校验配置，失败就给出具体提示，避免程序运行到一半才发现缺少必要参数。

## 错误信息要指向下一步

“请求失败”只能说明结果，不能帮助恢复。更好的错误信息应包含操作对象、失败原因和可执行的下一步，同时避免暴露凭证。

```python
def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}，请先复制示例配置")
    return Config.from_toml(path.read_text(encoding="utf-8"))
```

## 保留一条最简单的测试路径

为核心转换、边界值和失败分支编写表格驱动测试。无需一开始追求覆盖率，先确保最常用的行为不会因为一次重构悄悄改变。
