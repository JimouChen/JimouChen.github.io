---
title: "Go Web 服务上线前的五个工程检查"
description: "一份偏实践的 Go 服务检查清单：超时、退出、并发、错误语义与基础可观测性。"
pubDate: 2026-10-26
category: "Go"
tags: ["Go", "Hertz", "Backend"]
cover: "/images/posts/golang.jpg"
draft: false
---

服务能够启动，只说明它通过了最基础的检查。真正进入生产环境之前，我通常会先确认下面五件事。

## 所有外部调用都有截止时间

HTTP 客户端、数据库查询和 RPC 调用都应该接收上下文。没有截止时间的请求，会把下游故障转化为本服务的资源耗尽。

```go
ctx, cancel := context.WithTimeout(req.Context(), 800*time.Millisecond)
defer cancel()

result, err := repository.FindByID(ctx, id)
if err != nil {
    return fmt.Errorf("find record %q: %w", id, err)
}
_ = result
```

## 退出时停止接收新请求

收到终止信号后，服务应先停止接收流量，再给进行中的请求一段合理的完成时间，最后关闭连接池等资源。

```go
server := &http.Server{Addr: ":8080", Handler: router}
go func() {
    _ = server.ListenAndServe()
}()

<-signalContext.Done()
shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
defer cancel()
_ = server.Shutdown(shutdownCtx)
```

## 并发有明确的所有权

每个 goroutine 都应该知道由谁启动、由谁等待、在何处退出。共享数据优先考虑通过消息传递或明确的所有权隔离，而不是临时补一把锁。

### 不要忘记资源边界

连接池大小、队列长度和批量任务的并发度都需要上限。无限增长往往不是性能设计，而是把问题推迟到更严重的时刻。

## 错误能够被分类和定位

向上返回的错误要保留上下文，但不要重复记录同一错误。调用方需要能够区分可重试错误与永久错误，日志则应包含必要的请求标识。

## 黄金信号先于复杂仪表盘

先保证延迟、流量、错误和饱和度可以被看到，再逐步增加业务指标。凌晨排障时，一个准确的基础指标通常比十张精美图表更有价值。
