---
title: "SwiftUI 状态管理：从数据源开始思考"
description: "理解 State、Binding 与 Observable 的职责边界，用清晰的数据所有权减少视图刷新问题。"
pubDate: 2026-09-28
category: "iOS"
tags: ["SwiftUI", "Swift", "Architecture"]
cover: "/images/posts/ios.jpg"
draft: false
---

SwiftUI 的状态工具并不多，真正困难的是判断哪一份数据应该由谁拥有。先回答所有权问题，再选择属性包装器，代码通常会简单很多。

## 视图私有的临时状态使用 State

展开状态、输入框草稿和选中标签，如果只属于当前视图，就没有必要提升到全局模型。

```swift
struct SearchView: View {
    @State private var query = ""

    var body: some View {
        TextField("搜索文章", text: $query)
            .textFieldStyle(.roundedBorder)
    }
}
```

## 跨视图共享的是同一份事实

子视图需要修改父视图的数据时，传递 Binding，而不是复制一个 State。复制会让界面短暂看起来正确，却埋下同步问题。

```swift
struct SettingsView: View {
    @Binding var prefersReducedData: Bool

    var body: some View {
        Toggle("减少数据使用", isOn: $prefersReducedData)
    }
}
```

### 引用模型观察需要使用的字段

使用 Observable 模型时，让视图只读取自己真正关心的属性。巨型模型虽然方便传递，却可能让无关变化触发大范围的视图更新。

## 副作用跟随生命周期

网络请求应该由明确的任务触发，并在视图消失时取消。避免在视图 body 中直接执行副作用，因为 body 的求值次数不应成为业务逻辑的前提。

## 优先保证数据流可以解释

当界面出现异常刷新时，先画出数据从产生到展示的路径。能找到唯一数据源、知道每次修改发生在哪里，比引入更多状态工具更重要。
