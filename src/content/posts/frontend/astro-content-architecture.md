---
title: "用 Astro Content Collections 设计内容架构"
description: "从 Schema、稳定 URL 到派生数据，搭建新增 Markdown 即可上线的静态博客内容系统。"
pubDate: 2026-10-09
category: "Frontend"
tags: ["Astro", "TypeScript", "SSG"]
cover: "/images/posts/frontend.jpg"
draft: false
---

内容站点的路由、列表和归档，都应该从同一份内容数据派生。这样新增文章时，作者只需要关注文章本身。

## 用 Schema 守住内容边界

Frontmatter 是内容与页面之间的契约。标题、摘要和日期需要校验，草稿默认值与标签默认值也应明确，避免错误直到构建末端才暴露。

```ts
const posts = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/posts' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
  }),
});
```

## URL 是长期承诺

 slug 一旦发布，就可能出现在搜索结果、RSS 阅读器和外部链接中。不要因为它来自文件路径，就随意根据标题或分类重构路径。

### 派生数据不要重复维护

分类数量、标签归档、相关文章与阅读时间都可以从文章集合计算。把计算集中在少量函数里，页面只负责呈现。

```ts
export async function getPublishedPosts() {
  const posts = await getCollection('posts', ({ data }) => !data.draft);
  return posts.sort((a, b) =>
    b.data.pubDate.valueOf() - a.data.pubDate.valueOf()
  );
}
```

## 静态不等于功能受限

全文搜索可以通过构建期索引完成，RSS 与 Sitemap 可以在构建时生成。没有运行时服务端，反而让缓存、部署与故障恢复更简单。

## 为阅读体验保留优先级

内容架构最终服务于阅读。稳定的排版、清晰的标题层级、移动端代码块和可访问的目录，比炫目的首页动画更值得优先投入。
