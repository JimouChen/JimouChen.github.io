export const site = {
  name: 'Neaya',
  title: 'Neaya Blog',
  description: '记录 Python、Go、AI Agent 与独立开发，分享可靠、优雅且值得回读的技术内容。',
  url: 'https://jimouchen.github.io',
  author: 'Neaya',
  github: 'https://github.com/JimouChen',
  email: '1625645320@qq.com',
} as const;

export const navigation = [
  { label: '首页', href: '/' },
  { label: '博客', href: '/blog/' },
  { label: '项目', href: '/projects/' },
  { label: '时间轴', href: '/timeline/' },
  { label: '关于', href: '/about/' },
] as const;
