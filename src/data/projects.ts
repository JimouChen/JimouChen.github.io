export interface Project {
  name: string;
  description: string;
  stack: string[];
  github: string;
  demo?: string;
  status: 'Open Source' | 'Experiment';
  cover: string;
  featured?: boolean;
}

export const projects: Project[] = [
  {
    name: 'fk-sign-nte',
    description: '异环·塔吉多单账号自动签到工具，支持社区与角色签到、奖励查询、Token 安全轮换及 GitHub Actions 定时执行。',
    stack: ['Go', 'GitHub Actions', 'Automation', 'NaCl'],
    github: 'https://github.com/JimouChen/fk-sign-nte',
    status: 'Open Source',
    cover: '/images/projects/fk-sign-nte.jpg',
    featured: true,
  },
  {
    name: 'shortlink-system',
    description: '基于 Go Kitex 微服务、Redis 与 Vue 3 的前后端分离短链接系统。',
    stack: ['Go', 'Kitex', 'Redis', 'Vue 3'],
    github: 'https://github.com/JimouChen/shortlink-system',
    status: 'Open Source',
    cover: '/images/projects/shortlink-system.jpg',
    featured: true,
  },
  {
    name: 'NexusPushApp',
    description: '轻量级桌面客户端，支持本地文件上传远程服务器并自动执行命令。',
    stack: ['Vue', 'Electron', 'Desktop'],
    github: 'https://github.com/JimouChen/NexusPushApp',
    status: 'Open Source',
    cover: '/images/projects/nexus-push-app.jpg',
    featured: true,
  },
  {
    name: 'hertz-backend-base',
    description: '基于 Hertz 的 Go 后端 Web 服务基础框架，提供清晰的项目起点。',
    stack: ['Go', 'Hertz', 'Backend'],
    github: 'https://github.com/JimouChen/hertz-backend-base',
    status: 'Open Source',
    cover: '/images/projects/hertz-backend-base.jpg',
  },
  {
    name: 'py-handy',
    description: '实用、便捷的 Python 工具集，将重复的小需求沉淀为可复用能力。',
    stack: ['Python', 'CLI', 'Tools'],
    github: 'https://github.com/JimouChen/py-handy',
    status: 'Open Source',
    cover: '/images/projects/py-handy.jpg',
  },
  {
    name: 'SynapseClick',
    description: '可配置规则的浏览器自动点击扩展，检测目标 XPath 元素并触发操作。',
    stack: ['JavaScript', 'CSS', 'Extension'],
    github: 'https://github.com/JimouChen/SynapseClick',
    status: 'Experiment',
    cover: '/images/projects/synapse-click.jpg',
  },
  {
    name: 'reading-time-extension',
    description: '估算网页文本阅读时间的浏览器插件，让阅读安排更直观。',
    stack: ['JavaScript', 'HTML', 'Extension'],
    github: 'https://github.com/JimouChen/reading-time-extension',
    status: 'Experiment',
    cover: '/images/projects/reading-time-extension.jpg',
  },
];
