# CodeBuddy CLI 概述

**腾讯云智能编程助手 - 让代码开发更智能、更高效**

CodeBuddy Code 是基于腾讯云 AI 技术的智能编程工具，深度集成腾讯云生态，提供从代码编写到项目部署的全链路 AI 辅助。

## 为什么选择 CodeBuddy Code？

### 用自然语言驱动整个开发运维生命周期

CodeBuddy Code 让您能够用自然语言描述需求，自动化完成从代码编写、测试、调试到部署的全链路开发任务，实现极致的自动化效率提升。无论是简单的代码修改还是复杂的架构重构，都能通过对话式交互轻松完成。

### 终端原生，无缝集成

- **熟悉的环境**：直接在您熟悉的命令行环境中获得 AI 辅助，无需切换开发工具或学习新界面
- **原生体验**：完美融入现有的开发工作流，支持所有主流操作系统和终端
- **零学习成本**：保持原有的开发习惯，AI 助手静默工作在后台

### 开箱即用的强大能力

- **内置工具链**：集成文件编辑、命令运行、Git 操作、测试执行等核心开发工具
- **智能提交**：自动生成规范的提交信息，支持代码审查和变更管理
- **灵活扩展**：通过 MCP (模型上下文协议) 轻松集成第三方工具和服务
- **自定义开发工具**：根据项目需求定制专属的开发助手

### Unix 哲学的 AI 集成

- **管道友好**：像 `grep` 和 `awk` 一样，原生支持管道输入进行智能分析
- **脚本集成**：完美融入 shell 脚本和自动化工具链
- **组合能力**：与现有 Unix 工具无缝组合，构建强大的 AI 驱动工作流
- **标准输入输出**：遵循 Unix 标准，支持重定向和管道操作

**管道集成示例：**

```bash
git log --oneline | codebuddy "分析这些提交，找出可能的问题"
cat error.log | codebuddy "帮我分析这些错误日志"
```

## 快速体验

### 环境要求

- Node.js 18.0+

### 一键安装

```bash
npm install -g @tencent-ai/codebuddy-code
```

### 开始使用

```bash
# 进入项目目录
cd my-project

# 启动 CodeBuddy
codebuddy
# 或
cbc

# 或直接提问
codebuddy "帮我优化这个函数的性能"
cbc "帮我优化这个函数的性能"
```

## 文档导航

### 入门指南

- [快速开始](https://www.codebuddy.ai/docs/zh/cli/quickstart)
- [安装指南](https://www.codebuddy.ai/docs/zh/cli/installation)
- [常见工作流](https://www.codebuddy.ai/docs/zh/cli/common-workflows)
- [交互模式](https://www.codebuddy.ai/docs/zh/cli/interactive-mode)
- [无头模式](https://www.codebuddy.ai/docs/zh/cli/headless)
- [故障排除](https://www.codebuddy.ai/docs/zh/cli/troubleshooting)
- [最佳实践](https://www.codebuddy.ai/docs/zh/cli/best-practices)

### 配置和扩展

- [设置](https://www.codebuddy.ai/docs/zh/cli/settings)
- [模型配置](https://www.codebuddy.ai/docs/zh/cli/models)
- [记忆](https://www.codebuddy.ai/docs/zh/cli/memory)
- [MCP 使用文档](https://www.codebuddy.ai/docs/zh/cli/mcp)
- [斜杠命令](https://www.codebuddy.ai/docs/zh/cli/slash-commands)
- [自定义快捷键](https://www.codebuddy.ai/docs/zh/cli/keybindings)

### 高级功能

- [子代理](https://www.codebuddy.ai/docs/zh/cli/sub-agents)
- [Agent Teams](https://www.codebuddy.ai/docs/zh/cli/agent-teams)
- [Skills 功能](https://www.codebuddy.ai/docs/zh/cli/skills)
- [Hooks 使用指南](https://www.codebuddy.ai/docs/zh/cli/hooks-guide)
- [插件系统](https://www.codebuddy.ai/docs/zh/cli/plugins)
- [检查点](https://www.codebuddy.ai/docs/zh/cli/checkpointing)
- [Git Worktree 支持](https://www.codebuddy.ai/docs/zh/cli/worktree)

### 安全

- [安全概述](https://www.codebuddy.ai/docs/zh/cli/security)
- [身份和访问管理](https://www.codebuddy.ai/docs/zh/cli/iam)
- [Bash 沙箱](https://www.codebuddy.ai/docs/zh/cli/bash-sandboxing)

### SDK

- [SDK 快速开始](codebuddy-sdk-quickstart.md)
- [TypeScript SDK 参考](codebuddy-sdk-typescript.md)
- [Python SDK 参考](codebuddy-sdk-python.md)
- [SDK Hook 系统](codebuddy-sdk-hooks.md)
- [SDK 权限控制](codebuddy-sdk-permissions.md)
- [SDK 会话管理](codebuddy-sdk-sessions.md)
- [SDK 自定义工具](codebuddy-sdk-custom-tools.md)
- [SDK MCP 集成](codebuddy-sdk-mcp.md)
- [SDK 示例项目](codebuddy-sdk-demos.md)

## 反馈和支持

- 技术支持：codebuddy@tencent.com
- 国内官方网站：https://copilot.tencent.com/cli
- 海外官方网站：https://www.codebuddy.ai/cli
