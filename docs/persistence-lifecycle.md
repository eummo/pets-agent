# 持久化数据生命周期

本项目的业务状态默认集中在 `config/runtime.json` 的 `dbPath` SQLite 文件中。JSONL 日志保留为审计与排障数据，不承担业务状态恢复职责。显式启用的 file store 只用于本地轻量模式或临时 harness，不作为生产多进程方案。

## 数据分类

| 数据                                     | 默认位置                                                      | 生命周期                                  | 备份/恢复                | 说明                                                         |
| ---------------------------------------- | ------------------------------------------------------------- | ----------------------------------------- | ------------------------ | ------------------------------------------------------------ |
| 角色配置 `roles`                         | SQLite `roles`                                                | 长期保存，随配置变更更新                  | 必须进入 SQLite 备份     | 包含角色 prompt、工具、权限、模型和 workflow 配置            |
| 用户反馈 `feedback`                      | SQLite `feedback`                                             | 长期保存，按人工处理状态流转              | 必须进入 SQLite 备份     | 用于跟进被拒绝但有价值的更新/修改请求                        |
| 当前会话 `conversation_sessions`         | SQLite `conversation_sessions`                                | 活跃对话期间保存，`/new` 或显式删除后更新 | 必须进入 SQLite 备份     | 维持 SDK session id，不应从 JSONL 反推恢复                   |
| 当前历史 `conversation_histories`        | SQLite `conversation_histories`                               | 活跃对话期间保存，会被 compact 或 archive | 必须进入 SQLite 备份     | 存储多轮上下文，受 `historyMaxMessages` 和 compact 影响      |
| 历史归档 `conversation_history_archives` | SQLite `conversation_history_archives`                        | 默认保留 180 天，后台清理过期归档         | 必须进入 SQLite 备份     | `conversationArchiveRetentionDays` 控制保留期                |
| Cron 定义 `cron_jobs`                    | SQLite `cron_jobs`                                            | 长期保存，管理接口创建/更新/删除          | 必须进入 SQLite 备份     | `cron.jobStore=file` 仅为本地轻量模式                        |
| Cron 运行状态 `cron_run_state`           | SQLite `cron_run_state`                                       | 随调度 tick 和执行结果更新                | 必须进入 SQLite 备份     | 包含 `nextRunAt` 和 `lastResult`                             |
| 对话日志 `conversation.jsonl`            | `logDir`                                                      | 审计日志，按运维策略轮转/归档             | 不作为业务恢复来源       | 记录用户输入、最终输出和 workspace                           |
| 系统日志 `system.jsonl`                  | `logDir`                                                      | 审计/排障日志，按运维策略轮转/归档        | 不作为业务恢复来源       | 记录权限、runtime、cron、WeCom、context usage 等事件         |
| LLM raw 日志 `llm-raw.jsonl`             | `logDir`                                                      | 模型/工具观测日志，按运维策略轮转/归档    | 不作为业务恢复来源       | 记录请求/响应/工具事件，严禁写入密钥                         |
| 附件缓存                                 | `wechat.uploadRootPath`                                       | 本地附件落盘，按业务保留策略清理          | 视业务需要单独备份       | 日志只记录附件数量和元数据，不记录下载 URL、aeskey 或 base64 |
| File store JSON                          | `sessionStorePath` / `historyStorePath` / `cron.jobStorePath` | 仅本地轻量模式                            | 启用 file 模式时单独备份 | 不推荐生产多进程共享写入                                     |

## 默认策略

- SQLite 是业务状态的主存储。生产备份范围应覆盖 `roles`、`feedback`、conversation session/history/archive 和 cron job/run state。
- JSONL 是审计和排障材料。它可以帮助重建事件链路，但不能作为自动恢复业务状态的来源。
- File conversation store 和 file cron store 只用于本地轻量模式、调试或短期 harness，不承担生产多进程一致性。
- 重型导出、归档压缩、VACUUM 和恢复演练应在离线窗口执行，不放进主服务请求路径。
- 恢复 SQLite 前先停止服务，保留当前库的二次备份，恢复后执行 `npm run db:verify` 并检查 `/readyz`。

## 运维入口

- SQLite 备份、恢复和校验：`docs/sqlite-backup-restore.md`
- 默认状态库：`config/runtime.json` 的 `dbPath`
- 默认日志目录：`config/runtime.json` 的 `logDir`
- 启动状态确认：`pets-agent startup` 横幅中的 `state:` 与 `logs:` 行
