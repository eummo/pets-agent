# 项目优化变更记录

本文件记录已经落地的优化，避免和 `docs/optimization-backlog.md` 中的待办混在一起。

## 2026-06-10

- 新增 `src/loop` 领域模块，实现 Loop Engineering Phase 0：安全执行契约与最小状态机。
- `LoopService` 支持目标驱动的持续执行，含 run 级（queued/running/completed/failed/paused/cancelled）和 step 级（plan/act/observe/verify/decide）状态机。
- `InMemoryLoopStore` 提供定义/运行/步骤的 CRUD，含原子 claimStep CAS 和过期恢复查询。
- `ActionExecutor` 抽象执行缝隙，LoopService 不直接依赖 MessageGateway 或具体 agent runtime。
- `LoopExecutionContext` 贯穿 gateway、runtime、tool、log，支持按 loopRunId/stepId/attempt/idempotencyKey 追溯。
- AbortSignal 从 trigger 端到端传播到 ActionExecutor，支持 cancel 和 pause。
- 步骤租约（claimOwner + leaseExpiry）和 interrupted 状态确保中断步骤不盲目重放。
- 13 种结构化 loop 事件（`loop.started`、`loop.step.*`、`loop.completed` 等）写入 `system.jsonl`。
- `RoleCapability` 增加 `loop_manage`，控制 loop 定义的创建、启停、取消和查看。
- 新增 57 个 vitest 单元测试（35 store + 13 状态机 + 9 恢复与幂等）。
- 新增 6 个确定性冒烟案例（deterministic loop smoke），纳入 `npm run check` 质量门禁。
- 新增 `src/smoke/loopSmoke.ts` 真实 LLM 冒烟测试（`npm run smoke:loop`），验证 LoopService 与实际模型端到端工作。
- 新增 `docs/loop-engineering-upgrade-report.md`，记录 Phase 0-6 升级路线和设计决策。
- 更新 `docs/architecture.md`，增加 Loop Control Plane 层、loop 系统契约、`loop_manage` 能力和依赖规则。
- 更新 `README.md`，增加 `src/loop/` 架构条目、`smoke:loop` 命令和 loop 日志事件说明。
- 更新 `docs/development-workflow.md`，增加 loop 冒烟测试指导。

## 2026-06-06

- `package.json` 增加 `engines.npm`，明确 npm 10+。
- `.env.example` 补充 `AGENT_SDK_TYPE` 说明，避免把文档化变量误认为当前运行时选择入口。
- README 增加运维检查、系统日志和已知限制说明。
- WeCom 流式失败新增 `wechat.stream.failure` 事件与累计计数。
- Cron 缺少 app-message WeCom 配置时记录 `cron.wecom_config_missing` 告警事件。
- 新增 `/healthz` 和 `/readyz`，暴露进程存活与 SQLite、WeCom WSS、cron scheduler 就绪状态。
- JSONL logger 改为长生命周期 `WriteStream`，并提供 `flush()` / `close()`。
- Cron job 列表和详情暴露 `nextRunAt`、`lastStatus`、`lastError`，同时保留 `lastResult`。
- `system.jsonl` 周期记录 `wechat.session_metrics`，包含连接、活跃锁、inflight 和流式失败计数。
- 上传附件确认/描述类请求增加确定性 query 覆盖，避免模型把只读图片确认误判为 mutate 并拒绝。
- `docs/optimization-opportunities.md` 拆分为待办 `docs/optimization-backlog.md` 和本变更记录。
- WeCom 接入新增 `src/wechat/wecomSdkClient.ts` thin wrapper，`WechatSmartBotAdapter` 不再直接依赖 `@wecom/aibot-node-sdk` 类型。
- WeCom WSS 断连期默认继续处理已收到的消息，流式失败时走 HTTP fallback；如启用 `wechat.rejectWhenConnectionUnavailable`，消息不进入 gateway、不下载附件，并记录 `wechat.connection_unavailable_message_rejected` 与累计指标。
- 启动输出改为 `pets-agent startup` 横幅，集中展示 server、health、agent sdk、intent LLM、runtimes、WeCom WSS、cron、knowledge base、logs 和 state 路径，且不包含密钥字段。
- Cron scheduler 新增文件租约 leader guard，`/cron/status` 暴露当前实例 `leader` 状态，避免共享同一状态文件的多实例同时 tick。
- Conversation session/history 默认切换为 SQLite store，保留 `conversationStore: "file"` 作为轻量模式，降低多进程 JSON 文件同写风险。
- Conversation history archives 增加保留期清理任务与 `archived_at` 索引，默认保留 180 天并记录清理事件。
- CodeBuddy 真实 SDK smoke 接入受控 GitHub Actions，失败时保留 `.harness/codebuddy-smoke` artifact 便于排查 SDK/runtime 日志。
- Cron job/run state 默认切换为 SQLite store，保留 `cron.jobStore: "file"` 作为轻量模式；启动横幅会标明 `sqlite cron_jobs` 状态路径。
- `docs/architecture.md` 补充用户消息到 gateway/runtime/logs 的 mermaid 时序图；README 增加 WeCom、LLM 429、SQLite lock、cron leader 的 troubleshooting 入口。
- 增加 `npm run check:coverage` 可选入口，复用完整 check 流程并生成 Vitest 覆盖率报告，默认 `npm run check` 不强制覆盖率阈值。
- `src/wechat/wecomSdkClient.ts` 增加 SDK 隔离决策注释，说明 `@wecom/aibot-node-sdk` 官方性未完全证明、provider-specific 类型必须留在 wrapper 内。
- 增加 SQLite maintenance 工具与 `npm run db:backup` / `db:restore` / `db:verify`，并补充 `docs/sqlite-backup-restore.md` 备份、恢复、压缩演练说明。
- 增加 `docs/persistence-lifecycle.md`，明确 roles、feedback、conversation、cron、JSONL、附件和 file store 的生命周期、备份定位和生产使用边界。
- 增加 `src/dependencyRegression.test.ts`，覆盖 Zod 4 schema/default、TypeScript optional SDK auth 映射和 runtime factory cache key，作为依赖 patch/minor 升级的确定性回归样例。
- 增加 `docs/cron-production-plan.md`，把 cron 多副本生产化拆成外部调度器、Redis/Postgres leader election 和队列化执行三条路线，并定义迁移步骤与验收标准。

## 2026-05-25

- File conversation store 从 `src/core` 迁移到 `src/persistence`。
- 增加 `FileMutex`，降低单进程内文件 store 并发丢写风险。
- 拆分 server dev/wechat routes。
- 增加 LLM retry，并支持退避与 jitter。
- 收紧 dev role/feedback 本地访问。
- Runtime 错误响应改为更安全的用户提示。
- 意图分类增加确定性兜底。
- Runtime cache key 配置版本化。
- 增加 workspace resolver 缓存与诊断日志。
- Feedback 增加分页和索引。
- Dev 前端错误渲染与加载更多完成加固。
- 静态资源路径校验加固。
- 拆出 deterministic smoke。
- DB role config 增加运行时校验。
- 清理 `tsconfig.test.json` 测试范围。
- 完成 pi-ai、Vitest、Claude Agent SDK 等依赖升级窗口内的更新。
- Composition root 拆分出 runtime factory 等模块。
- 增加 workspace、role、permission、runtime 等 system 事件日志。
