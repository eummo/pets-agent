# 当前项目优化项清单

生成日期：2026-05-25

本清单基于当前仓库静态扫描、关键路径阅读和 `npm run check` 结果整理。项目当前类型检查、lint、单元测试和构建均通过；以下内容不是已确认线上故障，而是按稳定性、性能、可维护性和扩展性排序的后续优化机会。

更新说明：2026-05-25 二次检查时，工作树中已经包含一批优化改动，包括 file conversation store 从 `src/core` 迁移到 `src/persistence`、增加 `FileMutex`、拆分 server dev/wechat routes、增加 LLM retry、收紧 dev role/feedback 本地访问，以及将 runtime 错误响应改为更安全的用户提示。下面的清单已按当前工作树重新校准。

实施状态：2026-05-25 已按建议实施顺序完成大部分优化，包括意图分类确定性兜底、runtime cache key 配置版本化、`FileMutex` 清理与并发测试、JSONL logger 写队列、workspace resolver 缓存与诊断日志、feedback 分页/索引、dev 前端错误渲染与加载更多、retry 退避、静态资源路径加固、deterministic smoke、DB role config 运行时校验、`tsconfig.test.json` 清理、pi-ai/Vitest patch 升级、composition root 拆分和 system 事件日志。Claude Agent SDK patch 升级暂缓，因为 `0.3.150` 要求 Zod 4，而当前项目仍在 Zod 3。

## 摘要

| 优先级 | 数量 | 主要方向                                                                                                |
| ------ | ---: | ------------------------------------------------------------------------------------------------------- |
| P0     |    2 | 权限分类失败策略、运行时配置生效                                                                        |
| P1     |    8 | 文件锁完善、日志吞吐、工作区解析缓存、SQLite 查询边界、前端安全、retry 策略、静态资源路径、smoke 稳定性 |
| P2     |    5 | 类型校验、配置清理、依赖升级、代码组织收尾和可观测性增强                                                |

最新剩余优先处理项：

1. 单独规划 Zod 4 + Claude Agent SDK patch 升级。
2. 继续评估 ESLint 10、TypeScript 6、Zod 4 等大版本升级。
3. 若后续进入多进程部署，将 session/history file store 迁移到 SQLite 或增加跨进程文件锁。

## P0 优化项

### P0-1. 意图分类失败默认放行为 query，权限策略偏乐观

证据位置：

- `src/intent/llmIntentDetectionService.ts:55`、`:68`、`:70` 在模型错误、非法标签、异常时都返回 `{ type: "query" }`。
- `src/core/orchestrator.ts:73` 依赖分类结果决定是否走 mutate/update_kb 权限检查。

风险：

当 intent LLM 超时或端点异常时，明显的“修改、删除、更新文档”等请求会绕过修改意图识别，进入 reviewer runtime。虽然 reviewer 工具有额外限制，但用户体验和权限语义会变得不稳定。

建议：

- 增加确定性关键词兜底：中文如“修改、删除、更新、添加、写入、创建”，英文如 “change, update, delete, create, edit, implement”。
- 对分类服务异常写入内部日志，但不要把 provider 原始错误暴露给用户。
- 对高风险动词采用 fail-closed：无法确认时按 mutate 或 update_kb 处理并记录 feedback。

验证：

- 增加 `LlmIntentDetectionService` 单元测试，覆盖 timeout/error/非法标签下的中文和英文修改请求。
- 在 `src/smoke/regressionSmoke.ts` 增加“分类服务不可用时 reviewer 修改请求仍被拒绝”的运行时回归。

### P0-2. Runtime 缓存只按 role 保存，角色配置更新后不会自动生效

证据位置：

- `src/core/orchestrator.ts` 已将缓存改为 `Map` 并限制最大数量，但 cache key 仍然是 role。
- `src/index.ts` 启动时把 DB 中的 role config 转成 runtime。

风险：

如果角色配置、模型、工具列表或权限模式在数据库中被更新，已有进程仍可能继续使用旧 runtime，直到服务重启。后续增加管理页面或 API 修改角色配置时，这会成为明显的行为不一致。

建议：

- 给 `StoredRoleConfig` 增加 `updatedAt` 或版本号，并把 runtime cache key 扩展为 `role + version`。
- 或在 role config 更新路径中发布 invalidation 事件，让 orchestrator 清理对应 role runtime。
- 增加测试：修改 role allowedTools 后，同一进程下一次请求应使用新配置。

## P1 优化项

### P1-1. JSONL logger 每条日志都 mkdir + writeFile append，缺少写入队列

证据位置：

- `src/logging/jsonlLogger.ts:15` 每次写日志都 `mkdir`。
- `src/logging/jsonlLogger.ts:16` 每次写日志都 `writeFile(..., { flag: "a" })`。

影响：

吞吐较低时问题不大，但 smoke、流式事件或多用户并发时会重复做目录检查和打开文件。并发 append 虽通常可用，但没有统一 backpressure、flush 或 close 语义。

建议：

- logger 创建时确保目录存在一次。
- 使用内部 promise queue 串行化 append，或使用 `createWriteStream` 并提供 `flush/close`。
- 继续保留当前密钥脱敏测试，并新增并发写入行数测试。

### P1-2. 工作区解析每次请求都读 repos.json，且解析失败静默回退

证据位置：

- `src/workspace/configuredWorkspaceResolver.ts:53` 每次匹配前调用 `loadRepositories()`。
- `src/workspace/configuredWorkspaceResolver.ts:65` 每次从磁盘读取配置。
- `src/workspace/configuredWorkspaceResolver.ts:67` catch 后直接返回空数组。

影响：

每次消息都有额外 I/O。更重要的是，配置文件 JSON 损坏或 schema 不匹配时会静默回退到知识库，导致问题难以排查。

建议：

- 加入基于 mtime 的轻量缓存，配置变化时自动刷新。
- 配置解析失败时写入结构化日志，并在开发环境暴露诊断信息。
- 测试覆盖：合法配置缓存、配置变更刷新、非法配置可观测。

### P1-3. SQLite store 缺少分页和部分索引，后台数据增长后会拖慢管理接口

证据位置：

- `src/persistence/sqliteFeedbackStore.ts:74` 的 `getAll()` 返回所有 feedback。
- `src/persistence/sqliteFeedbackStore.ts:90` 仅按 id 倒序排序。
- `src/persistence/sqliteConnection.ts:16` 的 feedback 表没有按 status/user/workspace 的索引。

影响：

本地 harness 数据量小没有问题，但如果 feedback 成为长期管理入口，全量返回会让 `/dev/feedback` 响应和前端渲染越来越慢。

建议：

- `FeedbackStore.getAll` 改成支持 `limit/offset/status`。
- 增加索引：`feedback(status, id)`、`feedback(user_id, id)`，如后续常按 workspace 查，再加 `workspace_path`。
- 前端反馈页做分页或“加载更多”。

### P1-4. Dev 前端仍有少量 innerHTML 拼接错误文本

证据位置：

- `static/dev-chat/app.js:245` 把服务端错误拼进 `innerHTML`。
- `static/dev-chat/app.js:259` 把异常消息拼进 `innerHTML`。
- ESLint 当前在 `eslint.config.js:14` 忽略了 `static/**`。

风险：

该页面是本地开发页，风险范围较小，但如果将来作为真实管理页面暴露，错误消息中的 HTML 会带来 XSS 风险。

建议：

- 用 DOM API 创建错误节点并设置 `textContent`。
- 把 dev 前端迁移到 TypeScript 或至少加入 lightweight JS lint。
- 给反馈页错误展示加浏览器级回归测试或单元化 DOM 测试。

### P1-5. Smoke 依赖真实服务和模型，缺少“可控模型失败”场景

证据位置：

- `src/smoke/regressionSmoke.ts:48` 之后直接跑真实 `/dev/chat`。
- `src/smoke/regressionSmoke.ts:249` 到 `src/smoke/regressionSmoke.ts:298` 对 pi-ai 做真实模型调用。

影响：

当前 smoke 能验证真实集成，这是优点；但它也会受模型可用性、额度、网络和提示词波动影响。对于权限拒绝、日志、安全这些确定性行为，应该有不依赖模型的 smoke 子集。

建议：

- 拆出 `npm run smoke:deterministic`，使用 echo/faux provider 或可控 stub。
- 保留 `npm run smoke` 作为真实模型回归。
- 把“模型不可用时权限仍保守”的 case 放到 deterministic smoke 中。

### P1-6. `FileMutex` 已降低同进程丢写风险，但还缺少锁生命周期和并发测试

证据位置：

- `src/persistence/fileStoreUtils.ts` 新增了 `FileMutex`。
- `src/persistence/fileConversationSessionStore.ts` 和 `src/persistence/fileConversationHistoryStore.ts` 已在写路径使用 `mutex.acquire()`。
- 当前 `src/persistence/fileConversationSessionStore.test.ts`、`src/persistence/fileConversationHistoryStore.test.ts` 仍主要覆盖串行读写，没有覆盖并发写入。

风险：

这是对上一版 P0 的重要修复，但仍有两个边界：`FileMutex` 的 `locks` map 不删除已完成 key，长期动态文件路径会增长；该锁只覆盖当前 Node 进程，若未来多进程部署或多个 worker 同写同一 JSON 文件，仍无法提供跨进程事务。

建议：

- 在 release 时只在当前 promise 仍是最新锁时删除 map key。
- 增加并发单元测试：并发 `set` 多个 session、并发 `append` 同一 history，断言最终记录完整且顺序可接受。
- 在文档中明确 file store 是单进程开发/轻量部署方案；生产化优先迁移 SQLite。

### P1-7. Retry 逻辑固定间隔，缺少退避、抖动和可观测性

证据位置：

- `src/config/retry.ts` 新增了通用 `withRetry`。
- `src/intent/llmIntentDetectionService.ts` 已接入意图分类和 Bash 权限分类。

风险：

固定 500ms retry 对瞬时波动有帮助，但多个请求同时失败时会同步重试，可能放大 provider 压力。当前也没有记录 retry 次数和最终失败原因，不利于排查分类或权限判断不稳定。

建议：

- 支持指数退避和 jitter。
- 区分可重试错误与不可重试错误，例如鉴权失败、配置错误不应重试。
- 在内部日志记录 retry attempt、duration、最终结果，但不要记录 API key 或完整 provider payload。

### P1-8. Dev route 已拆分并增加本地限制，但静态资源路径校验仍可更稳

证据位置：

- `src/server/devRoutes.ts` 已把 dev 页面、role、feedback 路由拆到独立模块。
- role 和 feedback 管理接口已检查 `isLocalRequest(request.ip)`。
- 静态资源路由仍使用 `path.join(devChatDir, relativePath)` 后做 `startsWith(devChatDir)`。

风险：

`startsWith` 对路径边界比较脆弱，例如目录名共享前缀时容易误判。当前 Fastify 参数和 `path.join` 已降低风险，但静态文件服务最好用 `path.resolve` 后比较相对路径是否逃逸。

建议：

- 改为 `const resolved = path.resolve(devChatDir, relativePath)`，再用 `path.relative(devChatDir, resolved)` 判断是否以 `..` 或绝对路径开头。
- 给 `/dev/chat/*` 增加路径穿越测试，覆盖 URL 编码后的 `..`。

## P2 优化项

### P2-1. 数据库 JSON 字段读取只做类型断言，缺少运行时校验

证据位置：

- `src/persistence/sqliteRoleConfigStore.ts:15` 解析 capabilities 后直接断言类型。
- `src/persistence/sqliteRoleConfigStore.ts:19` 解析 allowedTools 后直接断言 `string[]`。

建议：

- 复用或新增 zod schema 校验 DB 中的 role config。
- 对损坏配置返回可诊断错误，避免把非法 permissionMode 或 tools 传入 runtime。

### P2-2. `tsconfig.test.json` 指向不存在的旧测试路径

证据位置：

- `tsconfig.test.json:8` include 了 `src/tasks/agent-manager.test.ts` 和 `src/memory/injector.test.ts`，当前仓库没有这些路径。

影响：

当前 `vitest.config.ts` 使用 `src/**/*.test.ts`，所以测试实际不受影响；但旧配置会误导后续维护者，也可能在 IDE 或脚本中造成混乱。

建议：

- 删除无用的 `tsconfig.test.json`，或改成当前测试范围。
- 如果保留，明确它的用途，并让 `npm run typecheck` 或 Vitest 配置实际引用它。

### P2-3. 依赖存在可控升级窗口

证据位置：

- `npm outdated --json` 显示 `@anthropic-ai/claude-agent-sdk` 当前 `0.3.146`，wanted/latest 为 `0.3.150`。
- `@earendil-works/pi-ai` 当前 `0.75.4`，wanted/latest 为 `0.75.5`。
- `vitest` / `@vitest/coverage-v8` 有 patch 版本 `4.1.7`。
- ESLint、TypeScript、Zod 有大版本更新，但风险更高，适合单独批次评估。

建议：

- 先做低风险 patch/minor：Claude Agent SDK、pi-ai、Vitest patch。
- 大版本升级分批处理：ESLint 10、TypeScript 6、Zod 4 各自单独分支，先读 release notes，再跑 check/smoke。

### P2-4. Composition root 偏长，配置加载和 runtime 构建可以继续拆小

证据位置：

- `src/index.ts` 仍同时负责环境读取、SDK 环境变量配置、SQLite 初始化、角色 seed、runtime 构建、server 创建和启动。
- server route 已拆分，这是好的第一步；runtime factory 和 config 解析仍可继续独立出来。

建议：

- 拆出 `src/config/runtimeConfig.ts` 负责环境变量解析和默认值。
- 拆出 `src/agent/createAgentRuntimes.ts` 负责 runtime factory 和缓存策略。
- 保持 `src/index.ts` 作为 composition root，但让主流程更短、更容易测试。

### P2-5. 可观测性可以补充分类、权限和 workspace 选择事件

证据位置：

- 当前 `src/core/orchestrator.ts:217` 的 conversation log 主要记录输入输出。
- `src/agent/claude/claudeSdkAgentRuntime.ts`、`src/agent/codebuddy/codebuddySdkAgentRuntime.ts`
  和 `src/agent/pi/piEventCollector.ts` 记录 runtime response。

建议：

- 增加内部事件：workspace resolved、role resolved、intent classified、permission denied、runtime selected。
- 继续沿用 `JsonlLogger` 的脱敏逻辑，不记录 authorization header、token、API key。
- smoke 中验证关键内部事件存在，尤其是被拒绝请求没有进入 runtime。

## 建议实施顺序

1. P0-1 意图分类失败的确定性保守兜底。
2. P0-2 runtime cache invalidation。
3. P1-6 完善 `FileMutex` 清理和并发测试。
4. P1-1 logger 写入队列和目录初始化优化。
5. P1-2 workspace resolver 缓存和诊断日志。
6. P1-3 feedback 分页、索引和前端分页。
7. P1-4 dev 前端错误展示去除 innerHTML。
8. P1-7 retry 退避、抖动和可观测性。
9. P1-8 静态资源路径校验加固。
10. P1-5 deterministic smoke 拆分。
11. P2-1 DB JSON schema 校验。
12. P2-2 清理 `tsconfig.test.json`。
13. P2-3 低风险依赖升级。
14. P2-4 拆分 composition root。
15. P2-5 增强内部可观测事件。

## 当前验证结果

已运行：

```bash
npm run check
```

结果：

- TypeScript typecheck 通过。
- ESLint 通过。
- Vitest 21 个测试文件、153 个测试通过。
- 生产构建通过。

已运行：

```bash
npm run smoke
npm run smoke:deterministic
```

结果：真实模型 smoke 和 deterministic smoke 均通过。真实 smoke 前已执行 `npm run harness -- --reset` 并启动本地 dev server；验证后已停止临时服务。
