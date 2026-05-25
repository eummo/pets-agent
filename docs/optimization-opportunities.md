# 当前项目优化项清单

生成日期：2026-05-25

本清单基于当前仓库静态扫描、关键路径阅读和 `npm run check` 结果整理。项目当前类型检查、lint、单元测试和构建均通过；以下内容不是已确认线上故障，而是按稳定性、性能、可维护性和扩展性排序的后续优化机会。

## 摘要

| 优先级 | 数量 | 主要方向 |
| --- | ---: | --- |
| P0 | 3 | 并发写入一致性、权限分类失败策略、运行时配置生效 |
| P1 | 5 | 日志吞吐、工作区解析缓存、SQLite 查询边界、前端安全和 smoke 稳定性 |
| P2 | 4 | 类型校验、配置清理、代码组织和可观测性增强 |

建议优先处理前三项：

1. 为 session/history 文件存储增加串行写队列或迁移到 SQLite，避免并发消息丢历史。
2. 为意图分类失败增加确定性保守兜底，避免分类服务超时后把明显修改请求当成普通查询。
3. 让角色配置变更能使 runtime 缓存失效，避免后台更新角色配置后仍使用旧工具和模型权限。

## P0 优化项

### 1. 文件型 session/history store 存在并发丢写风险

证据位置：

- `src/core/fileConversationSessionStore.ts:29` 先读整个 JSON，再在 `src/core/fileConversationSessionStore.ts:66` rename 覆盖。
- `src/core/fileConversationHistoryStore.ts:47` 同样先读整个 JSON，再在 `src/core/fileConversationHistoryStore.ts:108` 覆盖。

风险：

当同一进程内多个用户或同一用户多请求并发完成时，两个请求可能都基于旧文件内容修改，后完成的写入覆盖先完成的写入。现在的临时文件加 rename 能保证单次写原子性，但不能保证 read-modify-write 的事务性。

建议：

- 短期：在两个 file store 内实现按文件路径串行化的 promise queue，确保 `set/delete/append/archive` 串行执行。
- 中期：把 session/history 也迁移到 SQLite，和 feedback/roles 使用同一类事务边界。
- 为并发 append/set 增加 deterministic 单元测试，例如并发写 20 个不同 session key，断言最终都保留。

验证：

- 新增并发单元测试。
- 运行 `npm run check`。
- 如迁移存储，运行 `npm run harness -- --reset` 后再运行 `npm run smoke`。

### 2. 意图分类失败默认放行为 query，权限策略偏乐观

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

### 3. Runtime 缓存只按 role 保存，角色配置更新后不会自动生效

证据位置：

- `src/core/orchestrator.ts:37` 和 `src/core/orchestrator.ts:40` 初始化 `runtimeCache`。
- `src/core/orchestrator.ts:86` 到 `src/core/orchestrator.ts:90` 只有缺失 runtime 时才创建。
- `src/index.ts:152` 到 `src/index.ts:173` 启动时把 DB 中的 role config 转成 runtime。

风险：

如果角色配置、模型、工具列表或权限模式在数据库中被更新，已有进程仍可能继续使用旧 runtime，直到服务重启。后续增加管理页面或 API 修改角色配置时，这会成为明显的行为不一致。

建议：

- 给 `StoredRoleConfig` 增加 `updatedAt` 或版本号，并把 runtime cache key 扩展为 `role + version`。
- 或在 role config 更新路径中发布 invalidation 事件，让 orchestrator 清理对应 role runtime。
- 增加测试：修改 role allowedTools 后，同一进程下一次请求应使用新配置。

## P1 优化项

### 4. JSONL logger 每条日志都 mkdir + writeFile append，缺少写入队列

证据位置：

- `src/logging/jsonlLogger.ts:15` 每次写日志都 `mkdir`。
- `src/logging/jsonlLogger.ts:16` 每次写日志都 `writeFile(..., { flag: "a" })`。

影响：

吞吐较低时问题不大，但 smoke、流式事件或多用户并发时会重复做目录检查和打开文件。并发 append 虽通常可用，但没有统一 backpressure、flush 或 close 语义。

建议：

- logger 创建时确保目录存在一次。
- 使用内部 promise queue 串行化 append，或使用 `createWriteStream` 并提供 `flush/close`。
- 继续保留当前密钥脱敏测试，并新增并发写入行数测试。

### 5. 工作区解析每次请求都读 repos.json，且解析失败静默回退

证据位置：

- `src/repos/staticWorkspaceResolver.ts:53` 每次匹配前调用 `loadRepositories()`。
- `src/repos/staticWorkspaceResolver.ts:65` 每次从磁盘读取配置。
- `src/repos/staticWorkspaceResolver.ts:67` catch 后直接返回空数组。

影响：

每次消息都有额外 I/O。更重要的是，配置文件 JSON 损坏或 schema 不匹配时会静默回退到知识库，导致问题难以排查。

建议：

- 加入基于 mtime 的轻量缓存，配置变化时自动刷新。
- 配置解析失败时写入结构化日志，并在开发环境暴露诊断信息。
- 测试覆盖：合法配置缓存、配置变更刷新、非法配置可观测。

### 6. SQLite store 缺少分页和部分索引，后台数据增长后会拖慢管理接口

证据位置：

- `src/db/sqliteFeedbackStore.ts:74` 的 `getAll()` 返回所有 feedback。
- `src/db/sqliteFeedbackStore.ts:90` 仅按 id 倒序排序。
- `src/db/sqliteConnection.ts:16` 的 feedback 表没有按 status/user/workspace 的索引。

影响：

本地 harness 数据量小没有问题，但如果 feedback 成为长期管理入口，全量返回会让 `/dev/feedback` 响应和前端渲染越来越慢。

建议：

- `FeedbackStore.getAll` 改成支持 `limit/offset/status`。
- 增加索引：`feedback(status, id)`、`feedback(user_id, id)`，如后续常按 workspace 查，再加 `workspace_path`。
- 前端反馈页做分页或“加载更多”。

### 7. Dev 前端仍有少量 innerHTML 拼接错误文本

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

### 8. Smoke 依赖真实服务和模型，缺少“可控模型失败”场景

证据位置：

- `src/smoke/regressionSmoke.ts:48` 之后直接跑真实 `/dev/chat`。
- `src/smoke/regressionSmoke.ts:249` 到 `src/smoke/regressionSmoke.ts:298` 对 pi-ai 做真实模型调用。

影响：

当前 smoke 能验证真实集成，这是优点；但它也会受模型可用性、额度、网络和提示词波动影响。对于权限拒绝、日志、安全这些确定性行为，应该有不依赖模型的 smoke 子集。

建议：

- 拆出 `npm run smoke:deterministic`，使用 echo/faux provider 或可控 stub。
- 保留 `npm run smoke` 作为真实模型回归。
- 把“模型不可用时权限仍保守”的 case 放到 deterministic smoke 中。

## P2 优化项

### 9. 数据库 JSON 字段读取只做类型断言，缺少运行时校验

证据位置：

- `src/db/sqliteRoleConfigStore.ts:15` 解析 capabilities 后直接断言类型。
- `src/db/sqliteRoleConfigStore.ts:19` 解析 allowedTools 后直接断言 `string[]`。

建议：

- 复用或新增 zod schema 校验 DB 中的 role config。
- 对损坏配置返回可诊断错误，避免把非法 permissionMode 或 tools 传入 runtime。

### 10. `tsconfig.test.json` 指向不存在的旧测试路径

证据位置：

- `tsconfig.test.json:8` include 了 `src/tasks/agent-manager.test.ts` 和 `src/memory/injector.test.ts`，当前仓库没有这些路径。

影响：

当前 `vitest.config.ts` 使用 `src/**/*.test.ts`，所以测试实际不受影响；但旧配置会误导后续维护者，也可能在 IDE 或脚本中造成混乱。

建议：

- 删除无用的 `tsconfig.test.json`，或改成当前测试范围。
- 如果保留，明确它的用途，并让 `npm run typecheck` 或 Vitest 配置实际引用它。

### 11. Composition root 偏长，配置加载和 runtime 构建可以拆成小模块

证据位置：

- `src/index.ts` 同时负责环境读取、SDK 环境变量配置、SQLite 初始化、角色 seed、runtime 构建、server 创建和启动。

建议：

- 拆出 `src/config/runtimeConfig.ts` 负责环境变量解析和默认值。
- 拆出 `src/agent/createAgentRuntimes.ts` 负责 runtime factory 和缓存策略。
- 保持 `src/index.ts` 作为 composition root，但让主流程更短、更容易测试。

### 12. 可观测性可以补充分类、权限和 workspace 选择事件

证据位置：

- 当前 `src/core/orchestrator.ts:217` 的 conversation log 主要记录输入输出。
- `src/agent/claudeSdkAgentRuntime.ts:168` 记录 runtime response。

建议：

- 增加内部事件：workspace resolved、role resolved、intent classified、permission denied、runtime selected。
- 继续沿用 `JsonlLogger` 的脱敏逻辑，不记录 authorization header、token、API key。
- smoke 中验证关键内部事件存在，尤其是被拒绝请求没有进入 runtime。

## 建议实施顺序

1. P0-1 文件存储并发队列或 SQLite 迁移。
2. P0-2 意图分类失败的确定性保守兜底。
3. P0-3 runtime cache invalidation。
4. P1-4 logger 写入队列和目录初始化优化。
5. P1-5 workspace resolver 缓存和诊断日志。
6. P1-6 feedback 分页、索引和前端分页。
7. P1-7 dev 前端错误展示去除 innerHTML。
8. P1-8 deterministic smoke 拆分。
9. P2-9 DB JSON schema 校验。
10. P2-10 清理 `tsconfig.test.json`。
11. P2-11 拆分 composition root。
12. P2-12 增强内部可观测事件。

## 当前验证结果

已运行：

```bash
npm run check
```

结果：

- TypeScript typecheck 通过。
- ESLint 通过。
- Vitest 20 个测试文件、132 个测试通过。
- 生产构建通过。

未运行：

```bash
npm run smoke
```

原因：`npm run smoke` 需要本地服务先运行，并会触发真实模型回归；本次只新增文档，没有启动服务或调用模型。后续实施 P0/P1 中涉及运行时行为的优化时，应按 `docs/development-workflow.md` 先启动 harness/dev server，再运行 smoke。
