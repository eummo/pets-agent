# pets-agent 系统升级 Loop Engineering 报告

报告日期：2026-06-10  
报告对象：`D:\code\pets-agent`  
报告目的：整合 Loop Engineering 外部实践、pets-agent 当前实现和可执行升级路线，为架构评审与项目排期提供依据。

## 实施状态

| Phase | 状态 | 完成日期 | 备注 |
|-------|------|----------|------|
| Phase 0：安全执行契约和最小状态机 | ✅ 已完成 | 2026-06-10 | 全部工作项已交付，见下方偏差记录 |
| Phase 1：SQLite 运行账本和只读安全 Verifier | ⏳ 待开始 | — | — |
| Phase 2：有限迭代 Loop 与 Cron 触发 | ⏳ 待开始 | — | — |
| Phase 3：Human-in-the-Loop | 🔲 未排期 | — | — |
| Phase 4：代码任务 Worktree 隔离 | 🔲 未排期 | — | — |
| Phase 5：Planner 与 Reviewer | 🔲 未排期 | — | — |
| Phase 6：历史反馈和策略优化 | 🔲 未排期 | — | — |

### Phase 0 实施偏差

Phase 0 按计划完成，以下为与原方案的差异：

1. **`makeDecision` 和 `handleDecision` 签名精简**：Phase 0 中 `makeDecision` 只使用 `actResult` 和 `verificationPassed`，移除了未使用的 `definition`、`run`、`step` 参数（预留 Phase 2+ 扩展点）。`handleDecision` 同理移除了未使用的 `run` 和 `definition` 参数。原方案中这些参数作为"未来扩展"保留，实际因项目 lint 规则（禁止未使用参数）而精简。

2. **`extractTokenUsage` 无参数化**：Phase 0 中 token 用量统计为占位实现（返回 0），不解析 observation。原方案预期从 observation 提取，实际留待 Phase 2 从 `ActionResult` 结构中直接获取。

3. **未修改 `InboundMessage` 或 `AgentRequest`**：原方案提到为 `InboundMessage` / `AgentRequest` 设计 `AbortSignal` 传播。Phase 0 的 `LoopService` 通过 `ActionExecutor` 抽象缝隙独立运行，`AbortSignal` 在 Loop 内部从 trigger 传播到 executor，不触及 `InboundMessage` 或 `AgentRequest`。Phase 2 集成 `MessageGateway` 时再决定是否扩展这些契约。

4. **未接入 `src/index.ts` 组合根**：原方案未明确是否在 Phase 0 接入。实际保持零集成：Loop 模块仅在 smoke 测试中实例化，不影响主服务启动流程。

5. **测试覆盖超出预期**：原方案预期"全部状态转换有确定性单元测试"。实际交付 57 个单元测试 + 6 个确定性冒烟案例 + 3 个真实 LLM 冒烟案例，覆盖面大于原方案。

### Phase 0 交付清单

| 交付物 | 文件 | 状态 |
|--------|------|------|
| 领域类型 + Zod schema | `src/loop/loopTypes.ts` | ✅ |
| 事件辅助 | `src/loop/loopEventLogger.ts` | ✅ |
| InMemoryLoopStore | `src/loop/loopStore.ts` | ✅ |
| LoopService 状态机 | `src/loop/loopService.ts` | ✅ |
| barrel exports | `src/loop/index.ts` | ✅ |
| Store 单元测试（35 个） | `src/loop/loopStore.test.ts` | ✅ |
| 状态机单元测试（13 个） | `src/loop/loopService.test.ts` | ✅ |
| 恢复与幂等单元测试（9 个） | `src/loop/loopRecovery.test.ts` | ✅ |
| 确定性冒烟案例（6 个） | `src/smoke/deterministicSmoke.ts` | ✅ |
| 真实 LLM 冒烟（3 个） | `src/smoke/loopSmoke.ts` | ✅ |
| `loop_manage` 能力 | `src/auth/index.ts` | ✅ |
| 架构文档更新 | `docs/architecture.md` | ✅ |
| 升级报告状态更新 | 本文件 | ✅ |

## 1. 执行摘要

pets-agent 当前是一个架构边界清晰的知识库 Agent 网关：通道输入统一进入 `MessageGateway`，经过工作区解析、角色与权限判断、意图分类后，调用可替换的 `AgentRuntime`，并把会话、历史、反馈和运行日志持久化。

系统已经具备建设 Loop Engineering 的大部分基础设施：

- provider-neutral 的网关和 runtime 契约；
- SQLite 会话、历史、反馈和 cron 状态；
- cron 自动触发；
- 工具权限与工作区路径约束；
- conversation、system、llm-raw 三类 JSONL 可观测日志；
- unit、deterministic smoke 和真实模型 smoke 验证链路。

但当前执行模型仍以“一条消息触发一次 runtime”为主。cron 也只是定时重复一次消息处理，缺少任务级目标、步骤状态、验证结果、停止条件、人工审批和恢复能力。因此，当前系统可定义为：

> 单轮消息网关 + 定时任务能力，而不是目标驱动、可验证、可恢复的持续执行系统。

推荐在现有网关之上增加独立的 Loop Control Plane，以 `plan -> act -> observe -> verify -> decide` 为主循环。Loop 层继续通过 `MessageGateway` 或新的 provider-neutral 执行契约调用现有能力，不把 Claude、CodeBuddy、Pi、企业微信或 GitHub SDK 引入核心编排。

建议优先级：

1. P0：定义执行身份、取消传播、幂等恢复和执行上下文契约。
2. P0：持久化 `LoopRun`、`LoopStep` 和明确停止条件。
3. P0：接入受控的确定性 verifier，将现有 check、smoke 和日志断言纳入循环。
4. P1：增加暂停、人工审批和恢复。
5. P1：在开放文件修改前增加 git worktree 隔离。
6. P2：在有明确收益后再引入 planner、reviewer 等多 Agent 协作。

## 2. Loop Engineering 的工程含义

“Loop Engineering”目前不是统一标准或固定框架，更适合作为一组新兴工程实践的总称：人不再逐轮提示 Agent，而是设计一个能持续发现任务、提供上下文、执行动作、观察结果、验证目标并决定下一步的系统。

其核心不是“无限重试”，而是一个有边界、有状态、有反馈的控制循环：

```text
Goal
  -> Plan or select next action
  -> Act through an agent/tool
  -> Observe outputs and environment state
  -> Verify against acceptance criteria
  -> Complete / retry with changed strategy / pause / escalate
```

### 2.1 核心原则

1. **目标优先**：在执行前定义可观察、可验证的完成条件。
2. **环境反馈**：以测试、构建、日志、文件状态、API 返回等 ground truth 判断结果。
3. **外部状态**：任务状态必须持久化，不能只依赖模型上下文或聊天记录。
4. **明确终止**：成功、失败、超时、预算耗尽、停滞和等待人工都必须是显式状态。
5. **权限受控**：每一次工具调用仍受角色、工作区和动作权限约束。
6. **渐进复杂度**：先使用确定性流程和单 Agent，只有任务确实需要时才增加动态规划或多 Agent。
7. **全程可观测**：能够按 run、step、message、workspace 和时间线重建整个过程。

### 2.2 与相关 Agent 实践的关系

- Anthropic 的 agent 构建建议区分预定义 workflow 与由模型动态决定过程的 agent，并强调从简单、可组合模式开始。
- ReAct 模式把推理、动作和环境观察交错执行，是 loop 的基础行为模型之一。
- LangGraph 等框架把 thread、checkpoint、interrupt 和 resume 作为 durable execution 的关键能力。
- OpenAI Agents SDK 将 agent loop、guardrails、sessions、human-in-the-loop 和 tracing 作为运行时基础能力。

这些资料共同指向同一个结论：Loop Engineering 的价值来自控制系统，而不只是更长的 prompt 或更多模型调用。

## 3. pets-agent 当前系统基线

### 3.1 当前主链路

根据 `docs/architecture.md`，目标链路为：

```text
User channel
  -> channel adapter
  -> MessageGateway
  -> workspace + authorization + intent gate
  -> AgentRuntime adapter
  -> selected workspace / knowledge base
  -> persistence and JSONL logs
```

`AgentOrchestrator.handle()` 当前负责：

1. 处理 `/help`、`/new` 等命令；
2. 解析选中的 workspace；
3. 解析角色、分类意图并检查权限；
4. 选择角色对应的 runtime；
5. 执行一次 runtime 调用；
6. 保存 session、history、context usage 和最终响应；
7. 记录 workspace、role、intent、permission、runtime 等系统事件。

这条链路适合用户发起的知识问答和一次性开发任务，应继续保留为系统的快速路径。

### 3.2 已有 Loop 基础

| 能力         | 当前实现                                | 对 Loop 的价值                             |
| ------------ | --------------------------------------- | ------------------------------------------ |
| 统一执行入口 | `MessageGateway`                        | Loop 的 act 阶段可以复用现有授权和执行链路 |
| Runtime 抽象 | `AgentRuntime` / `AgentRuntimeFactory`  | Loop 不绑定 Claude、CodeBuddy 或 Pi        |
| 工作区解析   | `KnowledgeWorkspaceResolver`            | Run 启动时固定目标 workspace               |
| 权限控制     | `RequestAuthorizationGate`、tool policy | 每轮动作仍受角色和路径限制                 |
| 会话状态     | session/history store                   | 支持同一 run 内的上下文连续性              |
| 持久化       | SQLite stores                           | 可扩展 run、step、review 和 checkpoint 表  |
| 自动触发     | `TickCronScheduler`                     | 可作为 loop trigger，而不是 loop 本身      |
| 可观测性     | 三类 JSONL 日志                         | 支持复盘分类、模型调用、工具和最终结果     |
| 验证工具链   | `npm run check`、`npm run smoke`        | 可直接转化为 deterministic verifier        |

### 3.3 当前约束

- `Orchestrator` 一次只产生一个 `AgentResponse`，没有任务级状态机。
- `CronJobResult` 只记录单次 success、error、timeout 或 skipped，没有步骤和验证证据。
- conversation history 表达聊天上下文，不等价于任务运行账本。
- feedback store 主要记录被权限拒绝但有价值的请求，不承担审批恢复。
- 代码修改直接使用选定 workspace，多任务并发时缺少独立 worktree。
- 现有 smoke 是交付验证入口，但还不是运行中 loop 的 verifier。

## 4. 能力差距

| 维度          | 当前状态                                 | 目标状态                                             |
| ------------- | ---------------------------------------- | ---------------------------------------------------- |
| Goal          | 主要存在于用户 prompt 或 cron prompt     | 结构化目标和验收条件                                 |
| Run state     | conversation/session 和 cron last result | 独立 LoopRun、LoopStep、checkpoint                   |
| Verify        | runtime 自行判断或人工运行测试           | 确定性、日志型和模型型 verifier                      |
| Termination   | runtime 返回即结束                       | complete、retry、paused、blocked、failed、cancelled  |
| Retry         | 没有系统级策略                           | 有次数、预算、退避和停滞检测                         |
| Recovery      | 失败后重新发消息                         | 从持久化 checkpoint 恢复                             |
| Human review  | 权限拒绝和 feedback                      | pause、approve、reject、resume                       |
| Isolation     | 共享 workspace                           | 代码任务按 run 分配 worktree                         |
| Observability | 以 messageId 为主                        | 增加 loopRunId、stepId、attempt 和 verifier evidence |
| Cost control  | 单次 runtime 上下文指标                  | run 级 token、时间、调用次数和预算                   |
| Cancellation  | cron 只能停止等待，不能终止底层 runtime  | `AbortSignal` 从 trigger 传播到 runtime 和工具       |
| Identity      | cron 使用合成用户和可选 role override    | 明确请求人、执行主体、授权版本和审批人               |
| Idempotency   | 无任务步骤幂等契约                       | step lease、幂等键、interrupted 恢复和副作用核验     |

最重要的缺口不是“模型不够聪明”，而是系统无法可靠回答以下问题：

- 这个任务的完成条件是什么？
- 当前执行到哪一步？
- 上一步改变了什么？
- 验证为什么通过或失败？
- 下一次应该改变策略还是停止？
- 服务重启后如何继续？
- 哪些动作必须等待人工审批？

## 5. 目标架构

推荐新增 `src/loop` 领域模块，并保持依赖向内：

```text
Channel / Cron / API trigger
          |
          v
      LoopService
          |
          +--> LoopStore
          +--> LoopVerifier
          +--> LoopReviewStore
          +--> WorkspaceIsolationService
          |
          v
     MessageGateway or provider-neutral action executor
          |
          v
 Existing authorization -> AgentRuntime -> workspace
```

### 5.1 建议契约

```typescript
type LoopRunStatus =
  | "queued"
  | "running"
  | "paused"
  | "completed"
  | "blocked"
  | "failed"
  | "cancelled";

type LoopDecision =
  | { readonly kind: "complete"; readonly reason: string }
  | { readonly kind: "continue"; readonly nextAction: string }
  | { readonly kind: "pause"; readonly reason: string }
  | { readonly kind: "fail"; readonly reason: string };

type LoopExecutionContext = {
  readonly loopRunId: string;
  readonly stepId: string;
  readonly attempt: number;
  readonly idempotencyKey: string;
  readonly requestedBy: string;
  readonly executionPrincipal: string;
  readonly authorizedPolicyVersion: string;
};
```

核心对象建议包括：

- `LoopDefinition`：目标、workspace、role、触发方式、预算、验证策略和风险等级。
- `LoopRun`：一次运行实例及其当前状态、累计成本和 checkpoint。
- `LoopStep`：一次 plan、act、observe、verify、review 或 decision。
- `LoopExecutionContext`：贯穿 gateway、runtime、tool、verifier 和日志的关联与幂等信息。
- `LoopStore`：持久化 definition、run、step 和 checkpoint。
- `LoopVerifier`：输入目标、观察和证据，输出结构化验证结果。
- `LoopService`：推进状态机，不包含 provider SDK 细节。
- `LoopReviewStore`：保存审批请求、结论、审批人和恢复信息。
- `WorkspaceIsolationService`：provider-neutral 契约；git worktree 实现在基础设施适配器中。

### 5.2 执行身份与授权

Loop 不应仅以一个可选 role 运行。每个 definition 和 run 必须记录：

- `requestedBy`：创建或触发任务的真实用户或系统主体；
- `executionPrincipal`：实际执行动作的服务主体；
- `requestedRole`：请求使用的角色；
- `authorizedPolicyVersion`：创建或恢复时采用的权限配置版本；
- `approvedBy`：存在人工审批时的审批主体；
- `triggerType`：manual、cron、API 或 future event。

新增、启停、取消、恢复和审批 loop 应使用独立的 `loop_manage` 能力。`roleOverride` 不能单独作为持续执行授权依据。每次 act、resume 和高风险动作前都要重新检查当前权限；权限收紧后，已有 run 应暂停而不是沿用旧权限继续执行。

### 5.3 取消、超时与幂等恢复

`AbortSignal` 必须从 trigger 传播到 `LoopService`、单次 action executor、`AgentRequest` 和支持取消的工具适配器。超时或取消只有在底层执行确认停止后，才能把 step 标记为终止；不能只停止等待 Promise。

持久化恢复采用至少一次执行语义时，每个 step 必须具有：

- 稳定的 `idempotencyKey`；
- claim owner 和 lease expiry；
- `queued -> running -> succeeded/failed/interrupted` 状态；
- act 完成后先保存 observation/evidence，再提交 step 完成状态；
- 服务重启后把过期的 running step 转为 interrupted；
- interrupted step 先重新观察外部状态，再决定完成、补偿或重试，不能直接重放副作用。

文件修改、通知、发布和外部 API 调用应分别定义幂等或补偿策略。无法幂等且无法可靠观察的动作必须进入人工审批。

### 5.4 与现有 Orchestrator 的边界

不建议把多轮 loop 直接塞进 `AgentOrchestrator.handle()`。原因是：

- 普通聊天不应承担长任务的延迟和失败语义；
- 网关负责单次授权执行，LoopService 负责跨步骤生命周期；
- cron、人工请求、API 或未来事件都可以成为 loop trigger；
- LoopService 可在每个 act 前重新经过权限和工作区检查。

短期内，LoopService 可以通过构造 `InboundMessage` 调用 `MessageGateway`，但必须同时扩展 provider-neutral 执行上下文和取消契约。进入文件修改、恢复或多步骤执行前，应提取结构化 `AgentActionExecutor`，避免依赖自然语言输出判断 step 状态。

## 6. 验证与停止策略

### 6.1 Verifier 分层

按可靠性由高到低组合：

1. **确定性验证**：TypeScript、lint、unit test、build、deterministic smoke、文件或 schema 检查。
2. **运行时验证**：HTTP health/readiness、真实 smoke、业务接口返回和 delivery 状态。
3. **日志验证**：按 `loopRunId`、`messageId`、workspace 和事件类型检查行为链路。
4. **模型验证**：适用于文档质量、回答相关性等无法完全结构化的目标。
5. **人工验证**：外部发布、重要文件变更、低置信度结论和高风险操作。

模型 verifier 不应覆盖确定性失败。测试未通过时，无论模型如何评价，都不能判定代码目标已完成。

“确定性”只表示结果可重复判断，不表示执行天然安全。命令型 verifier 必须满足：

- 只能在显式信任的 workspace 或隔离 worktree 中执行；
- 只能引用管理员配置的 verifier 模板，不能接受用户提供的任意 shell 命令；
- 使用环境变量白名单，不继承不必要的 secrets；
- 限制运行时间、输出大小、CPU/内存和并发数；
- 记录命令模板版本、工作目录、退出码和经过截断/脱敏的证据；
- verifier 权限与 developer runtime 权限分离。

### 6.2 必须存在的边界

每个 definition 至少配置：

- `maxIterations`；
- `timeoutMs`；
- 模型调用或 token 预算；
- 允许的 role 和工具集合；
- success condition；
- retryable failure；
- stagnation threshold；
- 需要人工审批的动作；
- 失败和超时的通知目标。

连续获得相同失败证据时，系统必须改变策略、暂停或升级，不能原样重复 prompt。

## 7. 分阶段升级路线

### Phase 0：安全执行契约和最小状态机 ✅ 已完成（2026-06-10）

目标：建立 provider-neutral Loop 领域模型，并先解决身份、取消、幂等和日志关联，不改变现有聊天和 cron 行为。

工作项：

- 新增 `src/loop` 公共契约；
- 实现内存版 `LoopStore` 和 deterministic `LoopService`；
- 定义 `LoopExecutionContext`，贯穿 gateway、runtime、tool 和日志；
- 为 `InboundMessage` / action executor / `AgentRequest` 设计可选 `AbortSignal` 传播；
- 定义 requestedBy、executionPrincipal、policy version 和 `loop_manage` 授权规则；
- 定义 step lease、idempotency key、interrupted 和恢复决策；
- 覆盖完成、继续、暂停、超限、取消、租约过期和恢复测试；
- 定义 `loop.started`、`loop.step.*`、`loop.verified`、`loop.completed` 等日志事件。

验收：全部状态转换有确定性单元测试；取消可到达 mock runtime；过期 running step 不会被直接重放；现有行为无回归。

**验收结果：✅ 全部通过。** 57 个单元测试 + 6 个确定性冒烟案例 + 3 个真实 LLM 冒烟案例。`npm run check` 零回归。偏差见上方"Phase 0 实施偏差"。

### Phase 1：SQLite 运行账本和只读安全 Verifier

目标：形成可恢复、可审计的最小闭环。

工作项：

- 增加 loop definitions、runs、steps、step leases、reviews 表；
- 先支持文件、schema、日志和只读 HTTP 断言；
- 命令型 verifier 仅允许管理员配置的模板和可信 workspace；
- 服务重启后恢复 queued、paused，并将过期 running 转为 interrupted 后重新观察；
- JSONL 事件增加 `loopRunId`、`stepId`、`attempt`。

验收：执行中重启服务后能够安全继续；不会重复提交已完成副作用；失败原因和验证证据可完整重建。

### Phase 2：有限迭代 Loop 与 Cron 触发

目标：支持最多 2-3 轮的受控 act-verify-decide，并让 cron 只负责触发。

工作项：

- 保留现有 message job，新增 `mode: "message" | "loop"` 或 `loopDefinitionId`；
- 复用当前 scheduler、leader lease 和 delivery；
- 同一 definition 默认只允许一个活动 run，使用唯一约束或原子 claim 防重复；
- `/cron/status` 或新的 loop API 返回 run 状态；
- 保持无 loop 配置的旧 job 完全兼容。

验收：定时知识库巡检和有限诊断 loop 可自动执行、验证、改变下一轮策略、结束或升级，不重复创建同一活动 run。

### Phase 3：Human-in-the-Loop

目标：高风险动作可以暂停并由人恢复。

工作项：

- 增加 approve、reject、resume、cancel 操作；
- 审批记录包含 action、evidence、diff 摘要、风险和过期时间；
- 企业微信或浏览器通道只负责展示和提交决定，不拥有审批规则；
- 恢复时重新检查角色、权限和 workspace 状态。

验收：审批前不会执行受控动作；审批后从正确 checkpoint 恢复且有完整审计。

### Phase 4：代码任务 Worktree 隔离与文件修改开放

目标：多个开发 loop 不直接并发修改同一工作目录；文件修改型 loop 只有在本阶段完成后才开放。

工作项：

- 定义 `WorkspaceIsolationService` 契约；
- 在 infrastructure/adapter 层实现 git worktree；
- run 记录 branch、worktree path、base revision 和清理状态；
- verifier 默认在同一 run 的 worktree 中执行；
- 支持保留供人工检查、发布后清理和异常回收。

验收：两个并发 run 可以修改同一基础仓库而不互相覆盖，清理过程可恢复。

### Phase 5：Planner 与 Reviewer

目标：仅为真正复杂的任务增加多角色协作。

工作项：

- planner 输出结构化步骤和每步验收条件；
- executor 只执行当前获批步骤；
- reviewer 检查证据和目标，不替代确定性 verifier；
- 根据风险和任务复杂度决定是否启用，不作为所有请求默认路径。

验收：与单 Agent 基线相比，复杂任务成功率或人工返工量有可测改善，否则不扩大使用。

### Phase 6：历史反馈和策略优化

目标：利用 run 数据减少重复失败，而不是让模型无限读取所有历史。

工作项：

- 聚合同类目标的失败类型、验证结果和人工结论；
- 只向新 run 提供经过筛选的相关经验；
- 监测首轮成功率、平均迭代数、停滞率、人工介入率和单位成功成本；
- 对策略和 prompt 进行版本化，支持 A/B 或回滚。

## 8. 首批推荐场景

### 8.1 知识库健康巡检

定时检查 workspace 中架构文档、README、配置和实际代码是否一致。默认只读，输出带证据的差异报告。该场景风险低，适合作为第一个生产 loop。

### 8.2 客户问题日志复盘

从最新 `conversation.turn` 出发，使用相同 `messageId`、`userId`、`workspacePath` 和相邻时间戳关联 system、llm-raw 日志，生成分类、授权、runtime、工具调用和最终响应链路。该场景主要读取日志，验证标准清楚。

### 8.3 回归失败定位

接收 check、smoke 或 CI 失败，收集错误和相关日志，形成假设并运行有限的诊断步骤。初期只生成建议；稳定后再允许在 worktree 中修复。

该场景作为首个真正的迭代试点，限制为最多三轮：

1. 读取失败证据并形成一个可验证假设；
2. 执行一组白名单诊断并验证假设；
3. 若证据未改善，必须改变假设；连续两轮证据相同则暂停并升级人工。

### 8.4 受控代码修复

在独立 worktree 中执行修改，运行 `npm run check` 和必要的 `npm run smoke`，通过后暂停等待人工审查或交给 `ChangePublisher`。该场景价值高，但应在审批与隔离完成后上线。

## 9. 与现有路线图的关系

现有 `docs/cron-production-plan.md` 关注跨主机多副本下的调度、leader election 和队列化；Loop Engineering 关注单个任务从目标到验证的生命周期。两者互补，不能互相替代。

建议顺序：

1. 单实例内先完成 LoopRun 持久化和 verifier，证明业务价值。
2. 保持当前 cron leader lease，避免过早引入 Redis/Postgres。
3. 当需要跨主机运行或长任务队列时，按 cron 生产化方案迁移共享状态和执行队列。
4. LoopStore、CronJobStore 和 worker 通过契约替换后端，不改变 gateway 和 runtime 适配器。

## 10. 风险与控制

| 风险                  | 控制措施                                                       |
| --------------------- | -------------------------------------------------------------- |
| 无限重试和成本失控    | iteration、time、token、call budget；停滞检测；强制终止        |
| 模型自评导致假通过    | 确定性 verifier 优先；模型评估保留证据和置信度                 |
| 取消后后台继续执行    | `AbortSignal` 端到端传播；底层停止确认；取消状态审计           |
| 恢复导致重复副作用    | step lease、幂等键、interrupted 状态和恢复前外部状态核验       |
| 合成角色越权          | requestedBy/executionPrincipal 分离；`loop_manage`；恢复时重验 |
| verifier 执行恶意脚本 | 可信 workspace、命令模板、环境白名单和资源限制                 |
| 多任务修改冲突        | worktree 隔离、run lease、清理和恢复机制                       |
| 权限绕过              | 每个 act 继续经过 authorization 和 tool policy                 |
| 上下文持续膨胀        | run ledger 与 conversation 分离；step summary；只加载相关证据  |
| 服务重启丢状态        | SQLite checkpoint；幂等 step；恢复时重验外部状态               |
| 日志泄露秘密          | 延续现有脱敏规则，不记录 key、token、authorization header      |
| 过早引入多 Agent      | 以单 Agent + deterministic workflow 为基线，用指标证明增益     |
| WeCom 响应窗口不足    | 长 loop 异步运行，通道只返回已受理和进度/最终通知              |

## 11. 指标与验收

### 11.1 运行指标

- goal completion rate；
- first-iteration completion rate；
- 平均和 P95 迭代数、耗时、模型调用数；
- deterministic verifier pass rate；
- stagnation、blocked、timeout 和 cancellation rate；
- human review rate 和 review turnaround time；
- 单个成功 run 的 token 或成本；
- worktree 泄漏和恢复失败数。
- cancellation propagation failure；
- interrupted step replay prevention count；
- authorization recheck denial rate。

### 11.2 工程验收

每个阶段必须满足：

```bash
npm run check
npm run smoke
```

同时检查：

- `conversation.jsonl` 中输入和输出正确；
- `system.jsonl` 中 run、step、权限、runtime、验证和结束事件可串联；
- `llm-raw.jsonl` 中模型与工具事件有对应 run/step 标识；
- denied 或 paused 请求不会产生越权工具结果；
- cancelled step 的 runtime 和工具执行能够终止，且不会晚到提交成功状态；
- interrupted step 恢复前重新观察外部状态，不盲目重复副作用；
- 日志不包含 API key、access token 或 authorization header；
- bug 修复有 deterministic unit test，运行时行为变化有 smoke regression。

## 12. 建议决策

### 已完成

Phase 0 已于 2026-06-10 交付并通过全部验收。`src/loop` 模块已在生产质量门禁中覆盖（确定性冒烟纳入 `npm run check`，真实 LLM 冒烟通过 `npm run smoke:loop`）。

### 当前建议

推进 Phase 1-2 作为下一个可独立交付的最小版本：

- Phase 1：SQLite run ledger + 只读安全 verifier；
- Phase 2：有限迭代 loop + cron trigger。

完成后用”知识库健康巡检”验证运行账本，用”回归失败定位”验证真正的有限循环，再决定是否投资 Human-in-the-Loop、worktree 和多 Agent。文件修改型 loop 必须等待 Phase 3 审批和 Phase 4 worktree 隔离完成后开放。

## 13. 参考资料

- Addy Osmani, _Loop Engineering_, 2026-06-07: https://addyosmani.com/blog/loop-engineering/
- Anthropic, _Building Effective AI Agents_: https://www.anthropic.com/engineering/building-effective-agents
- Shunyu Yao et al., _ReAct: Synergizing Reasoning and Acting in Language Models_: https://arxiv.org/abs/2210.03629
- LangGraph, _Persistence_: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph, _Interrupts / Human-in-the-loop_: https://docs.langchain.com/oss/python/langgraph/interrupts
- OpenAI Agents SDK documentation: https://openai.github.io/openai-agents-python/
- 项目内部：`docs/architecture.md`
- 项目内部：`docs/development-workflow.md`
- 项目内部：`docs/cron-production-plan.md`
- 项目内部：`docs/persistence-lifecycle.md`
