# Pets-Agent 项目组：多角色自主开发团队

## 1. 概念与愿景

一个能够自主完成产品开发生命周期的多智能体团队。每个 agent 扮演固定角色（产品、设计、开发、测试、业务），通过结构化 SOP 在 `pets-agent` 框架上协作，实现从想法到可交付产品的完整流程。

核心理念：**SOP 驱动的多角色协作** — 每个角色有明确的输入/输出规范、决策边界、升级路径，而非通用 LLM。

## 2. 设计原则

### 2.1 角色分工

| 角色 | 职责 | 输出物 | 决策边界 |
|------|------|--------|----------|
| **ProjectManager (PM)** | 统筹协调、阶段推进、决策、风险管理 | 项目计划、阶段报告、决策记录 | 优先级排序、范围调整 |
| **ProductManager** | 需求分析、PRD 编写、用户故事、验收标准 | PRD、用户故事地图、PRD评审 | 功能范围、优先级 |
| **Designer** | UX研究、交互设计、信息架构、视觉规范 | 设计稿、组件规范、UX文档 | 视觉风格、布局方案 |
| **Developer** | 架构设计、编码实现、集成、文档 | 代码、API设计、技术方案 | 技术选型、代码结构 |
| **QATester** | 测试策略、用例设计、缺陷管理、回归测试 | 测试计划、测试报告、缺陷跟踪 | 测试优先级、覆盖率 |
| **BusinessAnalyst** | 市场分析、竞品研究、ROI评估、风险评估 | 市场报告、竞品分析、风险矩阵 | 商业模式、可行性判断 |

### 2.2 SOP（标准操作流程）

每个阶段遵循：`准备 → 执行 → 评审 → 决策 → 升级/通过`

### 2.3 协作模式

**层次化 + 事件驱动：**
```
User/PM发起
    ↓
ProjectManager 分解任务
    ↓
并行 spawn_agent (各角色专家)
    ↓
各角色按 SOP 执行
    ↓
结果汇聚 → PM 评审 → 决策门
    ↓
通过 → 下一阶段 | 不通过 → 迭代
```

### 2.4 记忆系统

利用已实现的 pets-agent 记忆系统：
- **PatternMemory**: 各角色成功工作流模式
- **PreferenceMemory**: 各 agent 类型在各任务类型上的成功率
- **ProjectMemory**: 按项目积累的技术栈、约定、上下文

## 3. 阶段流程

### 阶段 0：想法准入 (Idea Intake)
**输入**: 用户想法 / 原始需求
**输出**: 想法登记、初步可行性评估
**角色**: BusinessAnalyst + PM

### 阶段 1：可行性分析 (Feasibility Analysis)
**输入**: 想法描述
**输出**: 可行性报告（技术、财务、市场、风险）
**角色**: BusinessAnalyst（主导）、Developer（技术评审）

### 阶段 2：需求定义 (Requirements)
**输入**: 可行性通过
**输出**: PRD文档、用户故事地图、验收标准
**角色**: ProductManager（主导）、Designer、BusinessAnalyst

### 阶段 3：设计 (Design)
**输入**: PRD
**输出**: 架构设计、技术方案、UX设计
**角色**: Designer（UX）、Developer（架构）

### 阶段 4：实现 (Implementation)
**输入**: 设计稿
**输出**: 代码、单元测试、集成测试
**角色**: Developer（主导）、QATester（并行）

### 阶段 5：测试 (Testing)
**输入**: 实现代码
**输出**: 测试报告、缺陷列表、回归测试结果
**角色**: QATester（主导）、Developer（修复）

### 阶段 6：评估与交付 (Evaluation & Delivery)
**输入**: 测试报告
**输出**: 评估报告、上线清单、回顾记录
**角色**: PM（主导）、全角色

## 4. 技术架构

### 4.1 核心组件

```
src/multi-agent/
├── types.ts              # 角色、阶段、决策、产物类型定义
├── role.ts               # 角色基类（模板方法模式）
├── project-manager.ts     # PM 角色：任务分解、进度跟踪、决策
├── agents/
│   ├── product-agent.ts   # 产品经理角色
│   ├── design-agent.ts    # 设计角色
│   ├── developer-agent.ts # 开发者角色
│   ├── qa-agent.ts        # 测试角色
│   └── business-agent.ts  # 业务分析角色
├── meeting.ts             # 会议记录、决策记录
├── artifact.ts            # 产物管理（PRD、设计稿、代码、报告）
├── phase-controller.ts    # 阶段状态机
└── team.ts               # 团队初始化、角色注册
```

### 4.2 工具注册

```typescript
// PM 工具
registerTool("pm_create_project", { idea, target })          // 创建项目
registerTool("pm_plan_phase", { phase })                      // 规划阶段
registerTool("pm_review_deliverable", { artifact })           // 评审产物
registerTool("pm_decide", { options, criteria })             // 做决策
registerTool("pm_advance_phase", {})                          // 进入下一阶段

// 角色工具
registerTool("role_execute_sop", { role, phase, input })      // 执行角色SOP
registerTool("role_delegate", { subTask, toRole })            // 角色间委托
registerTool("artifact_create", { type, content })            // 创建产物
registerTool("artifact_review", { artifactId, verdict })      // 评审产物

// 团队工具
registerTool("team_meeting", { agenda, participants })        // 发起会议
registerTool("team_decision", { topic, options })              // 团队决策投票
registerTool("team_status", {})                                // 查看团队状态
```

## 5. 决策机制

### 5.1 决策类型

| 决策 | 方式 | 通过条件 |
|------|------|----------|
| 想法通过/拒绝 | BA + PM | 可行性评分 ≥ 6/10 |
| PRD 通过 | PM + Designer | 所有 Story 有验收标准 |
| 设计冻结 | PM + Dev + Designer | 设计评审无阻塞问题 |
| 代码合并 | PM + QA | 测试覆盖率 ≥ 80%, 无 P0 缺陷 |
| 上线 | PM + BA | 所有 P0/P1 缺陷已修复 |

### 5.2 升级路径

```
角色遇到阻塞 → 升级到 PM → PM 决策 → 解决 / 升级到团队投票
```

## 6. 自主运行机制

- **阶段自动推进**: 当前阶段所有产出通过评审后，自动通知下一阶段角色开始
- **阻塞检测**: 角色 3 次尝试无法推进，自动升级 PM
- **记忆学习**: 每阶段结束后保存成功模式到 PatternMemory
- **回顾机制**: 项目完成后进行 Retrospective，生成改进建议

## 7. 产出物清单

每个项目产出：
- `project.json` — 项目元信息
- `feasibility.md` — 可行性报告
- `prd.md` — 产品需求文档
- `design/` — 设计稿目录
- `spec/` — 技术规格文档
- `src/` — 源代码
- `test/` — 测试代码
- `assessment.md` — 评估报告
- `retrospective.md` — 回顾记录
