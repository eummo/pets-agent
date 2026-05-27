# TypeScript 开发规范

本规范用于本项目所有 TypeScript 新增、修改和重构工作。它补充
`docs/development-workflow.md` 的交付流程和 `docs/architecture.md` 的 contracts/adapters
边界要求。

## 设计原则

1. 可读性优先于技巧。主流程应能被顺序阅读，复杂判断拆成有业务含义的命名步骤。
2. 类型表达业务约束。让非法状态尽量不能被表示，不能静态表达的输入必须在边界校验。
3. 编排层保持供应商无关。Enterprise WeChat、Claude、MiniMax、GitHub、Codex、pi-agent
   等 SDK 细节只能出现在适配器内。
4. 副作用靠近边界。解析、选择、格式化、权限判断等核心逻辑优先写成纯函数。
5. 测试覆盖可观察行为。优先验证公开函数、系统契约、日志和最终响应，不围绕私有实现写脆弱测试。

## 项目配置基线

- 以仓库现有配置为准：`module` 和 `moduleResolution` 使用 `NodeNext`，运行时目标为
  `ES2022`，代码按 ESM 编写。
- 保持 `strict` 开启，并继续使用 `noUncheckedIndexedAccess`、`exactOptionalPropertyTypes`、
  `noImplicitOverride`、`noPropertyAccessFromIndexSignature`、`useUnknownInCatchVariables`、
  `isolatedModules`、`verbatimModuleSyntax` 等严格选项。
- 不为绕过类型错误放宽 `tsconfig.json`。确需调整编译选项时，必须说明业务原因并同步更新本规范。
- 使用仓库脚本验证代码：`npm run check` 负责 typecheck、lint、unit test 和 build；
  涉及运行时/模型行为时还必须执行 `npm run smoke`。

## 源文件结构

文件顶部按以下顺序排列，各节之间留一个空行：

1. 文件级 JSDoc（`@fileoverview`，如需要）
2. 版权声明（如需要）
3. `import` 语句
4. 实现代码

```typescript
/**
 * @fileoverview Orchestrator entry point for routing inbound messages to agent runtimes.
 */

// import type 用于纯类型导入，不产生运行时依赖
import type { AgentRequest, InboundMessage } from "./contracts.js";
import { AgentOrchestrator } from "./orchestrator.js";

// 实现
export class XxxOrchestrator { ... }
```

## 模块边界

- 稳定契约放在 `src/core/contracts.ts` 或邻近 core 模块。
- 通道、模型、数据库、工具和外部 SDK 实现放在适配器中，例如 `src/wechat`、`src/agent`、
  `src/server`、`src/persistence` 或未来供应商目录。
- `src/core` 不导入供应商 SDK 类型，也不依赖适配器实现。
- `src/index.ts` 是组合根，可以装配具体实现；其他编排代码应依赖抽象契约。
- 服务协作优先使用构造函数注入，避免在函数内部隐式创建外部依赖。

## 类型建模

- 按 ESLint 规则统一使用 `type`，不新增 `interface`，除非需要声明合并且已在评审中说明。
- 对契约、DTO、事件和配置使用 `readonly` 字段；集合默认使用 `readonly T[]`。
- 用字面量联合表达有限状态，例如 `"query" | "mutate" | "deny"`。
- 状态型对象使用 discriminated union，并用 `switch` 或显式分支完成穷尽处理。
- 严禁使用 `any`。外部输入、JSON、SDK 回调和 `catch` 中的错误必须先用 `unknown` 接住，
  再通过类型守卫、schema 校验或显式分支收窄成可用类型。
- 访问可能为空或类型不确定的值前必须先收窄，例如 `typeof value === "string"`、
  `value !== null`、`value !== undefined`、`Array.isArray(value)` 或项目内命名清楚的类型守卫。
- 禁止无意义的非空断言 `!`。只有在值的生命周期已经被同一作用域内的逻辑严格保证时才允许使用，
  并优先改成显式 guard、默认值或更准确的类型建模。
- 区分"字段缺失"和"值为空"：可省略字段使用 `?`，明确为空使用 `null` 或联合类型。不要用
  `undefined` 作为业务含义。
- 空值传递优先通过可选属性和严格联合类型表达，例如 `age?: number`、`name: string | null`。
- 不用宽泛的 `Record<string, unknown>` 代替已知结构。只有动态键值表才使用 `Record`，读取时必须处理
  `undefined`。
- 避免类型断言。优先通过解析器、类型守卫、`satisfies` 或更准确的泛型约束获得类型。

```ts
type IntentResult =
  | { readonly kind: "allowed"; readonly action: "query" | "mutate" }
  | { readonly kind: "denied"; readonly reason: string };

function describeIntent(result: IntentResult): string {
  switch (result.kind) {
    case "allowed":
      return result.action;
    case "denied":
      return result.reason;
  }
}
```

## 导入与导出

### Import 类型选择

| 类型 | 示例 | 使用场景 |
|------|------|----------|
| 命名导入 | `import {Foo} from './foo'` | 大多数场景，推荐优先使用 |
| 类型导入 | `import type {Foo} from './foo'` | 仅类型编译期使用 |
| 命名空间导入 | `import * as foo from './foo'` | 大量符号的 API，需权衡可读性 |
| 默认导入 | `import SomeThing from '...'` | 仅用于强制要求默认导出的第三方库 |

```typescript
// ✓ 优先命名导入，清晰且便于重构
import {describe, it, expect} from './testing';

// ✗ 滥用命名空间反而降低可读性
import * as testing from './testing';
testing.describe('foo', () => { testing.it('bar', ...) });

// ✗ 过长的重命名导入
import {Item as TableviewItem, Header as TableviewHeader} from './tableview';
// ✓ 改用命名空间或直接解构
import * as tableview from './tableview';
```

### 重命名导入

仅在以下情况重命名：
1. 消除命名冲突
2. 符号名由工具生成、不够清晰
3. 需要说明导入内容的业务含义

### 导出规范

```typescript
// ✓ 优先命名导出，便于 IDE 支持和重构
export class Foo { ... }
export const FOO = 1;
export function bar() { return 1; }

// ✗ 禁止默认导出——导致导入名不一致，无法静态检查成员是否存在
export default class Foo { ... }

// ✗ 禁止可变导出 let
export let foo = 3;
// ✓ 改为显式 getter
let foo = 3;
export function getFoo() { return foo; }

// ✗ 不要用容器类做命名空间
export class Container {
  static FOO = 1;
  static bar() { return 1; }
}
// ✓ 直接导出独立常量或函数
export const FOO = 1;
export function bar() { return 1; }
```

### 模块与命名空间

- **必须**使用 ES6 模块（`import`/`export`）
- **禁止**使用 TypeScript 命名空间（`namespace`）
- **禁止**使用 `require` 风格导入（`import x = require(...)`）

## 函数与控制流

- 一个函数只做一个决策或一个转换。超过一个抽象层级时拆分为命名 helper。
- 用早返回处理 guard、权限拒绝、无效输入和空结果，保持主路径缩进浅。
- 深层可选属性读取优先使用可选链 `?.`，不要写多层嵌套 `if` 只为了判空。
- 默认值优先使用空值合并运算符 `??`，不要用冗长三元表达式或 `||` 混淆空字符串、`0`、`false`
  与 `null`/`undefined`。
- 分支有业务含义时使用 `switch`、查表或命名函数，避免长三元表达式链。
- 函数参数超过 3 个或存在多个同类型参数时，改用只读参数对象。
- 导出的函数和 public 方法应有清晰返回类型；局部变量让 TypeScript 推断，除非显式类型能提升可读性。
- 不把日志、网络调用、数据库写入混入纯转换函数。

## 异步与资源

- 所有 Promise 都必须被 `await`、`return` 或显式收集后统一处理。
- 并行无依赖任务用 `Promise.all`；有顺序依赖时保持顺序 `await`，不要隐藏因果关系。
- 超时、取消和重试应靠近外部调用适配器实现，不泄漏到核心编排层。
- 流式事件必须映射到项目契约中的事件类型，不把供应商原始事件向上透传。

## 数组与对象

### 数组

```typescript
// ✗ 禁止使用 Array 构造函数
const a = new Array(2);    // [undefined, undefined]
const b = new Array(2, 3); // [2, 3]

// ✓ 使用字面量或 Array.from
const a = [2];
const b = [2, 3];
Array.from({length: 5}).fill(0);

// ✓ 复制/拼接使用展开语法
const foo2 = [ ...foo, 6, 7 ];

// ✓ 使用解构
const [first, ...rest] = items;
const [a, b, [, d]] = [1, 2, 3, 4];
```

### 对象

```typescript
// ✓ 使用对象解构
const { id, displayName } = user;

// ✓ 使用展开复制
const clone = { ...original };

// ✗ 不使用 Array 构造函数创建数组（见上方）
// ✗ 不使用 for...in 遍历数组，使用 for...of 或 .forEach
```

## 命名

- 契约按能力命名：`AgentRuntime`、`AuthorizationService`、`KnowledgeWorkspaceResolver`。
- 适配器按实现或传输命名：`ClaudeSdkAgentRuntime`、`InMemoryRoleAuthorizationService`。
- 布尔值用谓词：`allowed`、`isStreaming`、`hasSession`、`canMutate`。
- 函数名描述可观察动作：`resolve`、`classify`、`append`、`archive`、`formatError`。
- 判空变量和参数命名保持简洁，避免 `isDataUndefinedOrNull` 这类把实现细节塞进名字的冗长命名。
  优先使用 `value`、`input`、`existingUser`、`hasData`、`isMissing` 等能表达业务意图的名字。
- 避免 `utils`、`helpers`、`manager` 这类模糊命名，除非文件很小且只在局部模块内使用。
- 使用 `Foo`, `Bar` 等命名时确保上下文清晰，避免用 `tmp` 或 `temp` 作为有业务含义的变量名。

## 错误处理与日志

- 用户响应使用安全、清晰的业务错误；原始供应商错误只进入内部日志。
- `catch (error)` 视为 `unknown`，通过统一 helper 转成可记录文本或结构。
- 日志必须支持排查：在已有上下文中记录 `messageId`、`userId`、`workspacePath`、运行时选择、
  权限结果、工具调用和最终状态。
- 永远不要记录 API key、secret、authorization header、access token、refresh token 或完整敏感请求头。
- 日志文件包含 CJK 内容，Windows 上读取 `.harness/logs/*.jsonl` 必须使用 UTF-8 编码。

## 外部输入与运行时校验

- HTTP、WebSocket、SDK 回调、环境变量、配置文件和 JSON 反序列化都属于不可信输入。
- 边界层必须解析并校验输入，再转换成 core 契约。项目已有依赖 `zod` 时，优先用 schema 表达结构。
- 校验错误应返回可行动的信息，但不要泄露内部路径、密钥或供应商请求体。
- 供应商 DTO 与项目 domain type 分离；适配器负责二者转换。

## 测试规范

- 解析、路由、授权、配置、持久化、格式化等确定性逻辑写单元测试。
- 运行时/模型行为回归写入 `src/smoke/regressionSmoke.ts`，断言可观察响应和日志，而非私有实现。
- 测试命名描述被保护的行为，例如 `denies mutation when role lacks permission`。
- 每个 bug 修复都应添加最小回归用例；如果无法自动化，需在说明中写明原因和手动验证步骤。
- Vitest 负责执行测试；TypeScript 类型检查由 `npm run typecheck` 或 `npm run check` 保证。

## 格式化与 Lint

- 代码格式交给 Prettier，不在评审中争论缩进、换行等格式问题。
- ESLint 使用 `@typescript-eslint` 的 strict type-checked 与 stylistic type-checked 配置；新增代码不得靠
  `eslint-disable` 常驻绕过规则。
- 如必须禁用规则，只在最小行范围内禁用，并写明原因。
- 提交前运行 `npm run check`；格式不一致时运行 `npm run format`。

## 注释与文档

- 代码应通过命名和结构解释"做什么"。注释用于说明"为什么这样做"、协议限制、供应商怪癖或非显然约束。
- 不写复述代码的注释。
- 修改系统契约、配置格式、日志结构、运行时行为或开发流程时，同步更新相关文档。

## 模块结构

每个功能模块有自己的公开 API 出口文件（barrel file），类型定义按以下原则分布：

| 类型使用范围 | 存放位置 |
|------------|----------|
| 仅本模块使用 | `src/<module>/contracts.ts` |
| 跨模块共享 | `src/core/contracts.ts` |
| 模块公开类型 | `src/<module>/index.ts`（重导出） |

```typescript
// src/agent/index.ts — 模块公开 API
export type { AgentRequest, AgentResponse, AgentRuntime } from "./contracts.js";
export type { ProgressReporter } from "./contracts.js";

// src/core/contracts.ts — 真正的跨层共享基元
export type { ChannelUser, InboundMessage, OutboundMessage, MessageGateway } from "./contracts.js";
export type AgentConversationMessage = { ... };  // core 和 agent 共用
```

原则：
- 只被少数模块使用的类型不要堆到 `core/contracts.ts`，放在对应领域模块
- `core/contracts.ts` 只放真正跨层共享的基元类型
- 导入方从对应模块的 `index.ts` import，不直接从 `contracts.ts` 导入（除非是 `core/contracts.ts` 的跨层类型）

## 评审清单

- 是否遵守 contracts/adapters 边界？
- 主流程能否顺序读懂，复杂分支是否有业务命名？
- 外部输入是否已校验并收窄？
- 是否完全避免了 `any`、无意义非空断言和供应商类型泄漏？
- `undefined`、`null`、可选字段的含义是否清楚？
- 判空逻辑是否优先使用 `?.`、`??` 和简洁命名？
- 错误是否对用户安全，日志是否足够排查且不含敏感信息？
- 是否添加或更新了合适的单元测试或 smoke 回归？
- `npm run check` 是否通过？涉及运行时行为时 `npm run smoke` 是否通过？

## 参考资料

- [Google TypeScript Style Guide](https://google.github.io/styleguide/tsguide.html)
- [TypeScript Handbook: Narrowing](https://www.typescriptlang.org/docs/handbook/2/narrowing.html)
- [TypeScript Modules Reference](https://www.typescriptlang.org/docs/handbook/modules/reference)
- [TypeScript TSConfig Reference](https://www.typescriptlang.org/tsconfig/)
- [typescript-eslint: Linting with Type Information](https://typescript-eslint.io/getting-started/typed-linting)
- [typescript-eslint: Shared Configs](https://typescript-eslint.io/users/configs)
- [ESLint: Configuration Migration Guide](https://eslint.org/docs/latest/use/configure/migration-guide)
- [Prettier: Rationale](https://prettier.io/docs/rationale.html)
- [Vitest: Guide](https://vitest.dev/guide/)