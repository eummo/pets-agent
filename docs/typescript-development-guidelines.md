# TypeScript 开发规范

本规范用于本项目所有 TypeScript 新增、修改和重构工作。它补充
`docs/development-workflow.md` 的交付流程和 `docs/architecture.md` 的 contracts/adapters
边界要求。

规范基于 [Google TypeScript Style Guide](https://google.github.io/styleguide/tsguide.html)，
并针对项目架构做了补充。冲突处以 Google 规范为准。

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

## 源文件基础

### 文件编码

源文件使用 **UTF-8** 编码。除行终止符外，只使用 ASCII 水平空格字符（0x20），其他空白字符
在字符串字面量中必须转义。

### 转义序列

- 有特殊转义序列的字符（`\'`、`\"`、`\\`、`\b`、`\f`、`\n`、`\r`、`\t`、`\v`）必须使用
  转义序列，而非数字转义（如 `\x0a`）。禁止旧式八进制转义。
- 非 ASCII 可打印字符直接使用 Unicode 字符（如 `∞`）。不可打印字符可用十六进制或 Unicode
  转义，并附解释性注释。

```typescript
// ✓ 清晰，无需注释
const units = "μs";

// ✓ 不可打印字符用转义并注释
const output = "\ufeff" + content; // byte order mark
```

### 源文件结构

文件顶部按以下顺序排列，各节之间留一个空行：

1. 版权声明（如需要）
2. 文件级 JSDoc（`@fileoverview`，如需要）
3. `import` 语句
4. 实现代码

```typescript
/**
 * @fileoverview Orchestrator entry point for routing inbound messages to agent runtimes.
 */

import type { AgentRequest, InboundMessage } from "./contracts.js";
import { AgentOrchestrator } from "./orchestrator.js";

export class XxxOrchestrator { ... }
```

## 模块边界

- 稳定契约从 `src/core/index.ts` 或对应模块的 `index.ts` 导出。
- 通道、模型、数据库、工具和外部 SDK 实现放在适配器中，例如 `src/wechat`、`src/agent`、
  `src/server`、`src/persistence` 或未来供应商目录。
- `src/core` 不导入供应商 SDK 类型，也不依赖适配器实现。
- `src/index.ts` 是组合根，可以装配具体实现；其他编排代码应依赖抽象契约。
- 服务协作优先使用构造函数注入，避免在函数内部隐式创建外部依赖。

## 类型建模

### interface 与 type

对对象类型**使用 `type`**，与当前 ESLint `@typescript-eslint/consistent-type-definitions`
配置保持一致。`interface` 只在确实需要声明合并或实现第三方扩展点时使用，并在评审中说明原因。

```typescript
// ✓ 对象类型使用 type
type User = {
  firstName: string;
  lastName: string;
};

// ✗ 本项目默认不使用 interface 表达普通对象类型
interface User {
  firstName: string;
  lastName: string;
}
```

以下场景仍使用 `type`：

- 联合类型（`type Result = Success | Failure`）
- 交叉类型（`type Combined = A & B`）
- 字面量联合（`type Status = "active" | "inactive"`）
- 工具类型（`type Readonly<T> = ...`）
- discriminated union（`type IntentResult = { kind: "allowed" } | { kind: "denied" }`）

```typescript
// ✓ 联合类型用 type
type IntentResult =
  | { readonly kind: "allowed"; readonly action: "query" | "mutate" }
  | { readonly kind: "denied"; readonly reason: string };

// ✓ 字面量联合用 type
type AuthorizationAction = "query" | "mutate" | "update_kb";
```

### 通用规则

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
  优先使用可选字段（`?`）而非 `| undefined`。
- 不用宽泛的 `Record<string, unknown>` 代替已知结构。只有动态键值表才使用 `Record`，读取时必须处理
  `undefined`。
- 避免类型断言。优先通过解析器、类型守卫、`satisfies` 或更准确的泛型约束获得类型。
  确需使用时必须用 `as` 语法（不用尖括号），并用 `unknown` 作为中间类型，附注释说明原因。
- 使用结构化类型：提供结构化实现时，在声明处显式标注类型（`const foo: Foo = {...}`）。
- 类型别名中**不得**包含 `| null` 或 `| undefined`，只在实际使用处添加。
- 优先使用 ES6 `Map` 和 `Set` 而非基于对象的关联数组。
- 使用元组类型代替 Pair 接口：`[string, number]` 而非 `{first: string; second: number}`。
- 简单类型用 `T[]` 或 `readonly T[]`；复杂类型（如联合元素、对象元素）用 `Array<T>`。
- 映射类型和条件类型：始终使用能表达意图的最简类型构造。少量重复往往比复杂类型表达式
  的长期维护成本更低。
- 避免创建仅有返回类型泛型的 API。
- 禁止使用 `{}` 类型。使用 `unknown`、`Record<string, T>` 或 `object` 代替。
- 禁止使用包装类型 `String`、`Boolean`、`Number`、`Object`，使用小写原始类型。

## 导入与导出

### Import 类型选择

| 类型         | 示例                             | 使用场景                         |
| ------------ | -------------------------------- | -------------------------------- |
| 命名导入     | `import {Foo} from './foo'`      | 大多数场景，推荐优先使用         |
| 类型导入     | `import type {Foo} from './foo'` | 仅类型编译期使用                 |
| 命名空间导入 | `import * as foo from './foo'`   | 大量符号的 API，需权衡可读性     |
| 默认导入     | `import SomeThing from '...'`    | 仅用于强制要求默认导出的第三方库 |
| 副作用导入   | `import '...'`                   | 仅用于导入库的副作用             |

```typescript
// ✓ 优先命名导入，清晰且便于重构
import {describe, it, expect} from './testing';

// ✗ 滥用命名空间反而降低可读性
import * as testing from './testing';
testing.describe('foo', () => { testing.it('bar', ...) });

// ✗ 过长的重命名导入
import {Item as TableviewItem, Header as TableviewHeader} from './tableview';
// ✓ 改用命名空间
import * as tableview from './tableview';
```

### 导入路径

- 必须使用路径导入其他 TypeScript 代码。同项目内优先使用相对导入（`./foo`）。
- 限制父级步数（`../../../`），过多层级说明模块结构需要调整。

### 重命名导入

仅在以下情况重命名：

1. 消除命名冲突
2. 符号名由工具生成、不够清晰
3. 需要说明导入内容的业务含义

重命名时优先使用命名空间导入或重命名导出本身。

### 类型导入导出

仅作为类型使用的符号必须用 `import type` 导入：

```typescript
import type { Foo } from "./foo";
import { Bar } from "./foo";
import { type Foo, Bar } from "./foo"; // 混合导入时内联 type
```

重导出类型时使用 `export type`：

```typescript
export type { SomeType } from "./foo";
```

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

只导出模块外需要使用的符号，尽量最小化导出 API 面。

条件导出时先完成条件判断再导出。所有导出在模块体执行完毕后必须是最终的：

```typescript
function pickApi() {
  if (useOtherApi()) return OtherApi;
  return RegularApi;
}
export const SomeApi = pickApi();
```

### 模块与命名空间

- **必须**使用 ES6 模块（`import`/`export`）
- **禁止**使用 TypeScript 命名空间（`namespace`）
- **禁止**使用 `require` 风格导入（`import x = require(...)`）

## 变量声明

- 始终使用 `const` 或 `let` 声明变量。默认使用 `const`，仅在需要重新赋值时使用 `let`。禁止 `var`。
- 每个声明只声明一个变量：禁止 `let a = 1, b = 2;`。
- 变量不得在声明前使用。

## 函数与控制流

- 优先使用函数声明定义命名函数，而非 `const foo = () => ...`。
- 箭头函数仅在需要显式类型注解或保留词法 `this` 时使用。
- 箭头函数仅在返回值被使用时使用简洁体；否则使用块体。
- 函数表达式仅在需要动态绑定 `this` 或生成器函数时使用，否则用箭头函数。
- 一个函数只做一个决策或一个转换。超过一个抽象层级时拆分为命名 helper。
- 用早返回处理 guard、权限拒绝、无效输入和空结果，保持主路径缩进浅。
- 深层可选属性读取优先使用可选链 `?.`，不要写多层嵌套 `if` 只为了判空。
- 默认值优先使用空值合并运算符 `??`，不要用冗长三元表达式或 `||` 混淆空字符串、`0`、`false`
  与 `null`/`undefined`。
- 分支有业务含义时使用 `switch`、查表或命名函数，避免长三元表达式链。
- 函数参数超过 3 个或存在多个同类型参数时，改用只读参数对象。
- 可选参数可提供默认初始化器，但初始化器不得有可观察副作用。谨慎使用默认参数。
- 导出的函数和 public 方法应有清晰返回类型；局部变量让 TypeScript 推断，除非显式类型能提升可读性。
- 不把日志、网络调用、数据库写入混入纯转换函数。
- 函数体首尾不得有空行。可在函数体内使用单个空行进行逻辑分组，但需克制。

## 类

- 类声明不以分号结尾。包含类表达式的语句必须以分号结尾。
- 方法声明与周围代码用单个空行分隔。构造函数与上下代码均用单个空行分隔。
- 禁止使用 `#private` 字段，使用 TypeScript 的 `private` 可见性注解。
- 不在构造函数外重新赋值的属性标记 `readonly`。
- 对于显而易见的初始化传递，使用参数属性：`constructor(private readonly bar: Bar) {}`。
- 在声明处初始化类成员。构造函数完成后不得向实例添加或删除属性。
- Getter 必须是纯函数（无副作用）。属性的两个访问器中至少一个必须非平凡。
- 尽量限制符号可见性。TypeScript 默认 public，除声明非只读的 public 参数属性外
  不使用 `public` 修饰符。不得用 `obj['foo']` 绕过属性可见性。
- 优先使用模块局部函数而非私有静态方法。静态方法中不得使用 `this`。
  不要依赖静态方法的动态分派。

## this

仅在类构造函数和方法、声明了显式 `this` 类型的函数、或允许使用 `this` 的作用域中定义的
箭头函数内使用 `this`。函数表达式和函数声明不得使用 `this`，除非其目的就是重新绑定
`this` 指针。优先使用箭头函数。

## 数组与对象

### 数组

```typescript
// ✗ 禁止使用 Array 构造函数
const a = new Array(2);
const b = new Array(2, 3);

// ✓ 使用字面量或 Array.from
const a = [2];
const b = [2, 3];
Array.from({ length: 5 }).fill(0);

// ✓ 复制/拼接使用展开语法
const foo2 = [...foo, 6, 7];

// ✓ 使用解构
const [first, ...rest] = items;
const [a, b, [, d]] = [1, 2, 3, 4];
```

- 禁止在数组上定义或使用非数字属性（`length` 除外）。需要时使用 `Map` 或 `Object`。
- 展开语法：创建数组时只能展开可迭代对象。原始值（包括 `null` 和 `undefined`）不得展开。

```typescript
// ✗ 可能是 undefined
const bar = [5, ...(shouldUseFoo && foo)];

// ✓
const foo = shouldUseFoo ? [7] : [];
const bar = [5, ...foo];
```

- 解构时省略未使用的元素。可包含尾部 rest 元素。
- 函数参数中解构数组参数可选时，始终以 `[]` 为默认值：`function f([a = 4] = []) {}`

### 对象

```typescript
// ✓ 使用对象字面量
const obj = {};

// ✓ 使用对象解构
const { id, displayName } = user;

// ✓ 使用展开复制
const clone = { ...original };
```

- 禁止使用 `Object` 构造函数，使用对象字面量。
- 禁止使用未过滤的 `for (... in ...)`。优先 `for (... of Object.keys(...))`
  或 `Object.entries()`。
- 创建对象时只能展开对象；数组和原始值不得展开。同键后面的值覆盖前面的值。
- 避免展开原型非 Object.prototype 的对象（类定义、类实例、函数），行为不直观。

```typescript
// ✗ 可能是 undefined
const bar = { num: 5, ...(shouldUseFoo && foo) };

// ✓
const foo = shouldUseFoo ? { num: 7 } : {};
const bar = { num: 5, ...foo };
```

- 对象解构保持简单：单层无引号简写属性。默认值放在解构参数左侧。
- 计算属性名允许使用，视为字典式（引号）键，除非计算属性是 symbol。

## 字符串字面量

- 普通字符串使用**双引号**（`"`），与当前 Prettier 输出保持一致。包含双引号时考虑模板字符串避免转义。
- 禁止行续行（字符串字面量内以反斜杠结尾换行）。
- 复杂字符串拼接优先使用模板字面量。

## 类型强制转换

- 可使用 `String()`、`Boolean()`、字符串模板或 `!!` 进行类型转换。
- 枚举值**不得**用 `Boolean()` 或 `!!` 转换为布尔值，必须显式比较。
- 数值解析**必须**使用 `Number()`，并显式检查 `NaN`。
- **禁止**使用一元加号（`+`）将字符串转为数字。
- **禁止**使用 `parseInt` 或 `parseFloat`，非十进制字符串除外。

```typescript
// ✗ 枚举转布尔
let enabled = Boolean(level);

// ✓ 显式比较
let enabled = level !== SupportLevel.NONE;
```

## 控制结构

- 控制流语句始终使用花括号块，即使体只有一条语句。
  **例外**：适合一行的 `if` 可省略块：`if (x) x.doFoo();`
- 遍历数组优先 `for (... of someArr)`。`for...in` 仅用于字典式对象，
  优先 `for...of` 配合 `Object.keys/values/entries`。
- 所有 `switch` 语句**必须**包含 `default` 分支，即使无代码。`default` 必须在最后。
  非空分支**不得**穿透。
- 始终使用三等号（`===`）和不等号（`!==`）。
  **例外**：与 `null` 比较时可使用 `==`/`!=` 同时覆盖 `null` 和 `undefined`。

## 异常处理

- 实例化异常时始终使用 `new Error()`。只抛出 `Error` 的子类。
- 捕获错误时应假定所有抛出的错误都是 `Error` 实例。
- 空 catch 块极少正确。适当时用注释说明为何忽略。
- `catch (error)` 视为 `unknown`，通过统一 helper 转成可记录文本或结构。

## 异步与资源

- 所有 Promise 都必须被 `await`、`return` 或显式收集后统一处理。
- 并行无依赖任务用 `Promise.all`；有顺序依赖时保持顺序 `await`，不要隐藏因果关系。
- 超时、取消和重试应靠近外部调用适配器实现，不泄漏到核心编排层。
- 流式事件必须映射到项目契约中的事件类型，不把供应商原始事件向上透传。

## 命名

### 命名风格

| 风格             | 类别                                                   |
| ---------------- | ------------------------------------------------------ |
| `UpperCamelCase` | class / interface / type / enum / decorator / 类型参数 |
| `lowerCamelCase` | 变量 / 参数 / 函数 / 方法 / 属性 / 模块别名            |
| `CONSTANT_CASE`  | 全局常量值，包括枚举值                                 |

- 标识符只使用 ASCII 字母、数字、下划线（常量和结构化测试方法名）和（罕见）`$`。
- 缩写作为完整单词处理：`loadHttpUrl` 而非 `loadHTTPURL`。
- 不为私有属性或方法添加前后下划线。不为可选参数添加 `opt_` 前缀。
  不为接口添加特殊前缀（`IMyInterface`）或后缀（`MyFooInterface`）。
- `_` 不得单独作为标识符。
- 名称必须对新读者具有描述性和清晰性。
  **例外**：作用域不超过 10 行的变量和非导出 API 参数可使用短名。

### 项目特定命名约定

- 契约按能力命名：`AgentRuntime`、`AuthorizationService`、`KnowledgeWorkspaceResolver`。
- 适配器按实现或传输命名：`ClaudeSdkAgentRuntime`、`InMemoryRoleAuthorizationService`。
- 布尔值用谓词：`allowed`、`isStreaming`、`hasSession`、`canMutate`。
- 函数名描述可观察动作：`resolve`、`classify`、`append`、`archive`、`formatError`。
- 判空变量和参数命名保持简洁，避免 `isDataUndefinedOrNull` 这类把实现细节塞进名字的冗长命名。
  优先使用 `value`、`input`、`existingUser`、`hasData`、`isMissing` 等能表达业务意图的名字。
- 避免 `utils`、`helpers`、`manager` 这类模糊命名，除非文件很小且只在局部模块内使用。

## 类型推断

- 省略可平凡推断的类型注解。空集合和复杂表达式显式指定类型。
- 导出函数和 public 方法的返回类型是否注解由作者决定。评审者可要求为复杂返回类型添加注解。

```typescript
const x = 15; // 类型可推断
const x: boolean = true; // ✗ 'boolean' 无助于可读性
const x = new Set<string>(); // ✓ 空集合显式类型
```

## 错误处理与日志

- 用户响应使用安全、清晰的业务错误；原始供应商错误只进入内部日志。
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
- 文档注释使用 `/** JSDoc */`，实现注释使用 `// 行注释`。
- 多行注释**必须**使用多个单行注释（`//`），不得使用块注释（`/* */`）。
- JSDoc 以 Markdown 编写。结构化内容使用 Markdown 列表。
- 修改系统契约、配置格式、日志结构、运行时行为或开发流程时，同步更新相关文档。

## 工具链要求

- 所有 TypeScript 文件必须通过标准工具链的类型检查。
- **禁止**使用 `@ts-ignore`、`@ts-expect-error` 或 `@ts-nocheck`。
  **例外**：`@ts-expect-error` 可在单元测试中少量使用。
- **禁止**使用 `const enum`，使用普通 `enum`。
- Debugger 语句不得出现在生产代码中。
- 不依赖自动分号插入（ASI），所有语句显式以分号结尾。

## 禁止使用的特性

- 禁止实例化包装类（`new String()`、`new Boolean()`、`new Number()`）。
- 禁止使用 `with`。
- 禁止使用 `eval` 或 `Function(...string)`。
- 禁止使用非标准 ECMAScript 或 Web 平台特性。
- 禁止修改内置对象。

## 模块结构

每个功能模块有自己的公开 API 出口文件（barrel file），类型定义按以下原则分布：

| 类型使用范围 | 存放位置                                          |
| ------------ | ------------------------------------------------- |
| 仅本模块使用 | 模块内部文件，不从模块外导入                      |
| 跨模块共享   | `src/core/index.ts` 或拥有该契约的模块 `index.ts` |
| 模块公开类型 | `src/<module>/index.ts`（重导出）                 |

```typescript
// src/agent/index.ts — 模块公开 API
export type AgentRuntime = {
  readonly name: string;
  run(request: AgentRequest): Promise<AgentResponse>;
};

// src/core/index.ts — 真正的跨层共享基元
export type MessageGateway = {
  handle(message: InboundMessage): Promise<OutboundMessage>;
};
export type AgentConversationMessage = { ... }; // core 和 agent 共用
```

原则：

- 只被少数模块使用的类型不要堆到 `core/index.ts`，放在对应领域模块
- `core/index.ts` 只放真正跨层共享的基元类型
- 导入方从对应模块的 `index.ts` import，不跨模块直连内部实现文件

## 评审清单

- 是否遵守 contracts/adapters 边界？
- 主流程能否顺序读懂，复杂分支是否有业务命名？
- 外部输入是否已校验并收窄？
- 是否完全避免了 `any`、无意义非空断言和供应商类型泄漏？
- `undefined`、`null`、可选字段的含义是否清楚？
- 判空逻辑是否优先使用 `?.`、`??` 和简洁命名？
- 错误是否对用户安全，日志是否足够排查且不含敏感信息？
- 是否添加或更新了合适的单元测试或 smoke 回归？每次改动后需同时评估是否需要新增冒烟测试案例。
- `npm run check` 是否通过？涉及运行时行为时 `npm run smoke` 是否通过？
- 对象类型是否遵守当前 ESLint 约束使用了 `type`？

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
