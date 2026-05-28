# Google TypeScript Style Guide

This guide is based on the internal Google TypeScript style guide, but it has been slightly adjusted to remove Google-internal sections. Google's internal environment has different constraints on TypeScript than you might find outside of Google. The advice here is specifically useful for people authoring code they intend to import into Google, but otherwise may not apply in your external environment.

There is no automatic deployment process for this version as it's pushed on-demand by volunteers.

## Introduction

### Terminology notes

This Style Guide uses [RFC 2119](https://tools.ietf.org/html/rfc2119) terminology when using the phrases _must_, _must not_, _should_, _should not_, and _may_. The terms _prefer_ and _avoid_ correspond to _should_ and _should not_, respectively. Imperative and declarative statements are prescriptive and correspond to _must_.

### Guide notes

All examples given are **non-normative** and serve only to illustrate the normative language of the style guide. That is, while the examples are in Google Style, they may not illustrate the _only_ stylish way to represent the code. Optional formatting choices made in examples must not be enforced as rules.

## Source file basics

### File encoding: UTF-8

Source files are encoded in **UTF-8**.

#### Whitespace characters

Aside from the line terminator sequence, the ASCII horizontal space character (0x20) is the only whitespace character that appears anywhere in a source file. This implies that all other whitespace characters in string literals are escaped.

#### Special escape sequences

For any character that has a special escape sequence (`\'`, `\"`, `\\`, `\b`, `\f`, `\n`, `\r`, `\t`, `\v`), that sequence is used rather than the corresponding numeric escape (e.g `\x0a`, `\u000a`, or `\u{a}`). Legacy octal escapes are never used.

#### Non-ASCII characters

For the remaining non-ASCII characters, use the actual Unicode character (e.g. `∞`). For non-printable characters, the equivalent hex or Unicode escapes (e.g. `\u221e`) can be used along with an explanatory comment.

```typescript
// Perfectly clear, even without a comment.
const units = 'μs';

// Use escapes for non-printable characters.
const output = '\ufeff' + content;  // byte order mark
```

```typescript
// Hard to read and prone to mistakes, even with the comment.
const units = '\u03bcs'; // Greek letter mu, 's'

// The reader has no idea what this is.
const output = '\ufeff' + content;
```

## Source file structure

Files consist of the following, **in order**:

1. Copyright information, if present
2. JSDoc with `@fileoverview`, if present
3. Imports, if present
4. The file's implementation

**Exactly one blank line** separates each section that is present.

### Copyright information

If license or copyright information is necessary in a file, add it in a JSDoc at the top of the file.

### `@fileoverview` JSDoc

A file may have a top-level `@fileoverview` JSDoc. If present, it may provide a description of the file's content, its uses, or information about its dependencies. Wrapped lines are not indented.

Example:

```typescript
/**
 * @fileoverview Description of file. Lorem ipsum dolor sit amet, consectetur
 * adipiscing elit, sed do eiusmod tempor incididunt.
 */
```

### Imports

There are four variants of import statements in ES6 and TypeScript:

| Import type | Example | Use for |
|---|---|---|
| module | `import * as foo from '...';` | TypeScript imports |
| named | `import {SomeThing} from '...';` | TypeScript imports |
| default | `import SomeThing from '...';` | Only for other external code that requires them |
| side-effect | `import '...';` | Only to import libraries for their side-effects on load |

```typescript
// Good: choose between two options as appropriate (see below).
import * as ng from '@angular/core';
import {Foo} from './foo';

// Only when needed: default imports.
import Button from 'Button';

// Sometimes needed to import libraries for their side effects:
import 'jasmine';
import '@polymer/paper-button';
```

#### Import paths

TypeScript code _must_ use paths to import other TypeScript code. Paths _may_ be relative, i.e. starting with `.` or `..`, or rooted at the base directory, e.g. `root/path/to/file`.

Code _should_ use relative imports (`./foo`) rather than absolute imports `path/to/foo` when referring to files within the same (logical) project.

Consider limiting the number of parent steps (`../../../`) as those can make module and path structures hard to understand.

```typescript
import {Symbol1} from 'path/from/root';
import {Symbol2} from '../parent/file';
import {Symbol3} from './sibling';
```

#### Namespace versus named imports

Both namespace and named imports can be used.

Prefer named imports for symbols used frequently in a file or for symbols that have clear names. Named imports can be aliased to clearer names as needed with `as`.

Prefer namespace imports when using many different symbols from large APIs.

```typescript
// Bad: overlong import statement of needlessly namespaced names.
import {Item as TableviewItem, Header as TableviewHeader, Row as TableviewRow,
  Model as TableviewModel, Renderer as TableviewRenderer} from './tableview';

let item: TableviewItem|undefined;
```

```typescript
// Better: use the module for namespacing.
import * as tableview from './tableview';

let item: tableview.Item|undefined;
```

```typescript
import * as testing from './testing';

// Bad: The module name does not improve readability.
testing.describe('foo', () => {
  testing.it('bar', () => {
    testing.expect(null).toBeNull();
    testing.expect(undefined).toBeUndefined();
  });
});
```

```typescript
// Better: give local names for these common functions.
import {describe, it, expect} from './testing';

describe('foo', () => {
  it('bar', () => {
    expect(null).toBeNull();
    expect(undefined).toBeUndefined();
  });
});
```

#### Renaming imports

Code _should_ fix name collisions by using a namespace import or renaming the exports themselves. Code _may_ rename imports (`import {SomeThing as SomeOtherThing}`) if needed.

### Exports

Use named exports in all code:

```typescript
export class Foo { ... }
```

Do not use default exports:

```typescript
export default class Foo { ... } // BAD!
```

Why? Default exports provide no canonical name, which makes central maintenance difficult. Named exports error when importing something that hasn't been declared.

#### Export visibility

Only export symbols that are used outside of the module. Generally minimize the exported API surface of modules.

#### Mutable exports

`export let` is not allowed. If one needs externally accessible and mutable bindings, they _should_ use explicit getter functions.

```typescript
let foo = 3;
window.setTimeout(() => {
  foo = 4;
}, 1000);
// Use an explicit getter to access the mutable export.
export function getFoo() { return foo; }
```

For conditional exports, first do the conditional check, then the export. All exports must be final after the module's body has executed.

```typescript
function pickApi() {
  if (useOtherApi()) return OtherApi;
  return RegularApi;
}
export const SomeApi = pickApi();
```

#### Container classes

Do not create container classes with static methods or properties for the sake of namespacing. Instead, export individual constants and functions.

```typescript
// Good
export const FOO = 1;
export function bar() { return 1; }

// Bad
export class Container {
  static FOO = 1;
  static bar() { return 1; }
}
```

### Import and export type

Use `import type {...}` when you use the imported symbol only as a type. Use regular imports for values:

```typescript
import type {Foo} from './foo';
import {Bar} from './foo';
import {type Foo, Bar} from './foo';
```

Use `export type` when re-exporting a type:

```typescript
export type {AnInterface} from './foo';
```

### Use modules not namespaces

TypeScript `namespace`s are disallowed. Code _must_ refer to code in other files using imports and exports. Code _must not_ use `require` for imports. Always use ES6 module syntax.

## Language features

### Local variable declarations

#### Use const and let

Always use `const` or `let` to declare variables. Use `const` by default, unless a variable needs to be reassigned. Never use `var`.

Variables _must not_ be used before their declaration.

#### One variable per declaration

Every local variable declaration declares only one variable: declarations such as `let a = 1, b = 2;` are not used.

### Array literals

#### Do not use the `Array` constructor

Always use bracket notation to initialize arrays, or `from` to initialize an `Array` with a certain size.

#### Do not define properties on arrays

Do not define or use non-numeric properties on an array (other than `length`). Use a `Map` (or `Object`) instead.

#### Using spread syntax

When using spread syntax, the value being spread _must_ match what is being created. When creating an array, only spread iterables. Primitives (including `null` and `undefined`) _must not_ be spread.

```typescript
// Bad: might be undefined
const bar = [5, ...(shouldUseFoo && foo)];

// Good
const foo = shouldUseFoo ? [7] : [];
const bar = [5, ...foo];
```

#### Array destructuring

Elements should be omitted if they are unused. A final rest element may be included.

```typescript
const [a, b, c, ...rest] = generateResults();
let [, b,, d] = someArray;
```

For function parameters, always specify `[]` as the default value if a destructured array parameter is optional:

```typescript
function destructured([a = 4, b = 2] = []) {}
```

### Object literals

#### Do not use the `Object` constructor

Use an object literal (`{}` or `{a: 0, b: 1}`) instead.

#### Iterating objects

Do not use unfiltered `for (... in ...)` statements. Prefer `for (... of Object.keys(...))` or `Object.entries()`.

```typescript
for (const key of Object.keys(obj)) {
  doWork(key, obj[key]);
}
for (const [key, value] of Object.entries(obj)) {
  doWork(key, value);
}
```

#### Using spread syntax

When creating an object, only objects may be spread; arrays and primitives (including `null` and `undefined`) _must not_ be spread. Later values replace earlier values at the same key.

```typescript
// Bad: might be undefined
const bar = {num: 5, ...(shouldUseFoo && foo)};

// Good
const foo = shouldUseFoo ? {num: 7} : {};
const bar = {num: 5, ...foo};
```

Avoid spreading objects that have prototypes other than the Object prototype (e.g. class definitions, class instances, functions) as the behavior is unintuitive.

#### Computed property names

Computed property names are allowed, and are considered dict-style (quoted) keys unless the computed property is a symbol.

#### Object destructuring

Object destructuring should be kept as simple as possible: a single level of unquoted shorthand properties. Default values go in the left-hand-side of the destructured parameter.

```typescript
interface Options {
  num?: number;
  str?: string;
}

function destructured({num, str = 'default'}: Options = {}) {}
```

### Classes

#### Class declarations

Class declarations _must not_ be terminated with semicolons. Statements that contain class expressions _must_ be terminated with a semicolon.

Method declarations should be separated from surrounding code by a single blank line. The constructor should be separated from surrounding code both above and below by a single blank line.

#### No #private fields

Do not use private fields (`#ident`). Instead, use TypeScript's visibility annotations (`private ident`).

#### Use readonly

Mark properties that are never reassigned outside of the constructor with the `readonly` modifier.

#### Parameter properties

Rather than plumbing an obvious initializer through to a class member, use a TypeScript parameter property.

```typescript
class Foo {
  constructor(private readonly barService: BarService) {}
}
```

#### Field initializers

Initialize class members where they're declared. Properties should never be added to or removed from an instance after the constructor is finished.

#### Getters and setters

Getters _must_ be pure functions (no side effects). At least one accessor for a property _must_ be non-trivial: do not define pass-through accessors only for the purpose of hiding a property.

#### Visibility

- Limit symbol visibility as much as possible.
- TypeScript symbols are public by default. Never use the `public` modifier except when declaring non-readonly public parameter properties.
- Code _must not_ use `obj['foo']` to bypass the visibility of a property.

```typescript
// Bad
class Foo {
  public bar = new Bar();  // public modifier not needed
  constructor(public readonly baz: Baz) {}  // readonly implies public
}

// Good
class Foo {
  bar = new Bar();  // public modifier not needed
  constructor(public baz: Baz) {}  // public modifier allowed for non-readonly
}
```

#### Static methods

- Prefer module-local functions over private static methods.
- Code _must not_ use `this` in a static context.
- Do not rely on dynamic dispatch of static methods.

### Functions

#### Prefer function declarations for named functions

```typescript
// Good
function foo() {
  return 42;
}

// Bad
const foo = () => 42;
```

Arrow functions _may_ be used when an explicit type annotation is required.

#### Do not use function expressions

Use arrow functions instead. Exception: function expressions _may_ be used only if code has to dynamically rebind `this`, or for generator functions.

#### Arrow function bodies

Only use a concise body if the return value of the function is actually used. Otherwise use a block body.

```typescript
// Bad: return value unused
myPromise.then(v => console.log(v));

// Good
myPromise.then(v => {
  console.log(v);
});
```

#### Rebinding `this`

Function expressions and function declarations _must not_ use `this` unless they specifically exist to rebind the `this` pointer. Prefer arrow functions.

#### Parameter initializers

Optional function parameters _may_ be given a default initializer. Initializers _must not_ have any observable side effects. Use default parameters sparingly.

#### Formatting functions

Blank lines at the start or end of the function body are not allowed. A single blank line _may_ be used within function bodies sparingly to create logical groupings.

### this

Only use `this` in class constructors and methods, functions that have an explicit `this` type declared, or in arrow functions defined in a scope where `this` may be used.

### Interfaces

Prefer interfaces over type literal aliases for object types.

```typescript
// Good
interface User {
  firstName: string;
  lastName: string;
}

// Bad
type User = {
  firstName: string,
  lastName: string,
}
```

### String literals

#### Use single quotes

Ordinary string literals are delimited with single quotes (`'`), rather than double quotes (`"`).

Tip: if a string contains a single quote character, consider using a template string to avoid having to escape the quote.

#### No line continuations

Do not use line continuations (ending a line inside a string literal with a backslash).

#### Template literals

Use template literals over complex string concatenation.

### Type coercion

TypeScript code _may_ use `String()`, `Boolean()`, string template literals, or `!!` to coerce types.

Enum values _must not_ be converted to booleans with `Boolean()` or `!!`. Must instead be compared explicitly.

```typescript
// Bad
let enabled = Boolean(level);

// Good
let enabled = level !== SupportLevel.NONE;
```

Code _must_ use `Number()` to parse numeric values, and _must_ check its return for `NaN` values explicitly.

Code _must not_ use unary plus (`+`) to coerce strings to numbers.

Code _must not_ use `parseInt` or `parseFloat` except for non-base-10 strings.

### Control structures

#### Control flow statements and blocks

Control flow statements always use braced blocks, even if the body contains only a single statement.

**Exception:** `if` statements fitting on one line _may_ elide the block: `if (x) x.doFoo();`

#### Iterating containers

Prefer `for (... of someArr)` to iterate over arrays.

`for`-`in` loops may only be used on dict-style objects. Prefer `for`-`of` with `Object.keys`, `Object.values`, or `Object.entries` over `for`-`in`.

#### Exception handling

- Always use `new Error()` when instantiating exceptions.
- Only throw (subclasses of) `Error`.
- When catching errors, code _should_ assume that all thrown errors are instances of `Error`.
- Empty catch blocks are very rarely correct. When appropriate, explain why in a comment.

#### Switch statements

All `switch` statements _must_ contain a `default` statement group, even if it contains no code. The `default` statement group must be last.

Non-empty statement groups _must not_ fall through.

#### Equality checks

Always use triple equals (`===`) and not equals (`!==`).

**Exception:** Comparisons to `null` _may_ use `==` and `!=` to cover both `null` and `undefined`.

### Type and non-nullability assertions

Type assertions and non-nullability assertions are unsafe. You _should not_ use them without an obvious or explicit reason.

Instead, prefer runtime checks:

```typescript
// Bad
(x as Foo).foo();
y!.bar();

// Good
if (x instanceof Foo) { x.foo(); }
if (y) { y.bar(); }
```

When assertions are used, document why:

```typescript
// x is a Foo, because ...
(x as Foo).foo();
```

Type assertions _must_ use the `as` syntax (not angle brackets). Use `unknown` (not `any`) as the intermediate type for double assertions.

Use type annotations (`: Foo`) instead of type assertions (`as Foo`) on object literals.

### Disallowed features

- Do not instantiate wrapper classes (`new String()`, `new Boolean()`, `new Number()`).
- Do not rely on Automatic Semicolon Insertion (ASI). Explicitly end all statements with semicolons.
- Code _must not_ use `const enum`; use plain `enum` instead.
- Debugger statements _must not_ be included in production code.
- Do not use `with`.
- Do not use `eval` or `Function(...string)`.
- Do not use non-standard ECMAScript or Web Platform features.
- Never modify builtin objects.

## Naming

### Identifiers

Identifiers _must_ use only ASCII letters, digits, underscores (for constants and structured test method names), and (rarely) the '$' sign.

#### Naming style

- Do not use trailing or leading underscores for private properties or methods.
- Do not use the `opt_` prefix for optional parameters.
- Do not mark interfaces specially (`IMyInterface` or `MyFooInterface`).
- `_` _must not_ be used as an identifier by itself.

#### Descriptive names

Names _must_ be descriptive and clear to a new reader.

**Exception:** Variables that are in scope for 10 lines or fewer, including arguments that are _not_ part of an exported API, _may_ use short (e.g. single letter) variable names.

#### Camel case

Treat abbreviations like acronyms as whole words: `loadHttpUrl`, not `loadHTTPURL`.

### Rules by identifier type

| Style | Category |
|---|---|
| `UpperCamelCase` | class / interface / type / enum / decorator / type parameters |
| `lowerCamelCase` | variable / parameter / function / method / property / module alias |
| `CONSTANT_CASE` | global constant values, including enum values |

#### Constants

`CONSTANT_CASE` indicates that a value is _intended_ to not be changed. Only symbols declared on the module level, static fields of module level classes, and values of module level enums _may_ use `CONSTANT_CASE`.

## Type system

### Type inference

Leave out type annotations for trivially inferred types. Explicitly specify types for empty collections and complex expressions.

```typescript
const x = 15;  // Type inferred.
const x: boolean = true;  // Bad: 'boolean' does not aid readability
const x = new Set<string>();  // Good: explicit type for empty collection
```

#### Return types

Whether to include return type annotations is up to the code author. Reviewers _may_ ask for annotations to clarify complex return types.

### Undefined and null

TypeScript code can use either `undefined` or `null` to denote absence of a value.

Type aliases _must not_ include `|null` or `|undefined` in a union type. Add them only where the alias is actually used.

Prefer optional fields (`?`) over `|undefined`.

### Use structural types

When providing a structural-based implementation, explicitly include the type at the declaration:

```typescript
const foo: Foo = {
  a: 123,
  b: 'abc',
}
```

### Prefer interfaces over type literal aliases

For object types, use interfaces instead of type aliases.

### `Array<T>` Type

For simple types, use `T[]` or `readonly T[]`. For complex types, use `Array<T>`.

```typescript
let a: string[];
let b: readonly string[];
let e: Array<{n: number, s: string}>;
let f: Array<string|number>;
```

### Indexable types / index signatures

Consider using ES6 `Map` and `Set` types instead of object-based associative arrays.

### Mapped and conditional types

Always use the simplest type construct that can possibly express your code. A little bit of repetition or verbosity is often much cheaper than the long term cost of complex type expressions.

```typescript
// Instead of:
type FoodPreferences = Pick<User, 'favoriteIcecream'|'favoriteChocolate'>;

// Prefer:
interface FoodPreferences {
  favoriteIcecream: string;
  favoriteChocolate: string;
}
interface User extends FoodPreferences {
  shoeSize: number;
}
```

### `any` Type

Consider _not_ using `any`. Alternatives:

1. Provide a more specific type
2. Use `unknown`
3. Suppress the lint warning and document why

### `{}` Type

Code _should not_ use `{}` for most use cases. Prefer `unknown`, `Record<string, T>`, or `object`.

### Tuple types

Use tuple types for pairs instead of creating a Pair interface.

```typescript
function splitInHalf(input: string): [string, string] {
  return [x, y];
}
```

### Wrapper types

Never use `String`, `Boolean`, `Number`, or `Object`. Use lowercase primitives.

### Return type only generics

Avoid creating APIs that have return type only generics.

## Toolchain requirements

### TypeScript compiler

All TypeScript files must pass type checking using the standard tool chain.

#### @ts-ignore

Do not use `@ts-ignore` nor `@ts-expect-error` or `@ts-nocheck`. Exception: `@ts-expect-error` may be used in unit tests sparingly.

## Comments and documentation

### JSDoc versus comments

- Use `/** JSDoc */` for documentation comments.
- Use `// line comments` for implementation comments.

### Multi-line comments

Multi-line comments _must_ use multiple single-line comments (`//`-style), not block comment style (`/* */`).

### JSDoc general form

```typescript
/**
 * Multiple lines of JSDoc text are written here,
 * wrapped normally.
 * @param arg A number to do something to.
 */
function doSomething(arg: number) {}
```

### Markdown in JSDoc

JSDoc is written in Markdown. Use Markdown lists for structured content.

### JSDoc tags

Most tags must occupy their own line. Line-wrapped block tags are indented four spaces.
