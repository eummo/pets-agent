import Anthropic from "@anthropic-ai/sdk";
import type { ContentBlock, Message, MessageCreateParamsNonStreaming } from "@anthropic-ai/sdk/resources/messages";
import { execFile } from "node:child_process";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { z } from "zod";
import type { AgentRequest, AgentResponse, AgentRuntime } from "../core/ports.js";
import type { JsonlLogger } from "../logging/jsonlLogger.js";

const execFileAsync = promisify(execFile);

type AnthropicCompatibleClient = {
  readonly messages: {
    create(params: MessageCreateParamsNonStreaming): Promise<Message>;
  };
};

export type AnthropicCodeChangeRuntimeOptions = {
  readonly baseUrl: string;
  readonly apiKey: string;
  readonly modelId: string;
  readonly maxTokens?: number;
  readonly rawLogger?: JsonlLogger;
  readonly client?: AnthropicCompatibleClient;
};

const rawCodeChangeSchema = z.object({
  summary: z.string().min(1),
  changes: z.array(
    z.object({
      path: z.string().min(1),
      content: z.string().optional(),
      contentLines: z.array(z.string()).optional()
    })
  ).min(1),
  verificationCommand: z.string().min(1).optional()
});

type CodeChange = {
  readonly summary: string;
  readonly changes: readonly { readonly path: string; readonly content: string }[];
  readonly verificationCommand?: string;
};

export class AnthropicCodeChangeRuntime implements AgentRuntime {
  public readonly name = "anthropic-code-change";
  private readonly client: AnthropicCompatibleClient;
  private readonly modelId: string;
  private readonly maxTokens: number;
  private readonly rawLogger: JsonlLogger | undefined;

  public constructor(options: AnthropicCodeChangeRuntimeOptions) {
    this.client =
      options.client ??
      new Anthropic({
        apiKey: options.apiKey,
        baseURL: options.baseUrl
      });
    this.modelId = options.modelId;
    this.maxTokens = options.maxTokens ?? 8192;
    this.rawLogger = options.rawLogger;
  }

  public async run(request: AgentRequest): Promise<AgentResponse> {
    await request.progress?.({
      stage: "code_change.started",
      message: "代码变更流程已开始。",
      data: { runtime: this.name, modelId: this.modelId, workspacePath: request.workspacePath }
    });

    await request.progress?.({
      stage: "code_change.snapshot_started",
      message: "正在读取目标仓库上下文。"
    });
    const workspaceSnapshot = await buildEditableWorkspaceSnapshot(request.workspacePath);
    await request.progress?.({
      stage: "code_change.snapshot_ready",
      message: "目标仓库上下文已准备完成。"
    });
    const createParams: MessageCreateParamsNonStreaming = {
      model: this.modelId,
      max_tokens: this.maxTokens,
      system: [
        "You are a coding agent that edits a selected source repository.",
        "Return only valid JSON. Do not wrap it in Markdown.",
        "The JSON shape is:",
        "{\"summary\":\"...\",\"changes\":[{\"path\":\"relative/file.ts\",\"contentLines\":[\"line 1\",\"line 2\"]}],\"verificationCommand\":\"npm test\"}",
        "Every changed file must be represented with complete replacement contents.",
        "Use contentLines instead of multiline strings. Each contentLines item is one exact file line without a trailing newline.",
        "Use relative paths inside the selected workspace. Do not include absolute paths.",
        "Limit this pass to at most 3 changed files unless the user explicitly names more files.",
        "Do not ask for clarification. If the request is broad, implement a small safe vertical slice that fits the existing workspace.",
        "For an order-system refactor, create or improve order domain types and an order service entry point.",
        "Keep the change focused on the user's request."
      ].join("\n"),
      messages: [
        {
          role: "user",
          content: [
            `User request: ${request.text}`,
            `Workspace path: ${request.workspacePath}`,
            "",
            "Editable workspace snapshot:",
            workspaceSnapshot
          ].join("\n")
        }
      ]
    };

    await this.rawLogger?.write({
      type: "code_change.request",
      runtime: this.name,
      modelId: this.modelId,
      userId: request.user.id,
      workspacePath: request.workspacePath,
      request: createParams
    });
    await request.progress?.({
      stage: "code_change.sdk_request_sent",
      message: "已向 Claude/Anthropic SDK 发送代码变更请求。",
      data: { modelId: this.modelId }
    });

    const firstAttempt = await this.createCodeChange(createParams, request, 1);
    const codeChange = firstAttempt.codeChange;
    await request.progress?.({
      stage: "code_change.files_writing",
      message: "正在应用模型返回的文件变更。",
      data: { fileCount: codeChange.changes.length }
    });
    const writtenFiles = await applyCodeChange(request.workspacePath, codeChange.changes, request.progress);
    await request.progress?.({
      stage: "code_change.verification_started",
      message: "正在运行验证命令。",
      data: { command: codeChange.verificationCommand ?? "npm test" }
    });
    const verification = await runVerification(request.workspacePath, codeChange.verificationCommand ?? "npm test");
    await request.progress?.({
      stage: "code_change.verification_finished",
      message: verification.ok ? "验证命令已通过。" : "验证命令失败。",
      data: { command: verification.command, ok: verification.ok }
    });

    await this.rawLogger?.write({
      type: "code_change.response",
      runtime: this.name,
      modelId: this.modelId,
      userId: request.user.id,
      extractedText: firstAttempt.text,
      appliedFiles: writtenFiles,
      verification
    });
    await request.progress?.({
      stage: "code_change.completed",
      message: "代码变更流程已完成。",
      data: { appliedFiles: writtenFiles, verification: { command: verification.command, ok: verification.ok } }
    });

    return {
      text: [
        "已通过 Claude/Anthropic SDK 执行代码变更流程。",
        `摘要：${codeChange.summary}`,
        `修改文件：${writtenFiles.join(", ")}`,
        `验证：${verification.command} ${verification.ok ? "通过" : "失败"}.`,
        ...(verification.ok ? [] : [verification.output])
      ].join("\n")
    };
  }

  public disposeSession(): Promise<void> {
    return Promise.resolve();
  }

  private async createCodeChange(
    createParams: MessageCreateParamsNonStreaming,
    request: AgentRequest,
    attempt: 1 | 2
  ): Promise<{ readonly text: string; readonly codeChange: CodeChange }> {
    const message = await this.client.messages.create(createParams);
    const text = extractText(message.content);

    await this.rawLogger?.write({
      type: "code_change.raw_response",
      runtime: this.name,
      modelId: this.modelId,
      userId: request.user.id,
      attempt,
      response: message,
      extractedText: text
    });
    await request.progress?.({
      stage: "code_change.sdk_response_received",
      message: "已收到 Claude/Anthropic SDK 原始响应。",
      data: { attempt, textLength: text.length }
    });

    try {
      return { text, codeChange: parseCodeChange(text) };
    } catch (error) {
      if (attempt === 2) {
        throw error;
      }

      const retryParams: MessageCreateParamsNonStreaming = {
        ...createParams,
        messages: [
          ...createParams.messages,
          {
            role: "assistant",
            content: text
          },
          {
            role: "user",
            content: [
              "The previous response was not valid JSON for the requested schema.",
              `Parser error: ${error instanceof Error ? error.message : String(error)}`,
              "Return only corrected JSON with summary, changes, and verificationCommand.",
              "The changes field is required and must include at least one file replacement.",
              "Use contentLines arrays, not multiline strings. Do not ask for clarification."
            ].join("\n")
          }
        ]
      };

      await this.rawLogger?.write({
        type: "code_change.retry",
        runtime: this.name,
        modelId: this.modelId,
        userId: request.user.id,
        error: error instanceof Error ? error.message : String(error)
      });
      await request.progress?.({
        stage: "code_change.sdk_retry",
        message: "模型返回不是有效 JSON，正在请求 SDK 修复结构化输出。",
        data: { error: error instanceof Error ? error.message : String(error) }
      });

      return this.createCodeChange(retryParams, request, 2);
    }
  }
}

async function buildEditableWorkspaceSnapshot(workspacePath: string): Promise<string> {
  const files = await discoverEditableFiles(path.resolve(workspacePath), 20);
  const sections: string[] = [];

  for (const filePath of files) {
    const relativePath = path.relative(workspacePath, filePath);
    const content = await readFile(filePath, "utf8");
    sections.push([`--- ${relativePath} ---`, truncate(content, 8000)].join("\n"));
  }

  return sections.length > 0 ? sections.join("\n\n") : "No editable files were found.";
}

async function discoverEditableFiles(rootPath: string, maxFiles: number): Promise<string[]> {
  const files: string[] = [];
  await collectEditableFiles(rootPath, rootPath, files, maxFiles);
  return files;
}

async function collectEditableFiles(
  rootPath: string,
  currentPath: string,
  files: string[],
  maxFiles: number
): Promise<void> {
  if (files.length >= maxFiles) {
    return;
  }

  const entries = await readdir(currentPath, { withFileTypes: true });
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    if (files.length >= maxFiles || shouldSkip(entry.name)) {
      continue;
    }

    const entryPath = path.join(currentPath, entry.name);
    if (entry.isDirectory()) {
      await collectEditableFiles(rootPath, entryPath, files, maxFiles);
      continue;
    }

    if (entry.isFile() && isEditableFile(entry.name)) {
      const fileStat = await stat(entryPath);
      if (fileStat.size <= 64_000) {
        files.push(entryPath);
      }
    }
  }
}

function shouldSkip(name: string): boolean {
  return name === ".git" || name === "node_modules" || name === "dist" || name === "coverage";
}

function isEditableFile(name: string): boolean {
  return [".ts", ".tsx", ".js", ".jsx", ".json", ".md"].includes(path.extname(name));
}

function truncate(content: string, maxLength: number): string {
  return content.length <= maxLength ? content : `${content.slice(0, maxLength)}\n[truncated]`;
}

function extractText(blocks: readonly ContentBlock[]): string {
  return blocks
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("\n")
    .trim();
}

function parseCodeChange(text: string): CodeChange {
  const jsonText = stripJsonFence(text);
  const parsed = parseJsonWithStringNewlineRepair(jsonText);
  const rawCodeChange = rawCodeChangeSchema.parse(parsed);

  return {
    summary: rawCodeChange.summary,
    changes: rawCodeChange.changes.map((change) => {
      const content = change.content ?? change.contentLines?.join("\n");
      if (content === undefined) {
        throw new Error(`Change for ${change.path} must include content or contentLines.`);
      }

      return {
        path: change.path,
        content: ensureTrailingNewline(content.replace(/\r\n/g, "\n"))
      };
    }),
    ...(rawCodeChange.verificationCommand === undefined ? {} : { verificationCommand: rawCodeChange.verificationCommand })
  };
}

function parseJsonWithStringNewlineRepair(jsonText: string): unknown {
  try {
    return JSON.parse(jsonText) as unknown;
  } catch {
    return JSON.parse(escapeRawNewlinesInsideStrings(jsonText)) as unknown;
  }
}

function escapeRawNewlinesInsideStrings(value: string): string {
  let inString = false;
  let escaped = false;
  let output = "";

  for (const char of value) {
    if (escaped) {
      output += char;
      escaped = false;
      continue;
    }

    if (char === "\\") {
      output += char;
      escaped = true;
      continue;
    }

    if (char === "\"") {
      inString = !inString;
      output += char;
      continue;
    }

    if (inString && char === "\n") {
      output += "\\n";
      continue;
    }

    if (inString && char === "\r") {
      output += "\\r";
      continue;
    }

    output += char;
  }

  return output;
}

function ensureTrailingNewline(content: string): string {
  return content.endsWith("\n") ? content : `${content}\n`;
}

function stripJsonFence(text: string): string {
  const trimmed = text.trim();
  const fenced = /```(?:json)?\s*([\s\S]*?)\s*```/i.exec(trimmed);
  return fenced?.[1] ?? trimmed;
}

async function applyCodeChange(
  workspacePath: string,
  changes: readonly { readonly path: string; readonly content: string }[],
  progress?: AgentRequest["progress"]
): Promise<readonly string[]> {
  const rootPath = path.resolve(workspacePath);
  const writtenFiles: string[] = [];

  for (const change of changes) {
    const targetPath = path.resolve(rootPath, change.path);
    if (!isInsidePath(rootPath, targetPath)) {
      throw new Error(`Refusing to write outside workspace: ${change.path}`);
    }

    await mkdir(path.dirname(targetPath), { recursive: true });
    await writeFile(targetPath, change.content, "utf8");
    writtenFiles.push(path.relative(rootPath, targetPath));
    await progress?.({
      stage: "code_change.file_written",
      message: `已写入文件：${path.relative(rootPath, targetPath)}`,
      data: { path: path.relative(rootPath, targetPath) }
    });
  }

  return writtenFiles;
}

function isInsidePath(parentPath: string, childPath: string): boolean {
  const relativePath = path.relative(parentPath, childPath);
  return relativePath.length === 0 || (!relativePath.startsWith("..") && !path.isAbsolute(relativePath));
}

async function runVerification(
  workspacePath: string,
  command: string
): Promise<{ readonly command: string; readonly ok: boolean; readonly output: string }> {
  const [executable, ...args] = verificationCommandParts(command);

  try {
    const result = await execFileAsync(executable, args, {
      cwd: workspacePath,
      timeout: 60_000
    });
    return {
      command,
      ok: true,
      output: [result.stdout, result.stderr].join("\n").trim()
    };
  } catch (error) {
    return {
      command,
      ok: false,
      output: error instanceof Error ? error.message : String(error)
    };
  }
}

function verificationCommandParts(command: string): readonly [string, ...string[]] {
  if (command.trim().length === 0) {
    throw new Error("Verification command cannot be empty.");
  }

  if (process.platform === "win32") {
    return ["cmd.exe", "/d", "/s", "/c", command];
  }

  const parts = command.match(/(?:[^\s"]+|"[^"]*")+/g)?.map((part) => part.replace(/^"|"$/g, "")) ?? [];
  if (parts.length === 0 || parts[0] === undefined) {
    throw new Error("Verification command cannot be empty.");
  }

  return [parts[0], ...parts.slice(1)];
}
