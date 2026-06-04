import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import type { InboundAttachment } from "../core/index.js";

export type DevChatAttachmentPayload = {
  readonly name?: string;
  readonly mimeType?: string;
  readonly contentBase64?: string;
  readonly sizeBytes?: number;
};

export type SaveDevAttachmentsOptions = {
  readonly uploadRootPath: string;
  readonly messageId: string;
  readonly attachments: readonly DevChatAttachmentPayload[];
};

const MAX_ATTACHMENT_COUNT = 4;
const MAX_ATTACHMENT_BYTES = 256 * 1024;
const MAX_TOTAL_ATTACHMENT_BYTES = 512 * 1024;

const SUPPORTED_EXTENSIONS = new Set([".txt", ".md", ".markdown"]);
const SUPPORTED_MIME_TYPES = new Set([
  "text/plain",
  "text/markdown",
  "text/x-markdown",
  "application/octet-stream"
]);

export async function saveDevAttachments(
  options: SaveDevAttachmentsOptions
): Promise<readonly InboundAttachment[]> {
  if (options.attachments.length > MAX_ATTACHMENT_COUNT) {
    throw new Error(`Upload at most ${MAX_ATTACHMENT_COUNT} documents.`);
  }

  let totalBytes = 0;
  const saved: InboundAttachment[] = [];
  for (const [index, attachment] of options.attachments.entries()) {
    const decoded = decodeDevAttachment(attachment);
    totalBytes += decoded.content.length;
    if (totalBytes > MAX_TOTAL_ATTACHMENT_BYTES) {
      throw new Error(
        `Uploaded documents must be ${MAX_TOTAL_ATTACHMENT_BYTES} bytes or less in total.`
      );
    }

    const fileName = `${index + 1}-${sanitizeFileName(decoded.name)}`;
    const storagePath = resolveStoragePath(options.uploadRootPath, options.messageId, fileName);
    await mkdir(path.dirname(storagePath), { recursive: true });
    await writeFile(storagePath, decoded.content);
    saved.push({
      type: "document",
      name: decoded.name,
      mimeType: decoded.mimeType,
      storagePath,
      sizeBytes: decoded.content.length
    });
  }

  return saved;
}

function decodeDevAttachment(attachment: DevChatAttachmentPayload): {
  readonly name: string;
  readonly mimeType: string;
  readonly content: Buffer;
} {
  const name = normalizeName(attachment.name);
  const mimeType = normalizeMimeType(attachment.mimeType);
  validateDocumentType(name, mimeType);

  const contentBase64 = attachment.contentBase64?.trim();
  if (contentBase64 === undefined || contentBase64.length === 0) {
    throw new Error(`Uploaded document ${name} is missing content.`);
  }
  if (!isBase64(contentBase64)) {
    throw new Error(`Uploaded document ${name} has invalid content encoding.`);
  }

  const content = Buffer.from(contentBase64, "base64");
  if (content.length === 0) {
    throw new Error(`Uploaded document ${name} is empty.`);
  }
  if (content.length > MAX_ATTACHMENT_BYTES) {
    throw new Error(`Uploaded document ${name} must be ${MAX_ATTACHMENT_BYTES} bytes or less.`);
  }
  if (attachment.sizeBytes !== undefined && attachment.sizeBytes !== content.length) {
    throw new Error(`Uploaded document ${name} size does not match its content.`);
  }

  return { name, mimeType, content };
}

function normalizeName(name: string | undefined): string {
  if (name === undefined || name.trim().length === 0) {
    throw new Error("Uploaded document name is required.");
  }

  const fileName = name.trim().split(/[\\/]/).pop() ?? "";
  if (fileName.length === 0 || fileName === "." || fileName === "..") {
    throw new Error("Uploaded document name is invalid.");
  }
  return fileName;
}

function normalizeMimeType(mimeType: string | undefined): string {
  if (mimeType === undefined || mimeType.trim().length === 0) return "application/octet-stream";
  return mimeType.split(";")[0]?.trim().toLowerCase() ?? "application/octet-stream";
}

function validateDocumentType(name: string, mimeType: string): void {
  const extension = path.extname(name).toLowerCase();
  if (!SUPPORTED_EXTENSIONS.has(extension)) {
    throw new Error(`Uploaded document ${name} must be a .txt or .md file.`);
  }
  if (!SUPPORTED_MIME_TYPES.has(mimeType)) {
    throw new Error(`Uploaded document ${name} has unsupported media type ${mimeType}.`);
  }
}

function isBase64(value: string): boolean {
  return value.length % 4 === 0 && /^[A-Za-z0-9+/]+={0,2}$/.test(value);
}

function sanitizeFileName(name: string): string {
  const sanitized = name.replace(/[^A-Za-z0-9._-]/g, "_");
  return sanitized.length > 0 ? sanitized : "document.txt";
}

function resolveStoragePath(uploadRootPath: string, messageId: string, fileName: string): string {
  const rootPath = path.resolve(uploadRootPath);
  const messagePath = path.resolve(rootPath, sanitizeFileName(messageId));
  const storagePath = path.resolve(messagePath, fileName);
  if (isPathOutsideDirectory(storagePath, rootPath)) {
    throw new Error("Resolved upload path is outside the upload directory.");
  }
  return storagePath;
}

function isPathOutsideDirectory(filePath: string, directoryPath: string): boolean {
  const relativePath = path.relative(path.resolve(directoryPath), path.resolve(filePath));
  return relativePath.startsWith("..") || path.isAbsolute(relativePath);
}
