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
const MAX_DOCUMENT_BYTES = 256 * 1024;
const MAX_IMAGE_BYTES = 1024 * 1024;
const MAX_TOTAL_ATTACHMENT_BYTES = 2 * 1024 * 1024;

const DOCUMENT_EXTENSIONS = new Set([".txt", ".md", ".markdown"]);
const DOCUMENT_MIME_TYPES = new Set([
  "text/plain",
  "text/markdown",
  "text/x-markdown",
  "application/octet-stream"
]);
const IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp"]);
const IMAGE_MIME_TYPES = new Set(["image/png", "image/jpeg", "image/gif", "image/webp"]);

export async function saveDevAttachments(
  options: SaveDevAttachmentsOptions
): Promise<readonly InboundAttachment[]> {
  if (options.attachments.length > MAX_ATTACHMENT_COUNT) {
    throw new Error(`Upload at most ${MAX_ATTACHMENT_COUNT} attachments.`);
  }

  let totalBytes = 0;
  const saved: InboundAttachment[] = [];
  for (const [index, attachment] of options.attachments.entries()) {
    const decoded = decodeDevAttachment(attachment);
    totalBytes += decoded.content.length;
    if (totalBytes > MAX_TOTAL_ATTACHMENT_BYTES) {
      throw new Error(
        `Uploaded attachments must be ${MAX_TOTAL_ATTACHMENT_BYTES} bytes or less in total.`
      );
    }

    const fileName = `${index + 1}-${sanitizeFileName(decoded.name)}`;
    const storagePath = resolveStoragePath(options.uploadRootPath, options.messageId, fileName);
    await mkdir(path.dirname(storagePath), { recursive: true });
    await writeFile(storagePath, decoded.content);
    saved.push({
      type: decoded.type,
      name: decoded.name,
      mimeType: decoded.mimeType,
      storagePath,
      sizeBytes: decoded.content.length
    });
  }

  return saved;
}

function decodeDevAttachment(attachment: DevChatAttachmentPayload): {
  readonly type: "document" | "image";
  readonly name: string;
  readonly mimeType: string;
  readonly content: Buffer;
} {
  const name = normalizeName(attachment.name);
  const mimeType = normalizeMimeType(attachment.mimeType);
  const classification = classifyAttachment(name, mimeType);

  const contentBase64 = attachment.contentBase64?.trim();
  if (contentBase64 === undefined || contentBase64.length === 0) {
    throw new Error(`Uploaded attachment ${name} is missing content.`);
  }
  if (!isBase64(contentBase64)) {
    throw new Error(`Uploaded attachment ${name} has invalid content encoding.`);
  }

  const content = Buffer.from(contentBase64, "base64");
  if (content.length === 0) {
    throw new Error(`Uploaded attachment ${name} is empty.`);
  }
  const maxBytes = classification.type === "image" ? MAX_IMAGE_BYTES : MAX_DOCUMENT_BYTES;
  if (content.length > maxBytes) {
    throw new Error(`Uploaded ${classification.type} ${name} must be ${maxBytes} bytes or less.`);
  }
  if (attachment.sizeBytes !== undefined && attachment.sizeBytes !== content.length) {
    throw new Error(`Uploaded attachment ${name} size does not match its content.`);
  }

  return { type: classification.type, name, mimeType: classification.mimeType, content };
}

function normalizeName(name: string | undefined): string {
  if (name === undefined || name.trim().length === 0) {
    throw new Error("Uploaded attachment name is required.");
  }

  const fileName = name.trim().split(/[\\/]/).pop() ?? "";
  if (fileName.length === 0 || fileName === "." || fileName === "..") {
    throw new Error("Uploaded attachment name is invalid.");
  }
  return fileName;
}

function normalizeMimeType(mimeType: string | undefined): string {
  if (mimeType === undefined || mimeType.trim().length === 0) return "application/octet-stream";
  return mimeType.split(";")[0]?.trim().toLowerCase() ?? "application/octet-stream";
}

function classifyAttachment(
  name: string,
  mimeType: string
): { readonly type: "document" | "image"; readonly mimeType: string } {
  const extension = path.extname(name).toLowerCase();
  if (DOCUMENT_EXTENSIONS.has(extension)) {
    if (!DOCUMENT_MIME_TYPES.has(mimeType)) {
      throw new Error(`Uploaded document ${name} has unsupported media type ${mimeType}.`);
    }
    return { type: "document", mimeType };
  }

  if (IMAGE_EXTENSIONS.has(extension)) {
    const imageMimeType =
      mimeType === "application/octet-stream" ? inferImageMimeType(extension) : mimeType;
    if (imageMimeType === undefined || !IMAGE_MIME_TYPES.has(imageMimeType)) {
      throw new Error(`Uploaded image ${name} has unsupported media type ${mimeType}.`);
    }
    return { type: "image", mimeType: imageMimeType };
  }

  throw new Error(`Uploaded attachment ${name} must be a .txt, .md, or supported image file.`);
}

function inferImageMimeType(extension: string): string | undefined {
  switch (extension) {
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".gif":
      return "image/gif";
    case ".webp":
      return "image/webp";
    default:
      return undefined;
  }
}

function isBase64(value: string): boolean {
  return value.length % 4 === 0 && /^[A-Za-z0-9+/]+={0,2}$/.test(value);
}

function sanitizeFileName(name: string): string {
  const sanitized = name.replace(/[^A-Za-z0-9._-]/g, "_");
  return sanitized.length > 0 ? sanitized : "attachment";
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
