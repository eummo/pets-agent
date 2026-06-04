import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import type { InboundAttachment } from "../core/index.js";

export type WechatDownloadedAttachment = {
  readonly kind: "image" | "file";
  readonly name?: string;
  readonly content: Buffer;
};

export type SaveWechatAttachmentsOptions = {
  readonly uploadRootPath: string;
  readonly messageId: string;
  readonly attachments: readonly WechatDownloadedAttachment[];
};

const MAX_ATTACHMENT_COUNT = 4;
const MAX_DOCUMENT_BYTES = 256 * 1024;
const MAX_IMAGE_BYTES = 1024 * 1024;
const MAX_TOTAL_ATTACHMENT_BYTES = 2 * 1024 * 1024;

const DOCUMENT_EXTENSIONS = new Set([".txt", ".md", ".markdown"]);
const IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp"]);

export async function saveWechatAttachments(
  options: SaveWechatAttachmentsOptions
): Promise<readonly InboundAttachment[]> {
  if (options.attachments.length > MAX_ATTACHMENT_COUNT) {
    throw new Error(`Upload at most ${MAX_ATTACHMENT_COUNT} attachments.`);
  }

  let totalBytes = 0;
  const saved: InboundAttachment[] = [];
  for (const [index, attachment] of options.attachments.entries()) {
    const decoded = decodeWechatAttachment(attachment, index);
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

function decodeWechatAttachment(
  attachment: WechatDownloadedAttachment,
  index: number
): {
  readonly type: "document" | "image";
  readonly name: string;
  readonly mimeType: string;
  readonly content: Buffer;
} {
  if (attachment.content.length === 0) {
    throw new Error("Uploaded attachment is empty.");
  }

  const classification = classifyWechatAttachment(attachment, index);
  const maxBytes = classification.type === "image" ? MAX_IMAGE_BYTES : MAX_DOCUMENT_BYTES;
  if (attachment.content.length > maxBytes) {
    throw new Error(
      `Uploaded ${classification.type} ${classification.name} must be ${maxBytes} bytes or less.`
    );
  }

  return {
    ...classification,
    content: attachment.content
  };
}

function classifyWechatAttachment(
  attachment: WechatDownloadedAttachment,
  index: number
): { readonly type: "document" | "image"; readonly name: string; readonly mimeType: string } {
  const imageType = detectImageType(attachment.content);
  const name = normalizeName(
    attachment.name ??
      (imageType !== undefined ? `wechat-image-${index + 1}.${imageType.extension}` : undefined)
  );
  const extension = path.extname(name).toLowerCase();

  if (attachment.kind === "image" || IMAGE_EXTENSIONS.has(extension)) {
    const detected = imageType ?? imageTypeFromExtension(extension);
    if (detected === undefined) {
      throw new Error(`Uploaded image ${name} has unsupported media type.`);
    }
    return {
      type: "image",
      name: ensureExtension(name, detected.extension),
      mimeType: detected.mimeType
    };
  }

  if (DOCUMENT_EXTENSIONS.has(extension)) {
    return { type: "document", name, mimeType: documentMimeType(extension) };
  }

  throw new Error(`Uploaded attachment ${name} must be a .txt, .md, or supported image file.`);
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

function detectImageType(
  content: Buffer
): { readonly extension: string; readonly mimeType: string } | undefined {
  if (content.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) {
    return { extension: "png", mimeType: "image/png" };
  }
  if (content.subarray(0, 3).equals(Buffer.from([255, 216, 255]))) {
    return { extension: "jpg", mimeType: "image/jpeg" };
  }
  const header = content.subarray(0, 12).toString("ascii");
  if (header.startsWith("GIF87a") || header.startsWith("GIF89a")) {
    return { extension: "gif", mimeType: "image/gif" };
  }
  if (header.startsWith("RIFF") && header.slice(8, 12) === "WEBP") {
    return { extension: "webp", mimeType: "image/webp" };
  }
  return undefined;
}

function imageTypeFromExtension(
  extension: string
): { readonly extension: string; readonly mimeType: string } | undefined {
  switch (extension) {
    case ".png":
      return { extension: "png", mimeType: "image/png" };
    case ".jpg":
    case ".jpeg":
      return { extension: "jpg", mimeType: "image/jpeg" };
    case ".gif":
      return { extension: "gif", mimeType: "image/gif" };
    case ".webp":
      return { extension: "webp", mimeType: "image/webp" };
    default:
      return undefined;
  }
}

function documentMimeType(extension: string): string {
  switch (extension) {
    case ".md":
    case ".markdown":
      return "text/markdown";
    default:
      return "text/plain";
  }
}

function ensureExtension(name: string, extension: string): string {
  return path.extname(name).length > 0 ? name : `${name}.${extension}`;
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
