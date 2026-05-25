export function formatSafeRuntimeError(error: unknown): string {
  if (!(error instanceof Error)) {
    return "Model call failed due to an unexpected error. Please try again later.";
  }

  if (error.message.includes("invalid api key")) {
    return "Model call failed: API key is invalid or not configured. Contact an administrator.";
  }

  return "Model call failed. The service encountered an error processing your request. Please try again later.";
}

export function formatInternalError(error: unknown): string {
  if (!(error instanceof Error)) {
    return String(error);
  }

  return error.message.split("\n")[0] ?? "Unknown error.";
}
