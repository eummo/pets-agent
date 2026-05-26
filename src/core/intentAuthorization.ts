import type { AuthorizationAction, UserIntent } from "./contracts.js";

export function actionForIntent(intent: UserIntent): AuthorizationAction | undefined {
  if (intent.type === "mutate") {
    return "mutate";
  }
  if (intent.type === "update_kb") {
    return "update_kb";
  }
  return undefined;
}

export function responseForDeniedIntent(intent: UserIntent): string {
  if (intent.type === "update_kb") {
    return "感谢您的反馈！我已记录您希望更新知识库的请求。当前文档助手权限仅支持查看知识库，不支持修改内容。您的请求已保存，管理员将尽快审核处理。";
  }

  return "我已识别到这是修改请求，但你当前是文档助手权限，只能查看知识库，不能修改文件。您的请求已记录，管理员将尽快审核处理。";
}
