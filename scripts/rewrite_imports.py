#!/usr/bin/env python3
"""Rewrite import paths after contracts.ts modularization."""

import re
from pathlib import Path
import sys

REWRITES = [
    # agent/ imports
    ("src/agent/claudeSdkAgentRuntime.ts", [
        'import type { AgentRequest, AgentResponse, AgentRuntime, ContextUsageReport, StoredRoleConfig } from "../core/contracts.js";',
        'import type { AgentRequest, AgentResponse, AgentRuntime, ContextUsageReport } from "./contracts.js";\nimport type { StoredRoleConfig } from "../auth/contracts.js";',
    ]),
    ("src/agent/claudeSdkMessageMapper.ts", [
        'import type { AgentRequest, AgentStreamEvent, StoredRoleConfig } from "../core/contracts.js";',
        'import type { AgentRequest, AgentStreamEvent } from "./contracts.js";\nimport type { StoredRoleConfig } from "../auth/contracts.js";',
    ]),
    ("src/agent/codebuddySdkAgentRuntime.ts", [
        'import type { AgentRequest, AgentResponse, AgentRuntime, ContextUsageReport, StoredRoleConfig } from "../core/contracts.js";',
        'import type { AgentRequest, AgentResponse, AgentRuntime, ContextUsageReport } from "./contracts.js";\nimport type { StoredRoleConfig } from "../auth/contracts.js";',
    ]),
    ("src/agent/createAgentRuntimes.ts", [
        'import type { AgentRuntime, AgentRuntimeFactory, RoleConfigStore, StoredRoleConfig } from "../core/contracts.js";',
        'import type { AgentRuntime, AgentRuntimeFactory } from "./contracts.js";\nimport type { RoleConfigStore, StoredRoleConfig } from "../auth/contracts.js";',
    ]),
    ("src/agent/intentAgentRuntime.ts", [
        'import type { AgentRequest, AgentResponse, AgentRuntime, UserIntent } from "../core/contracts.js";',
        'import type { AgentRequest, AgentResponse, AgentRuntime } from "./contracts.js";\nimport type { UserIntent } from "../intent/contracts.js";',
    ]),
    ("src/agent/llmBashPermissionDecider.ts", [
        'import type { StoredRoleConfig } from "../core/contracts.js";',
        'import type { StoredRoleConfig } from "../auth/contracts.js";',
    ]),
    ("src/agent/piAgentRuntime.ts", [
        'import type { AgentRequest, AgentResponse, AgentRuntime, StoredRoleConfig } from "../core/contracts.js";',
        'import type { AgentRequest, AgentResponse, AgentRuntime } from "./contracts.js";\nimport type { StoredRoleConfig } from "../auth/contracts.js";',
    ]),
    ("src/agent/piEventCollector.ts", [
        'import type { AgentRequest, AgentResponse, ContextUsageReport, StoredRoleConfig } from "../core/contracts.js";',
        'import type { AgentRequest, AgentResponse, ContextUsageReport } from "./contracts.js";\nimport type { StoredRoleConfig } from "../auth/contracts.js";',
    ]),
    ("src/agent/toolPolicy.ts", [
        'import type { StoredRoleConfig } from "../core/contracts.js";',
        'import type { StoredRoleConfig } from "../auth/contracts.js";',
    ]),
    ("src/agent/toolPolicy.ts", [
        'import { FILE_MUTATION_TOOLS } from "../core/contracts.js";',
        'import { FILE_MUTATION_TOOLS } from "../auth/contracts.js";',
    ]),
    ("src/agent/workspacePromptBuilder.ts", [
        'import type { AgentRequest } from "../core/contracts.js";',
        'import type { AgentRequest } from "./contracts.js";',
    ]),
    # agent test files
    ("src/agent/claudeSdkAgentRuntime.test.ts", [
        'import type { AgentStreamEvent, StoredRoleConfig } from "../core/contracts.js";',
        'import type { AgentStreamEvent } from "./contracts.js";\nimport type { StoredRoleConfig } from "../auth/contracts.js";',
    ]),
    ("src/agent/claudeToolPolicy.test.ts", [
        'import type { StoredRoleConfig } from "../core/contracts.js";',
        'import type { StoredRoleConfig } from "../auth/contracts.js";',
    ]),
    ("src/agent/codebuddySdkAgentRuntime.test.ts", [
        'import type { AgentStreamEvent, StoredRoleConfig } from "../core/contracts.js";',
        'import type { AgentStreamEvent } from "./contracts.js";\nimport type { StoredRoleConfig } from "../auth/contracts.js";',
    ]),
    ("src/agent/intentAgentRuntime.test.ts", [
        'import type { AgentRequest } from "../core/contracts.js";',
        'import type { AgentRequest } from "./contracts.js";',
    ]),
    ("src/agent/llmBashPermissionDecider.test.ts", [
        'import type { StoredRoleConfig } from "../core/contracts.js";',
        'import type { StoredRoleConfig } from "../auth/contracts.js";',
    ]),
    ("src/agent/piAgentRuntime.test.ts", [
        'import type { AgentStreamEvent, StoredRoleConfig } from "../core/contracts.js";',
        'import type { AgentStreamEvent } from "./contracts.js";\nimport type { StoredRoleConfig } from "../auth/contracts.js";',
    ]),
    # auth/
    ("src/auth/inMemoryRoleAuthorizationService.ts", [
        'import type {\n  AuthorizationAction,\n  AuthorizationDecision,\n  AuthorizationService,\n  ChannelUser,\n  KnowledgeWorkspace,\n  RoleCapability,\n  RoleConfigStore,\n  StoredRoleConfig,\n  UserRole\n} from "../core/contracts.js";',
        'import type { AuthorizationAction, AuthorizationDecision, AuthorizationService } from "./contracts.js";\nimport type { ChannelUser, KnowledgeWorkspace } from "../core/contracts.js";\nimport type { RoleCapability, RoleConfigStore, StoredRoleConfig } from "./contracts.js";',
    ]),
    ("src/auth/inMemoryRoleAuthorizationService.ts", [
        'import { FILE_MUTATION_TOOLS } from "../core/contracts.js";',
        'import { FILE_MUTATION_TOOLS } from "./contracts.js";',
    ]),
    ("src/auth/inMemoryRoleAuthorizationService.test.ts", [
        'import type { RoleCapability, RoleConfigStore } from "../core/contracts.js";',
        'import type { RoleCapability, RoleConfigStore } from "./contracts.js";',
    ]),
    # core/
    ("src/core/runtimeCache.ts", [
        'import type { AgentRuntime, AgentRuntimeFactory } from "./contracts.js";',
        'import type { AgentRuntime, AgentRuntimeFactory } from "../agent/contracts.js";',
    ]),
    ("src/core/streamProgressMapper.ts", [
        'import type { AgentProgressEvent, AgentStreamEvent } from "./contracts.js";',
        'import type { AgentProgressEvent, AgentStreamEvent } from "../agent/contracts.js";',
    ]),
    ("src/core/intentAuthorization.ts", [
        'import type { AuthorizationAction, UserIntent } from "./contracts.js";',
        'import type { AuthorizationAction } from "../auth/contracts.js";\nimport type { UserIntent } from "../intent/contracts.js";',
    ]),
    ("src/core/intentHeuristics.ts", [
        'import type { UserIntent } from "./contracts.js";',
        'import type { UserIntent } from "../intent/contracts.js";',
    ]),
    ("src/core/defaultRoles.ts", [
        'import type { StoredRoleConfig } from "./contracts.js";',
        'import type { StoredRoleConfig } from "../auth/contracts.js";',
    ]),
    # intent/
    ("src/intent/llmIntentDetectionService.ts", [
        'import type { AgentConversationMessage, UserIntent, UserRole } from "../core/contracts.js";',
        'import type { AgentConversationMessage } from "../core/contracts.js";\nimport type { UserIntent } from "./contracts.js";\nimport type { UserRole } from "../core/contracts.js";',
    ]),
    ("src/intent/llmIntentDetectionService.test.ts", [
        'import type { UserIntent } from "../core/contracts.js";',
        'import type { UserIntent } from "./contracts.js";',
    ]),
    # persistence/
    ("src/persistence/fileConversationHistoryStore.ts", [
        'import type { AgentConversationMessage, ConversationHistoryStore, ConversationSessionKey } from "../core/contracts.js";',
        'import type { AgentConversationMessage, ConversationHistoryStore, ConversationSessionKey } from "./contracts.js";',
    ]),
    ("src/persistence/fileConversationSessionStore.ts", [
        'import type { ConversationSessionKey, ConversationSessionStore } from "../core/contracts.js";',
        'import type { ConversationSessionKey, ConversationSessionStore } from "./contracts.js";',
    ]),
    ("src/persistence/fileStoreUtils.ts", [
        'import type { ConversationSessionKey } from "../core/contracts.js";',
        'import type { ConversationSessionKey } from "./contracts.js";',
    ]),
    ("src/persistence/seedRoles.ts", [
        'import type { RoleConfigStore, StoredRoleConfig } from "../core/contracts.js";',
        'import type { RoleConfigStore, StoredRoleConfig } from "./contracts.js";',
    ]),
    ("src/persistence/sqliteFeedbackStore.ts", [
        'import type { FeedbackEntry, FeedbackQuery, FeedbackStore, FeedbackStatus, UserIntent } from "../core/contracts.js";',
        'import type { FeedbackEntry, FeedbackQuery, FeedbackStore, FeedbackStatus } from "./contracts.js";\nimport type { UserIntent } from "../intent/contracts.js";',
    ]),
    ("src/persistence/sqliteRoleConfigStore.ts", [
        'import type { RoleConfigStore, StoredRoleConfig } from "../core/contracts.js";',
        'import type { RoleConfigStore, StoredRoleConfig } from "./contracts.js";',
    ]),
    # server/
    ("src/server/devChatRoutes.ts", [
        'import type { AgentStreamEvent } from "../core/contracts.js";',
        'import type { AgentStreamEvent } from "../agent/contracts.js";',
    ]),
    ("src/server/sseProgressBroker.ts", [
        'import type { AgentProgressEvent, ChannelUser, ProgressReporter } from "../core/contracts.js";',
        'import type { ChannelUser } from "../core/contracts.js";\nimport type { AgentProgressEvent } from "../agent/contracts.js";',
    ]),
    # wechat/
    ("src/wechat/wechatSmartBotAdapter.ts", [
        'import type { ConversationLogger, InboundMessage, MessageGateway, AgentStreamPublisher } from "../core/contracts.js";',
        'import type { ConversationLogger, InboundMessage, MessageGateway, AgentStreamPublisher } from "../core/contracts.js";',
    ]),
    # workspace/
    ("src/workspace/configuredWorkspaceResolver.ts", [
        'import type {\n  InboundMessage,\n  KnowledgeWorkspace,\n  KnowledgeWorkspaceResolver\n} from "../core/contracts.js";',
        'import type { InboundMessage } from "../core/contracts.js";\nimport type { KnowledgeWorkspace, KnowledgeWorkspaceResolver } from "./contracts.js";',
    ]),
    # createServer.test.ts
    ("src/server/createServer.test.ts", [
        'import type { AuthorizationService, FeedbackEntry, MessageGateway, OutboundMessage, RoleCapability, UserRole, ChannelUser, AuthorizationAction, AuthorizationDecision } from "../core/contracts.js";',
        'import type { ChannelUser, MessageGateway, OutboundMessage } from "../core/contracts.js";\nimport type { AuthorizationService, AuthorizationAction, AuthorizationDecision } from "../auth/contracts.js";\nimport type { FeedbackEntry } from "../persistence/contracts.js";\nimport type { RoleCapability } from "../auth/contracts.js";',
    ]),
    # sqliteFeedbackStore.test.ts
    ("src/persistence/sqliteFeedbackStore.test.ts", [
        'import type { FeedbackEntry } from "../core/contracts.js";',
        'import type { FeedbackEntry } from "./contracts.js";',
    ]),
    # sqliteRoleConfigStore.test.ts
    ("src/persistence/sqliteRoleConfigStore.test.ts", [
        'import type { StoredRoleConfig } from "../core/contracts.js";',
        'import type { StoredRoleConfig } from "./contracts.js";',
    ]),
    # wechat test
    ("src/wechat/wechatSmartBotAdapter.test.ts", [
        'import type { ConversationLogger, MessageGateway, OutboundMessage } from "../core/contracts.js";',
        'import type { ConversationLogger, MessageGateway, OutboundMessage } from "../core/contracts.js";',
    ]),
]


def do_rewrite(path: Path, old: str, new: str) -> bool:
    content = path.read_text(encoding="utf-8")
    if old not in content:
        return False
    path.write_text(content.replace(old, new, 1), encoding="utf-8")
    return True


def main():
    ok = 0
    fail = 0
    for file_rel, changes in REWRITES:
        p = Path(file_rel)
        if not p.exists():
            print(f"SKIP (not found): {file_rel}")
            continue
        for old, new in changes:
            if do_rewrite(p, old, new):
                print(f"  OK: {p} [{old[:40]}...]")
            else:
                print(f"FAIL: {p} [{old[:40]}...] -- pattern not found")
                fail += 1
        ok += 1
    print(f"\n{ok} files touched, {fail} failures")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())