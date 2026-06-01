import type {
  AuthorizationAction,
  AuthorizationDecision,
  AuthorizationService
} from "../auth/index.js";
import type { UserIntent } from "../intent/index.js";
import type { KnowledgeWorkspace } from "../workspace/index.js";
import type { AgentConversationMessage, InboundMessage, UserRole } from "./index.js";
import { actionForIntent, responseForDeniedIntent } from "./intentAuthorization.js";

export type IntentDetector = (
  userMessage: string,
  role: UserRole,
  history?: readonly AgentConversationMessage[]
) => Promise<UserIntent>;

export type RequestAuthorizationGateDependencies = {
  readonly authorization: AuthorizationService;
  readonly detectIntent: IntentDetector;
};

export type RequestAuthorizationGateInput = {
  readonly message: InboundMessage;
  readonly workspace: KnowledgeWorkspace;
  readonly history?: readonly AgentConversationMessage[];
};

export type AuthorizedRequest = {
  readonly status: "allowed";
  readonly role: UserRole;
  readonly intent: UserIntent;
  readonly requiredAction?: AuthorizationAction;
};

export type DeniedRequest =
  | {
      readonly status: "denied";
      readonly deniedAt: "read";
      readonly role: UserRole;
      readonly decision: AuthorizationDecision;
      readonly responseText: string;
    }
  | {
      readonly status: "denied";
      readonly deniedAt: "intent";
      readonly role: UserRole;
      readonly intent: UserIntent;
      readonly requiredAction: AuthorizationAction;
      readonly decision: AuthorizationDecision;
      readonly responseText: string;
    };

export type RequestAuthorizationGateResult = AuthorizedRequest | DeniedRequest;

export class RequestAuthorizationGate {
  public constructor(private readonly dependencies: RequestAuthorizationGateDependencies) {}

  public async evaluate(
    input: RequestAuthorizationGateInput
  ): Promise<RequestAuthorizationGateResult> {
    const role = await this.resolveRole(input.message);
    const readDecision = await this.canAction(input.message, role, "read", input.workspace);
    if (!readDecision.allowed) {
      return {
        status: "denied",
        deniedAt: "read",
        role,
        decision: readDecision,
        responseText: readDecision.reason ?? "You do not have permission to access this workspace."
      };
    }

    const intent = await this.dependencies.detectIntent(input.message.text, role, input.history);
    const requiredAction = actionForIntent(intent);
    if (requiredAction === undefined) {
      return {
        status: "allowed",
        role,
        intent
      };
    }

    const intentDecision = await this.canAction(
      input.message,
      role,
      requiredAction,
      input.workspace
    );
    if (intentDecision.allowed) {
      return {
        status: "allowed",
        role,
        intent,
        requiredAction
      };
    }

    return {
      status: "denied",
      deniedAt: "intent",
      role,
      intent,
      requiredAction,
      decision: intentDecision,
      responseText: responseForDeniedIntent(intent)
    };
  }

  private async resolveRole(message: InboundMessage): Promise<UserRole> {
    if (message.roleOverride !== undefined) {
      return message.roleOverride;
    }
    return this.dependencies.authorization.roleFor(message.user);
  }

  private canAction(
    message: InboundMessage,
    role: UserRole,
    action: AuthorizationAction,
    workspace: KnowledgeWorkspace
  ): Promise<AuthorizationDecision> {
    if (
      message.roleOverride !== undefined &&
      this.dependencies.authorization.canRole !== undefined
    ) {
      return this.dependencies.authorization.canRole(role, action, workspace);
    }
    return this.dependencies.authorization.can(message.user, action, workspace);
  }
}
