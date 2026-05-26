const form = document.querySelector("#chat-form");
const messages = document.querySelector("#messages");
const userIdEl = document.querySelector("#user-id");
const roleSelect = document.querySelector("#role-select");
const input = document.querySelector("#message-input");
const button = document.querySelector("#send-button");
const feedbackList = document.querySelector("#feedback-list");
const refreshFeedback = document.querySelector("#refresh-feedback");
const feedbackPageSize = 20;
let feedbackOffset = 0;

// ── Tab switching ──
const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    panels.forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    const panelId = tab.dataset.tab === "chat" ? "chat-panel" : "feedback-panel";
    document.getElementById(panelId).classList.add("active");
    if (tab.dataset.tab === "feedback") loadFeedback();
  });
});

// ── SSE progress channel ──
let eventsSource;
function connectEvents() {
  if (eventsSource) eventsSource.close();
  eventsSource = new EventSource("/dev/events?userId=" + encodeURIComponent(userIdEl.value));
  eventsSource.addEventListener("progress", (event) => {
    const payload = JSON.parse(event.data);
    if (payload.stage?.startsWith("agent.")) return;
    const details = payload.data ? "\n" + JSON.stringify(payload.data, null, 2) : "";
    addSystemMessage("[" + payload.stage + "] " + payload.message + details);
  });
  eventsSource.onerror = () => {};
}
userIdEl.addEventListener("change", connectEvents);
connectEvents();

// ── Helpers ──
function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = "message " + role;
  const meta = document.createElement("span");
  meta.className = "meta";
  meta.textContent = role;
  el.append(meta, document.createTextNode(text));
  messages.append(el);
  el.scrollIntoView({ block: "end", behavior: "smooth" });
  return el;
}

function addSystemMessage(text) {
  const el = document.createElement("div");
  el.className = "message system";
  const meta = document.createElement("span");
  meta.className = "meta";
  meta.textContent = "system";
  el.append(meta, document.createTextNode(text));
  messages.append(el);
  el.scrollIntoView({ block: "end", behavior: "smooth" });
}

function createAgentMessage() {
  const el = document.createElement("div");
  el.className = "message agent";
  const meta = document.createElement("span");
  meta.className = "meta";
  meta.textContent = "agent";
  const content = document.createElement("div");
  content.className = "agent-content";
  el.append(meta, content);
  messages.append(el);
  return { el, content };
}

function createToolCallCard(toolName, inputObj, toolUseId) {
  const card = document.createElement("div");
  card.className = "tool-call";
  card.dataset.toolUseId = toolUseId;

  const header = document.createElement("div");
  header.className = "tool-call-header";
  const icon = document.createElement("span");
  icon.className = "tool-call-icon";
  icon.textContent = "🔧";
  const name = document.createElement("span");
  name.className = "tool-call-name";
  name.textContent = toolName;
  const summary = document.createElement("span");
  summary.className = "tool-call-summary";
  summary.textContent = " " + JSON.stringify(inputObj).slice(0, 80);
  const toggle = document.createElement("span");
  toggle.className = "tool-call-toggle";
  toggle.textContent = "▶";
  header.append(icon, name, summary, toggle);

  const body = document.createElement("div");
  body.className = "tool-call-body";
  const pre = document.createElement("pre");
  pre.textContent = "Input: " + JSON.stringify(inputObj, null, 2);
  body.append(pre);

  header.addEventListener("click", () => {
    body.classList.toggle("open");
    toggle.textContent = body.classList.contains("open") ? "▼" : "▶";
  });

  card.append(header, body);
  return card;
}

function updateToolCallCard(toolUseId, result, isError) {
  const card = document.querySelector('[data-tool-use-id="' + toolUseId + '"]');
  if (!card) return;
  const body = card.querySelector(".tool-call-body");
  if (!body) return;
  const pre = document.createElement("pre");
  pre.textContent = (isError ? "Error: " : "Result: ") + (result || "").slice(0, 2000);
  body.append(pre);
  body.classList.add("open");
  const toggle = card.querySelector(".tool-call-toggle");
  if (toggle) toggle.textContent = "▼";
}

// ── Capabilities ──
// Must stay in sync with RoleCapability in src/core/contracts.ts
const CAP = Object.freeze({
  WORKSPACE_READ: "workspace_read",
  WORKSPACE_MUTATE: "workspace_mutate",
  KNOWLEDGE_BASE_UPDATE: "knowledge_base_update",
  FEEDBACK_VIEW: "feedback_view",
  FEEDBACK_MANAGE: "feedback_manage",
});

function hasCapability(cap) {
  return currentCapabilities.includes(cap);
}

// ── Role ──
let currentCapabilities = [];

async function loadRoles() {
  try {
    const response = await fetch("/dev/roles");
    const data = await response.json();
    const roles = data.roles || [];
    if (roles.length === 0) {
      addFallbackRoles();
      return;
    }
    const prevRole = roleSelect.value;
    roleSelect.innerHTML = "";
    for (const role of roles) {
      const option = document.createElement("option");
      option.value = role.name;
      option.textContent = role.name;
      roleSelect.append(option);
    }
    // Preserve current selection; default to "reviewer" if available
    if (roleSelect.querySelector('option[value="' + prevRole + '"]')) {
      roleSelect.value = prevRole;
    } else if (roleSelect.querySelector('option[value="reviewer"]')) {
      roleSelect.value = "reviewer";
    }
    // Store capabilities for selected role
    updateCapabilitiesFromRoles(roles);
  } catch (_error) {
    addFallbackRoles();
  }
  updateFeedbackTabVisibility();
}

function updateCapabilitiesFromRoles(roles) {
  const selected = roles.find((r) => r.name === roleSelect.value);
  currentCapabilities = selected?.capabilities || [];
  updateFeedbackTabVisibility();
}

function addFallbackRoles() {
  roleSelect.innerHTML = "";
  const reviewer = document.createElement("option");
  reviewer.value = "reviewer";
  reviewer.textContent = "reviewer";
  reviewer.selected = true;
  const developer = document.createElement("option");
  developer.value = "developer";
  developer.textContent = "developer";
  const admin = document.createElement("option");
  admin.value = "admin";
  admin.textContent = "admin";
  roleSelect.append(reviewer, developer, admin);
  currentCapabilities = [CAP.WORKSPACE_READ];
}

async function setRole() {
  const response = await fetch("/dev/role", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ userId: userIdEl.value, role: roleSelect.value })
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Failed to set role");
  return body;
}

roleSelect.addEventListener("change", async () => {
  try {
    const body = await setRole();
    addSystemMessage("角色已切换为 " + body.role);
    // Reload roles to update capabilities
    await loadRoles();
  } catch (error) {
    addMessage("error", error instanceof Error ? error.message : String(error));
  }
});

loadRoles();

// ── Feedback management ──
function updateFeedbackTabVisibility() {
  const feedbackTab = document.querySelector('.tab[data-tab="feedback"]');
  const feedbackPanel = document.getElementById("feedback-panel");
  const canView = hasCapability(CAP.FEEDBACK_VIEW);
  if (feedbackTab) {
    feedbackTab.style.display = canView ? "" : "none";
  }
  // Switch back to chat panel if feedback panel is active but no longer visible
  if (!canView && feedbackPanel?.classList.contains("active")) {
    feedbackPanel.classList.remove("active");
    document.getElementById("chat-panel").classList.add("active");
    tabs.forEach((t) => t.classList.remove("active"));
    const chatTab = document.querySelector('.tab[data-tab="chat"]');
    if (chatTab) chatTab.classList.add("active");
  }
}

async function loadFeedback() {
  if (!feedbackList) return;
  feedbackOffset = 0;
  setFeedbackMessage("feedback-loading", "加载中...");
  try {
    await fetchFeedbackPage(false);
  } catch (error) {
    setFeedbackMessage("feedback-error", "加载失败: " + (error instanceof Error ? error.message : String(error)));
  }
}

async function fetchFeedbackPage(append) {
  const response = await fetch(
    "/dev/feedback?userId=" + encodeURIComponent(userIdEl.value)
      + "&limit=" + feedbackPageSize
      + "&offset=" + feedbackOffset
  );
    if (!response.ok) {
      const body = await response.json();
    setFeedbackMessage("feedback-error", body.error || "无法加载反馈");
      return;
    }
    const data = await response.json();
    const entries = data.feedback || [];
  if (entries.length === 0 && !append) {
    setFeedbackMessage("feedback-empty", "暂无反馈记录");
      return;
    }
  removeLoadMoreButton();
  if (!append) {
    feedbackList.replaceChildren();
  }
  for (const entry of entries) {
    feedbackList.append(createFeedbackCard(entry));
  }
  feedbackOffset += entries.length;
  if (entries.length === feedbackPageSize) {
    feedbackList.append(createLoadMoreButton());
  }
}

function setFeedbackMessage(className, text) {
  const el = document.createElement("div");
  el.className = className;
  el.textContent = text;
  feedbackList.replaceChildren(el);
}

function createLoadMoreButton() {
  const button = document.createElement("button");
  button.className = "feedback-btn";
  button.type = "button";
  button.dataset.feedbackLoadMore = "true";
  button.textContent = "加载更多";
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await fetchFeedbackPage(true);
    } catch (error) {
      addSystemMessage("加载更多反馈失败: " + (error instanceof Error ? error.message : String(error)));
      button.disabled = false;
    }
  });
  return button;
}

function removeLoadMoreButton() {
  const existing = feedbackList.querySelector('[data-feedback-load-more="true"]');
  existing?.remove();
}

function parseConversationContext(text) {
  const messages = [];
  const lines = text.split("\n");
  let currentRole = null;
  let currentContent = [];

  for (const line of lines) {
    if (line.startsWith("user: ") || line.startsWith("assistant: ")) {
      if (currentRole !== null) {
        messages.push({ role: currentRole, content: currentContent.join("\n") });
      }
      currentRole = line.startsWith("user:") ? "user" : "assistant";
      currentContent = [line.slice(line.indexOf(": ") + 2)];
    } else if (currentRole !== null) {
      currentContent.push(line);
    }
  }
  if (currentRole !== null) {
    messages.push({ role: currentRole, content: currentContent.join("\n") });
  }

  return messages;
}

function createFeedbackCard(entry) {
  const card = document.createElement("div");
  card.className = "feedback-card";
  card.dataset.id = entry.id;

  const header = document.createElement("div");
  header.className = "feedback-card-header";

  const statusBadge = document.createElement("span");
  statusBadge.className = "feedback-status status-" + entry.status;
  statusBadge.textContent = statusLabel(entry.status);
  header.append(statusBadge);

  const intentBadge = document.createElement("span");
  intentBadge.className = "feedback-intent";
  intentBadge.textContent = entry.intentType || "unknown";
  header.append(intentBadge);

  const timeEl = document.createElement("span");
  timeEl.className = "feedback-time";
  timeEl.textContent = entry.createdAt || "";
  header.append(timeEl);

  card.append(header);

  const userEl = document.createElement("div");
  userEl.className = "feedback-user";
  userEl.textContent = (entry.roleName || entry.userId) + ": " + entry.userMessage;
  card.append(userEl);

  if (entry.conversationContext) {
    const contextEl = document.createElement("details");
    contextEl.className = "feedback-context";
    const summary = document.createElement("summary");
    const contextLines = entry.conversationContext.split("\n");
    const messageCount = contextLines.filter((line) => line.startsWith("user:") || line.startsWith("assistant:")).length;
    summary.textContent = "完整对话上下文 (" + messageCount + " 条消息)";
    contextEl.append(summary);

    // Parse conversation into structured messages for better readability
    const messages = parseConversationContext(entry.conversationContext);
    const container = document.createElement("div");
    container.className = "feedback-context-messages";
    for (const msg of messages) {
      const msgEl = document.createElement("div");
      msgEl.className = "feedback-context-msg feedback-context-msg-" + msg.role;
      const roleLabel = document.createElement("span");
      roleLabel.className = "feedback-context-role";
      roleLabel.textContent = msg.role === "user" ? "用户" : "助手";
      const contentEl = document.createElement("div");
      contentEl.className = "feedback-context-content";
      contentEl.textContent = msg.content;
      msgEl.append(roleLabel, contentEl);
      container.append(msgEl);
    }
    contextEl.append(container);
    card.append(contextEl);
  }

  // Action buttons for admin (feedback_manage capability)
  if (hasCapability(CAP.FEEDBACK_MANAGE) && entry.status === "pending") {
    const actions = document.createElement("div");
    actions.className = "feedback-actions";

    const reviewBtn = document.createElement("button");
    reviewBtn.className = "feedback-btn review-btn";
    reviewBtn.textContent = "标记已审阅";
    reviewBtn.type = "button";
    reviewBtn.addEventListener("click", () => updateFeedbackStatus(entry.id, "reviewed"));
    actions.append(reviewBtn);

    const resolveBtn = document.createElement("button");
    resolveBtn.className = "feedback-btn resolve-btn";
    resolveBtn.textContent = "标记已解决";
    resolveBtn.type = "button";
    resolveBtn.addEventListener("click", () => updateFeedbackStatus(entry.id, "resolved"));
    actions.append(resolveBtn);

    card.append(actions);
  }

  return card;
}

function statusLabel(status) {
  switch (status) {
    case "pending": return "待处理";
    case "reviewed": return "已审阅";
    case "resolved": return "已解决";
    default: return status;
  }
}

async function updateFeedbackStatus(id, status) {
  try {
    const response = await fetch("/dev/feedback/" + id, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ status, userId: userIdEl.value }),
    });
    if (!response.ok) {
      const body = await response.json();
      addSystemMessage("更新反馈状态失败: " + (body.error || "未知错误"));
      return;
    }
    addSystemMessage("反馈 #" + id + " 已更新为 " + statusLabel(status));
    loadFeedback();
  } catch (error) {
    addSystemMessage("更新反馈状态失败: " + (error instanceof Error ? error.message : String(error)));
  }
}

if (refreshFeedback) {
  refreshFeedback.addEventListener("click", loadFeedback);
}

// ── Chat with SSE streaming ──
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  button.disabled = true;
  addMessage("user", text);

  try {
    await setRole();
    const response = await fetch("/dev/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ userId: userIdEl.value, text })
    });

    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.error || "Request failed");
    }

    const { el: agentEl, content: agentContent } = createAgentMessage();
    let textContent = "";
    let thinkingEl = null;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const evt = JSON.parse(line.slice(6));
            switch (evt.type) {
              case "text_delta":
                if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
                textContent += evt.text;
                agentContent.textContent = textContent;
                agentEl.scrollIntoView({ block: "end", behavior: "smooth" });
                break;
              case "tool_use_start":
                if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
                agentContent.append(createToolCallCard(evt.toolName, evt.input, evt.toolUseId));
                agentEl.scrollIntoView({ block: "end", behavior: "smooth" });
                break;
              case "tool_use_result":
                updateToolCallCard(evt.toolUseId, evt.result, evt.isError);
                agentEl.scrollIntoView({ block: "end", behavior: "smooth" });
                break;
              case "thinking":
                if (!thinkingEl) {
                  thinkingEl = document.createElement("span");
                  thinkingEl.className = "thinking-indicator";
                  thinkingEl.textContent = "thinking";
                  agentContent.append(thinkingEl);
                }
                agentEl.scrollIntoView({ block: "end", behavior: "smooth" });
                break;
              case "completed":
                if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
                if (evt.text && evt.text !== textContent) {
                  textContent = evt.text;
                  agentContent.textContent = textContent;
                }
                break;
              case "error":
                addMessage("error", evt.message);
                break;
            }
          } catch (_parseError) {
            // skip malformed SSE data
          }
        }
      }
    }
  } catch (error) {
    addMessage("error", error instanceof Error ? error.message : String(error));
  } finally {
    button.disabled = false;
    input.focus();
  }
});
