const form = document.querySelector("#chat-form");
const messages = document.querySelector("#messages");
const userIdEl = document.querySelector("#user-id");
const roleSelect = document.querySelector("#role-select");
const input = document.querySelector("#message-input");
const button = document.querySelector("#send-button");
const feedbackList = document.querySelector("#feedback-list");
const refreshFeedback = document.querySelector("#refresh-feedback");
const cronJobList = document.querySelector("#cron-job-list");
const refreshCron = document.querySelector("#refresh-cron");
const addCronJob = document.querySelector("#add-cron-job");
const cronFormOverlay = document.querySelector("#cron-form-overlay");
const cronJobForm = document.querySelector("#cron-job-form");
const cronFormClose = document.querySelector("#cron-form-close");
const cronFormCancel = document.querySelector("#cron-form-cancel");
const cronFormTitle = document.querySelector("#cron-form-title");
const cronEditId = document.querySelector("#cron-edit-id");
const cronScheduleType = document.querySelector("#cron-schedule-type");
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
    const panelMap = { chat: "chat-panel", feedback: "feedback-panel", cron: "cron-panel" };
    const panelId = panelMap[tab.dataset.tab] || "chat-panel";
    document.getElementById(panelId).classList.add("active");
    if (tab.dataset.tab === "feedback") loadFeedback();
    if (tab.dataset.tab === "cron") loadCronJobs();
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
  CRON_MANAGE: "cron_manage",
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
  updateCronTabVisibility();
}

function updateCapabilitiesFromRoles(roles) {
  const selected = roles.find((r) => r.name === roleSelect.value);
  currentCapabilities = selected?.capabilities || [];
  updateFeedbackTabVisibility();
  updateCronTabVisibility();
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
  updateCronTabVisibility();
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

function updateCronTabVisibility() {
  const cronTab = document.querySelector('.tab[data-tab="cron"]');
  const cronPanel = document.getElementById("cron-panel");
  const canManage = hasCapability(CAP.CRON_MANAGE);
  if (cronTab) {
    cronTab.style.display = canManage ? "" : "none";
  }
  if (!canManage && cronPanel?.classList.contains("active")) {
    cronPanel.classList.remove("active");
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

// ── Cron management ──
const scheduleTypeLabels = { cron: "Cron", interval: "固定间隔", once: "单次" };
const statusLabels = { success: "成功", error: "失败", timeout: "超时", skipped: "跳过" };

function formatSchedule(schedule) {
  switch (schedule.type) {
    case "cron": return schedule.expression;
    case "interval": return "每 " + (schedule.milliseconds / 60000) + " 分钟";
    case "once": return new Date(schedule.runAt).toLocaleString("zh-CN");
    default: return JSON.stringify(schedule);
  }
}

function formatChannels(channels) {
  return channels.map((c) => {
    if (c.startsWith("wecom:")) return "企微:" + c.slice(6);
    if (c.startsWith("sse:")) return "SSE:" + c.slice(4);
    if (c.startsWith("webhook:")) return "Webhook:" + c.slice(8);
    return c;
  }).join("、");
}

async function loadCronStatus() {
  const statusEl = document.querySelector("#cron-status");
  if (!statusEl) return;
  try {
    const resp = await fetch("/cron/status?userId=" + encodeURIComponent(userIdEl.value));
    if (!resp.ok) { statusEl.textContent = "未连接"; statusEl.className = "cron-scheduler-status stopped"; return; }
    const data = await resp.json();
    statusEl.textContent = data.running ? "运行中 (" + data.enabledJobs + "/" + data.totalJobs + ")" : "已停止";
    statusEl.className = "cron-scheduler-status" + (data.running ? "" : " stopped");
  } catch (_e) {
    statusEl.textContent = "未连接";
    statusEl.className = "cron-scheduler-status stopped";
  }
}

async function loadCronJobs() {
  if (!cronJobList) return;
  setCronMessage("cron-loading", "加载中...");
  await loadCronStatus();
  try {
    const resp = await fetch("/cron/jobs?userId=" + encodeURIComponent(userIdEl.value));
    if (!resp.ok) {
      const body = await resp.json();
      setCronMessage("cron-error", body.error || "加载失败");
      return;
    }
    const jobs = await resp.json();
    if (jobs.length === 0) {
      setCronMessage("cron-empty", "暂无定时任务，点击「新建任务」添加");
      return;
    }
    cronJobList.replaceChildren();
    for (const job of jobs) {
      cronJobList.append(createCronJobCard(job));
    }
  } catch (error) {
    setCronMessage("cron-error", "加载失败: " + (error instanceof Error ? error.message : String(error)));
  }
}

function setCronMessage(className, text) {
  const el = document.createElement("div");
  el.className = className;
  el.textContent = text;
  cronJobList.replaceChildren(el);
}

function createCronJobCard(job) {
  const card = document.createElement("div");
  card.className = "cron-job-card" + (job.enabled ? "" : " disabled");

  const header = document.createElement("div");
  header.className = "cron-job-card-header";

  const nameEl = document.createElement("span");
  nameEl.className = "cron-job-name";
  nameEl.textContent = job.name;
  header.append(nameEl);

  const idEl = document.createElement("span");
  idEl.className = "cron-job-id";
  idEl.textContent = job.id;
  header.append(idEl);

  const badge = document.createElement("span");
  badge.className = "cron-enabled-badge " + (job.enabled ? "enabled" : "disabled");
  badge.textContent = job.enabled ? "启用" : "停用";
  header.append(badge);

  card.append(header);

  const details = document.createElement("div");
  details.className = "cron-job-details";

  const fields = [
    ["调度", scheduleTypeLabels[job.schedule.type] + ": " + formatSchedule(job.schedule)],
    ["下次执行", job.nextRunAt ? new Date(job.nextRunAt).toLocaleString("zh-CN") : "未计算"],
    ["提示词", job.prompt],
    ["工作空间", job.workspacePath],
  ];
  if (job.role) fields.push(["角色", job.role]);
  fields.push(["投递渠道", formatChannels(job.delivery.channels)]);

  for (const [label, value] of fields) {
    const labelEl = document.createElement("span");
    labelEl.className = "cron-job-detail-label";
    labelEl.textContent = label;
    const valueEl = document.createElement("span");
    valueEl.className = "cron-job-detail-value";
    valueEl.textContent = value;
    details.append(labelEl, valueEl);
  }
  card.append(details);

  // Last result
  if (job.lastResult) {
    const resultEl = document.createElement("div");
    resultEl.className = "cron-job-result";
    const statusEl = document.createElement("div");
    statusEl.className = "cron-result-status " + job.lastResult.status;
    statusEl.textContent = (statusLabels[job.lastResult.status] || job.lastResult.status)
      + " — " + new Date(job.lastResult.finishedAt).toLocaleString("zh-CN");
    resultEl.append(statusEl);
    if (job.lastResult.output) {
      const outputEl = document.createElement("div");
      outputEl.className = "cron-result-output";
      outputEl.textContent = job.lastResult.output.slice(0, 500);
      resultEl.append(outputEl);
    }
    if (job.lastResult.error) {
      const errorEl = document.createElement("div");
      errorEl.className = "cron-result-output";
      errorEl.style.color = "#a94442";
      errorEl.textContent = job.lastResult.error;
      resultEl.append(errorEl);
    }
    card.append(resultEl);
  }

  // Actions
  const actions = document.createElement("div");
  actions.className = "cron-job-actions";

  const triggerBtn = document.createElement("button");
  triggerBtn.className = "cron-action-btn cron-trigger-btn";
  triggerBtn.type = "button";
  triggerBtn.textContent = "立即执行";
  triggerBtn.addEventListener("click", () => triggerCronJob(job.id));
  actions.append(triggerBtn);

  const editBtn = document.createElement("button");
  editBtn.className = "cron-action-btn cron-edit-btn";
  editBtn.type = "button";
  editBtn.textContent = "编辑";
  editBtn.addEventListener("click", () => openCronForm(job));
  actions.append(editBtn);

  const deleteBtn = document.createElement("button");
  deleteBtn.className = "cron-action-btn cron-delete-btn";
  deleteBtn.type = "button";
  deleteBtn.textContent = "删除";
  deleteBtn.addEventListener("click", () => deleteCronJob(job.id, job.name));
  actions.append(deleteBtn);

  card.append(actions);
  return card;
}

async function triggerCronJob(jobId) {
  try {
    const resp = await fetch("/cron/jobs/" + encodeURIComponent(jobId) + "/trigger", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ userId: userIdEl.value }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      addSystemMessage("触发任务失败: " + (data.error || "未知错误"));
      return;
    }
    addSystemMessage("任务 " + jobId + " 已触发: " + (statusLabels[data.status] || data.status));
    loadCronJobs();
  } catch (error) {
    addSystemMessage("触发任务失败: " + (error instanceof Error ? error.message : String(error)));
  }
}

async function deleteCronJob(jobId, jobName) {
  if (!confirm("确认删除定时任务「" + jobName + "」？")) return;
  try {
    const resp = await fetch("/cron/jobs/" + encodeURIComponent(jobId), {
      method: "DELETE",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ userId: userIdEl.value }),
    });
    if (!resp.ok && resp.status !== 204) {
      const data = await resp.json();
      addSystemMessage("删除任务失败: " + (data.error || "未知错误"));
      return;
    }
    addSystemMessage("定时任务「" + jobName + "」已删除");
    loadCronJobs();
  } catch (error) {
    addSystemMessage("删除任务失败: " + (error instanceof Error ? error.message : String(error)));
  }
}

// ── Cron form ──
function openCronForm(job) {
  cronEditId.value = job ? job.id : "";
  cronFormTitle.textContent = job ? "编辑定时任务" : "新建定时任务";
  document.querySelector("#cron-name").value = job ? job.name : "";
  document.querySelector("#cron-prompt").value = job ? job.prompt : "";
  document.querySelector("#cron-workspace").value = job ? job.workspacePath : ".harness/knowledge-base";
  document.querySelector("#cron-role").value = job?.role || "";
  document.querySelector("#cron-channels").value = job ? job.delivery.channels.join("\n") : "wecom:chat:";
  document.querySelector("#cron-timeout").value = job?.timeoutMs || 120000;
  document.querySelector("#cron-silent-empty").checked = job?.silentOnEmpty || false;
  document.querySelector("#cron-enabled").checked = job ? job.enabled : true;

  if (job) {
    cronScheduleType.value = job.schedule.type;
  } else {
    cronScheduleType.value = "cron";
  }
  updateScheduleFields();

  if (job?.schedule?.type === "cron") {
    document.querySelector("#cron-expression").value = job.schedule.expression;
  } else if (job?.schedule?.type === "interval") {
    document.querySelector("#cron-interval-ms").value = job.schedule.milliseconds;
  } else if (job?.schedule?.type === "once") {
    // Convert ISO to datetime-local
    const d = new Date(job.schedule.runAt);
    const local = new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
    document.querySelector("#cron-run-at").value = local;
  }

  cronFormOverlay.classList.remove("hidden");
}

function closeCronForm() {
  cronFormOverlay.classList.add("hidden");
}

function updateScheduleFields() {
  const type = cronScheduleType.value;
  document.querySelector("#cron-expr-label").classList.toggle("hidden", type !== "cron");
  document.querySelector("#cron-interval-label").classList.toggle("hidden", type !== "interval");
  document.querySelector("#cron-once-label").classList.toggle("hidden", type !== "once");
}

cronScheduleType.addEventListener("change", updateScheduleFields);
cronFormClose.addEventListener("click", closeCronForm);
cronFormCancel.addEventListener("click", closeCronForm);
addCronJob.addEventListener("click", () => openCronForm(null));
refreshCron.addEventListener("click", loadCronJobs);

// ── Natural language cron parse ──
const cronNlParse = document.querySelector("#cron-nl-parse");
const cronNlInput = document.querySelector("#cron-nl-input");

if (cronNlParse) {
  cronNlParse.addEventListener("click", async () => {
    const description = cronNlInput?.value?.trim();
    if (!description) { alert("请输入自然语言描述"); return; }

    cronNlParse.disabled = true;
    cronNlParse.textContent = "解析中...";
    try {
      const resp = await fetch("/cron/parse", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ description, userId: userIdEl.value }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        alert("解析失败: " + (data.error || "未知错误") + (data.details ? "\n" + data.details : ""));
        return;
      }
      // Fill form fields from parse result
      if (data.name) document.querySelector("#cron-name").value = data.name;
      if (data.prompt) document.querySelector("#cron-prompt").value = data.prompt;
      if (data.workspacePath) document.querySelector("#cron-workspace").value = data.workspacePath;
      if (data.role) document.querySelector("#cron-role").value = data.role;
      if (data.timeoutMs) document.querySelector("#cron-timeout").value = data.timeoutMs;
      if (data.silentOnEmpty !== undefined) document.querySelector("#cron-silent-empty").checked = data.silentOnEmpty;

      // Fill schedule
      if (data.schedule) {
        cronScheduleType.value = data.schedule.type;
        updateScheduleFields();
        if (data.schedule.type === "cron" && data.schedule.expression) {
          document.querySelector("#cron-expression").value = data.schedule.expression;
        } else if (data.schedule.type === "interval" && data.schedule.milliseconds) {
          document.querySelector("#cron-interval-ms").value = data.schedule.milliseconds;
        } else if (data.schedule.type === "once" && data.schedule.runAt) {
          const d = new Date(data.schedule.runAt);
          const local = new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
          document.querySelector("#cron-run-at").value = local;
        }
      }

      // Fill delivery channels
      if (data.delivery?.channels) {
        document.querySelector("#cron-channels").value = data.delivery.channels.join("\n");
      }

      addSystemMessage("自然语言描述已解析，请检查并调整表单字段后保存");
    } catch (error) {
      alert("解析失败: " + (error instanceof Error ? error.message : String(error)));
    } finally {
      cronNlParse.disabled = false;
      cronNlParse.textContent = "智能解析";
    }
  });
}

cronJobForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const editId = cronEditId.value;
  const type = cronScheduleType.value;

  let schedule;
  if (type === "cron") {
    const expr = document.querySelector("#cron-expression").value.trim();
    if (!expr) { alert("请输入 Cron 表达式"); return; }
    schedule = { type: "cron", expression: expr };
  } else if (type === "interval") {
    const ms = parseInt(document.querySelector("#cron-interval-ms").value, 10);
    if (!ms || ms < 60000) { alert("间隔不能小于 60000 毫秒"); return; }
    schedule = { type: "interval", milliseconds: ms };
  } else {
    const runAt = document.querySelector("#cron-run-at").value;
    if (!runAt) { alert("请选择执行时间"); return; }
    schedule = { type: "once", runAt: new Date(runAt).toISOString() };
  }

  const channels = document.querySelector("#cron-channels").value
    .split("\n").map((s) => s.trim()).filter((s) => s.length > 0);
  if (channels.length === 0) { alert("请至少添加一个投递渠道"); return; }

  const body = {
    name: document.querySelector("#cron-name").value.trim(),
    schedule,
    prompt: document.querySelector("#cron-prompt").value.trim(),
    workspacePath: document.querySelector("#cron-workspace").value.trim(),
    delivery: { channels },
    enabled: document.querySelector("#cron-enabled").checked,
    timeoutMs: parseInt(document.querySelector("#cron-timeout").value, 10) || 120000,
    silentOnEmpty: document.querySelector("#cron-silent-empty").checked,
    userId: userIdEl.value,
  };
  const role = document.querySelector("#cron-role").value;
  if (role) body.role = role;

  try {
    const url = editId ? "/cron/jobs/" + encodeURIComponent(editId) : "/cron/jobs";
    const method = editId ? "PATCH" : "POST";
    const resp = await fetch(url, { method, headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    if (!resp.ok) {
      const data = await resp.json();
      alert("保存失败: " + (data.error || "未知错误") + (data.details ? "\n" + data.details.join("\n") : ""));
      return;
    }
    closeCronForm();
    loadCronJobs();
  } catch (error) {
    alert("保存失败: " + (error instanceof Error ? error.message : String(error)));
  }
});

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
