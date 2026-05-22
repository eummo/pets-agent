const form = document.querySelector("#chat-form");
const messages = document.querySelector("#messages");
const userIdEl = document.querySelector("#user-id");
const roleSelect = document.querySelector("#role-select");
const input = document.querySelector("#message-input");
const button = document.querySelector("#send-button");

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

// ── Role ──
async function loadRoles() {
  try {
    const response = await fetch("/dev/roles");
    const data = await response.json();
    const roles = data.roles || [];
    if (roles.length === 0) {
      addFallbackRoles();
      return;
    }
    roleSelect.innerHTML = "";
    for (const role of roles) {
      const option = document.createElement("option");
      option.value = role.name;
      option.textContent = role.name;
      roleSelect.append(option);
    }
    // Default to "reviewer" if available
    if (roleSelect.querySelector('option[value="reviewer"]')) {
      roleSelect.value = "reviewer";
    }
  } catch (_error) {
    addFallbackRoles();
  }
}
loadRoles();

function addFallbackRoles() {
  roleSelect.innerHTML = "";
  const reviewer = document.createElement("option");
  reviewer.value = "reviewer";
  reviewer.textContent = "reviewer";
  reviewer.selected = true;
  const developer = document.createElement("option");
  developer.value = "developer";
  developer.textContent = "developer";
  roleSelect.append(reviewer, developer);
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
  } catch (error) {
    addMessage("error", error instanceof Error ? error.message : String(error));
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
