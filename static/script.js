const chatEl = document.getElementById("chat");
const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const welcome = document.getElementById("welcome");

// Conversation sent back to the server so follow-up questions have context.
let history = [];
let busy = false;

// ---------- Helpers ----------
function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Minimal, safe markdown: escape everything first, then add bold, links, and bullet/numbered lists.
function renderMarkdown(text) {
  const inline = (s) =>
    escapeHtml(s)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
      .replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');

  let html = "";
  let list = null; // "ul" or "ol" while inside a list
  for (const line of text.split("\n")) {
    const item = line.match(/^\s*([-*•]|\d+\.)\s+(.*)/);
    if (item) {
      const type = /\d/.test(item[1]) ? "ol" : "ul";
      if (list !== type) {
        if (list) html += `</${list}>`;
        html += `<${type}>`;
        list = type;
      }
      html += `<li>${inline(item[2])}</li>`;
      continue;
    }
    if (list) { html += `</${list}>`; list = null; }
    if (line.trim()) html += `<p>${inline(line)}</p>`;
  }
  return list ? html + `</${list}>` : html;
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addMessage(role, html) {
  welcome.hidden = true;
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = html;
  chatEl.appendChild(el);
  scrollToBottom();
  return el;
}

function renderSources(sources) {
  if (!sources.length) return "";
  // Several chunks often come from the same page; list each page once (sources arrive best-first).
  const seen = new Set();
  sources = sources.filter((s) => !seen.has(s.url) && seen.add(s.url));
  const items = sources
    .map((s) => `<li><a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.title || s.url)}</a> <span class="score">(${s.score.toFixed(2)})</span></li>`)
    .join("");
  return `<details class="sources"><summary>Sources (${sources.length})</summary><ul>${items}</ul></details>`;
}

function setBusy(value) {
  busy = value;
  sendBtn.disabled = value;
  input.disabled = value;
  if (!value) input.focus();
}

// ---------- Chat ----------
async function ask(question) {
  if (busy || !question.trim()) return;
  setBusy(true);

  addMessage("user", escapeHtml(question));
  const botEl = addMessage("bot", '<div class="typing"><span></span><span></span><span></span></div>');

  let answer = "";
  let sources = [];

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
    if (!res.ok) throw new Error(`Server error (${res.status})`);

    // Read the newline-delimited JSON stream as it arrives.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep any partial line for the next read

      for (const line of lines) {
        if (!line.trim()) continue;
        const evt = JSON.parse(line);
        if (evt.type === "sources") {
          sources = evt.sources;
        } else if (evt.type === "token") {
          answer += evt.text;
          botEl.innerHTML = renderMarkdown(answer);
          scrollToBottom();
        } else if (evt.type === "error") {
          throw new Error(evt.message);
        }
      }
    }

    botEl.innerHTML = renderMarkdown(answer) + renderSources(sources);
    history.push({ role: "user", content: question }, { role: "assistant", content: answer });
  } catch (err) {
    botEl.classList.add("error");
    botEl.textContent = `⚠️ ${err.message}`;
  } finally {
    scrollToBottom();
    setBusy(false);
  }
}

// ---------- Events ----------
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value;
  input.value = "";
  input.style.height = "auto";
  ask(question);
});

// Enter sends, Shift+Enter adds a new line.
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

// Auto-grow the textarea as the user types.
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
});

document.querySelectorAll(".suggestion").forEach((btn) =>
  btn.addEventListener("click", () => ask(btn.textContent))
);

document.getElementById("clear-btn").addEventListener("click", () => {
  if (busy) return;
  history = [];
  chatEl.querySelectorAll(".msg").forEach((el) => el.remove());
  welcome.hidden = false;
  input.focus();
});

// Show which model is answering.
fetch("/api/info")
  .then((r) => r.json())
  .then((info) => {
    document.getElementById("subtitle").textContent =
      `Answers grounded in zoikorooms.com · ${info.chunks} indexed chunks · ${info.model} on Groq`;
  })
  .catch(() => {});

input.focus();
