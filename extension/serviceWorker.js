"use strict";

const API_BASE = "http://127.0.0.1:8765";

async function parseApiResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).join(" ")
      : detail || `Local service returned ${response.status}.`;
    throw new Error(message);
  }
  return payload;
}

async function getHealth() {
  const response = await fetch(`${API_BASE}/health`, { method: "GET" });
  return parseApiResponse(response);
}

async function predict(payload) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return parseApiResponse(response);
}

async function extractActiveEmail() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id || !String(tab.url || "").startsWith("https://mail.google.com/")) {
    return {
      ok: false,
      code: "NOT_GMAIL",
      message: "Open a controlled test email in Gmail, or use Paste mode."
    };
  }
  try {
    return await chrome.tabs.sendMessage(tab.id, { action: "EXTRACT_VISIBLE_EMAIL" });
  } catch (_error) {
    return {
      ok: false,
      code: "EXTRACTION_UNAVAILABLE",
      message: "Reload Gmail after installing the extension, or use Paste mode."
    };
  }
}

async function openDemo() {
  await chrome.tabs.create({ url: `${API_BASE}/demo` });
  return { opened: true };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  const actions = {
    HEALTH: getHealth,
    PREDICT: () => predict(message.payload),
    EXTRACT_ACTIVE_EMAIL: extractActiveEmail,
    OPEN_DEMO: openDemo
  };
  const action = actions[message && message.action];
  if (!action) return false;

  action()
    .then((data) => sendResponse({ ok: true, data }))
    .catch((error) =>
      sendResponse({
        ok: false,
        error: error instanceof TypeError
          ? "Local service is offline. Start it, then try again."
          : error.message
      })
    );
  return true;
});
