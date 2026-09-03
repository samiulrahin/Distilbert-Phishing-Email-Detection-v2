"use strict";

const elements = {
  serviceState: document.querySelector("#serviceState"),
  serviceText: document.querySelector("#serviceText"),
  gmailTab: document.querySelector("#gmailTab"),
  pasteTab: document.querySelector("#pasteTab"),
  gmailPanel: document.querySelector("#gmailPanel"),
  pastePanel: document.querySelector("#pastePanel"),
  subjectInput: document.querySelector("#subjectInput"),
  bodyInput: document.querySelector("#bodyInput"),
  sampleSelect: document.querySelector("#sampleSelect"),
  loadSampleButton: document.querySelector("#loadSampleButton"),
  characterCount: document.querySelector("#characterCount"),
  scanButton: document.querySelector("#scanButton"),
  scanButtonText: document.querySelector("#scanButtonText"),
  openDemoButton: document.querySelector("#openDemoButton"),
  messagePanel: document.querySelector("#messagePanel"),
  resultPanel: document.querySelector("#resultPanel"),
  resultMark: document.querySelector("#resultMark"),
  resultLabel: document.querySelector("#resultLabel"),
  probabilityValue: document.querySelector("#probabilityValue"),
  confidenceBar: document.querySelector("#confidenceBar"),
  inferenceValue: document.querySelector("#inferenceValue"),
  modelValue: document.querySelector("#modelValue"),
  policyValue: document.querySelector("#policyValue"),
  signalsSection: document.querySelector("#signalsSection"),
  signalsList: document.querySelector("#signalsList"),
  limitationText: document.querySelector("#limitationText")
};

let mode = "gmail";

const samples = {
  legitimate: {
    subject: "Project supervision meeting agenda",
    body: "Hello, the agenda for our scheduled supervision meeting is available in the course workspace. We will discuss progress, risks, and next steps. Thank you."
  },
  phishing: {
    subject: "Urgent: account access expires today",
    body: "Your mailbox will be disabled today. Click https://account-check.example.invalid immediately and enter your password to verify your account."
  },
  uncertain: {
    subject: "Service message: Review of recent account activity",
    body: "We noticed a sign-in from a new browser. If you recognise it, you do not need to do anything. Otherwise, visit the service by typing its normal address yourself and check recent activity. Keep this email as a record of the notification."
  },
  limitation: {
    subject: "Service message: Two-step verification enabled",
    body: "Two-step verification was enabled on your account. This message confirms the change and will never ask you to send a password or verification code. Keep this email as a record of the notification."
  }
};

function sendMessage(message) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
      if (!response || !response.ok) return reject(new Error(response?.error || "Unexpected extension error."));
      resolve(response.data);
    });
  });
}

function setMode(nextMode) {
  mode = nextMode;
  const gmailActive = mode === "gmail";
  elements.gmailTab.classList.toggle("active", gmailActive);
  elements.pasteTab.classList.toggle("active", !gmailActive);
  elements.gmailTab.setAttribute("aria-selected", String(gmailActive));
  elements.pasteTab.setAttribute("aria-selected", String(!gmailActive));
  elements.gmailPanel.classList.toggle("hidden", !gmailActive);
  elements.pastePanel.classList.toggle("hidden", gmailActive);
  elements.scanButtonText.textContent = gmailActive ? "Scan current email" : "Analyse pasted text";
  hideMessage();
  elements.resultPanel.classList.add("hidden");
}

function showMessage(text, isError = false) {
  elements.messagePanel.textContent = text;
  elements.messagePanel.classList.toggle("error", isError);
  elements.messagePanel.classList.remove("hidden");
}

function hideMessage() {
  elements.messagePanel.classList.add("hidden");
}

function setBusy(busy) {
  elements.scanButton.disabled = busy;
  if (busy) elements.scanButtonText.textContent = "Analysing...";
  else elements.scanButtonText.textContent = mode === "gmail" ? "Scan current email" : "Analyse pasted text";
}

function loadSample() {
  const sample = samples[elements.sampleSelect.value];
  if (!sample) {
    showMessage("Choose a safe example first.");
    return;
  }
  elements.subjectInput.value = sample.subject;
  elements.bodyInput.value = sample.body;
  elements.characterCount.textContent = sample.body.length.toLocaleString();
  hideMessage();
  elements.resultPanel.classList.add("hidden");
  elements.scanButton.focus();
}

function renderResult(result) {
  const label = result.label;
  const labels = { phishing: "High phishing risk", legitimate: "Low phishing risk", uncertain: "Review required" };
  const marks = { phishing: "!", legitimate: "✓", uncertain: "?" };
  const probability = Math.round(result.phishing_probability * 100);

  elements.resultPanel.className = `result-panel ${label}`;
  elements.resultMark.textContent = marks[label] || "?";
  elements.resultLabel.textContent = labels[label] || "Uncertain result";
  elements.probabilityValue.textContent = `${probability}%`;
  elements.confidenceBar.style.width = `${probability}%`;
  elements.inferenceValue.textContent = `${Number(result.inference_ms).toFixed(1)} ms`;
  elements.modelValue.textContent = result.model_mode === "distilbert"
    ? `DistilBERT ${result.model_version}`
    : result.model_mode === "baseline" ? "TF-IDF baseline" : "Demo rules";
  elements.policyValue.textContent = `${Math.round(result.legitimate_max * 100)}-${Math.round(result.phishing_min * 100)}%`;
  elements.limitationText.textContent = result.limitation;

  elements.signalsList.replaceChildren();
  for (const signal of result.risk_signals || []) {
    const item = document.createElement("li");
    item.textContent = signal.description;
    elements.signalsList.appendChild(item);
  }
  elements.signalsSection.classList.toggle("hidden", !(result.risk_signals || []).length);
  elements.resultPanel.classList.remove("hidden");
}

async function scan() {
  hideMessage();
  elements.resultPanel.classList.add("hidden");
  setBusy(true);
  try {
    let payload;
    if (mode === "gmail") {
      const extracted = await sendMessage({ action: "EXTRACT_ACTIVE_EMAIL" });
      if (!extracted.ok) throw new Error(extracted.message);
      payload = { subject: extracted.subject, body: extracted.body, source: extracted.source };
    } else {
      payload = {
        subject: elements.subjectInput.value.trim(),
        body: elements.bodyInput.value.trim(),
        source: "paste-mode"
      };
      if (!payload.subject && !payload.body) throw new Error("Enter a subject or controlled email body first.");
    }
    renderResult(await sendMessage({ action: "PREDICT", payload }));
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function checkHealth() {
  try {
    const health = await sendMessage({ action: "HEALTH" });
    elements.serviceState.className = "service-state online";
    elements.serviceText.textContent = health.model_mode === "distilbert" ? "Model ready" : "Demo ready";
    elements.serviceState.title = `${health.model_name}; local-only service`;
  } catch (_error) {
    elements.serviceState.className = "service-state offline";
    elements.serviceText.textContent = "Offline";
    elements.serviceState.title = "Start the local service on 127.0.0.1:8765";
  }
}

elements.gmailTab.addEventListener("click", () => setMode("gmail"));
elements.pasteTab.addEventListener("click", () => setMode("paste"));
elements.bodyInput.addEventListener("input", () => {
  elements.characterCount.textContent = elements.bodyInput.value.length.toLocaleString();
});
elements.loadSampleButton.addEventListener("click", loadSample);
elements.scanButton.addEventListener("click", scan);
elements.openDemoButton.addEventListener("click", () => sendMessage({ action: "OPEN_DEMO" }).catch((error) => showMessage(error.message, true)));
checkHealth();
