"use strict";

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

const elements = {
  serviceState: document.querySelector("#serviceState"),
  serviceText: document.querySelector("#serviceText"),
  form: document.querySelector("#analysisForm"),
  subject: document.querySelector("#subjectInput"),
  body: document.querySelector("#bodyInput"),
  characterCount: document.querySelector("#characterCount"),
  reset: document.querySelector("#resetButton"),
  analyse: document.querySelector("#analyseButton"),
  analyseText: document.querySelector("#analyseButtonText"),
  formMessage: document.querySelector("#formMessage"),
  resultWorkspace: document.querySelector("#resultWorkspace"),
  emptyResult: document.querySelector("#emptyResult"),
  resultContent: document.querySelector("#resultContent"),
  resultMark: document.querySelector("#resultMark"),
  resultLabel: document.querySelector("#resultLabel"),
  resultAction: document.querySelector("#resultAction"),
  probabilityValue: document.querySelector("#probabilityValue"),
  probabilityBar: document.querySelector("#probabilityBar"),
  legitimateMarker: document.querySelector("#legitimateMarker"),
  phishingMarker: document.querySelector("#phishingMarker"),
  thresholdLabel: document.querySelector("#thresholdLabel"),
  inferenceValue: document.querySelector("#inferenceValue"),
  modelValue: document.querySelector("#modelValue"),
  signalsSection: document.querySelector("#signalsSection"),
  signalsList: document.querySelector("#signalsList"),
  limitationText: document.querySelector("#limitationText")
};

const stepItems = Array.from(document.querySelectorAll(".steps li"));

function setStep(activeIndex) {
  stepItems.forEach((item, index) => {
    item.classList.toggle("active", index === activeIndex);
    item.classList.toggle("completed", index < activeIndex);
  });
}

function setMessage(message = "") {
  elements.formMessage.textContent = message;
  elements.formMessage.classList.toggle("hidden", !message);
}

function updateCount() {
  elements.characterCount.textContent = elements.body.value.length.toLocaleString();
}

function setBusy(busy) {
  elements.analyse.disabled = busy;
  elements.analyseText.textContent = busy ? "Analysing locally..." : "Analyse email";
}

function loadSample(name) {
  const sample = samples[name];
  if (!sample) return;
  elements.subject.value = sample.subject;
  elements.body.value = sample.body;
  updateCount();
  setMessage();
  for (const button of document.querySelectorAll(".sample-button")) {
    button.classList.toggle("selected", button.dataset.sample === name);
  }
  setStep(1);
  elements.analyse.focus();
}

function reset() {
  elements.form.reset();
  updateCount();
  setMessage();
  elements.resultWorkspace.className = "result-workspace empty";
  elements.emptyResult.classList.remove("hidden");
  elements.resultContent.classList.add("hidden");
  for (const button of document.querySelectorAll(".sample-button")) button.classList.remove("selected");
  setStep(0);
  elements.subject.focus();
}

function modelName(mode, version = "") {
  if (mode === "distilbert") return `DistilBERT ${version}`.trim();
  if (mode === "baseline") return "TF-IDF baseline";
  return "Rule demonstration";
}

function renderResult(result) {
  const labels = { phishing: "High phishing risk", legitimate: "Low phishing risk", uncertain: "Review required" };
  const actions = {
    phishing: "Treat as suspicious. Verify the sender through a separate trusted channel.",
    legitimate: "No strong phishing classification, but continue normal checks before acting.",
    uncertain: "Do not rely on this result alone. Review the sender, links, and requested action."
  };
  const marks = { phishing: "!", legitimate: "\u2713", uncertain: "?" };
  const probability = Math.round(result.phishing_probability * 100);
  const legitimateMax = Math.round(result.legitimate_max * 100);
  const phishingMin = Math.round(result.phishing_min * 100);

  elements.resultWorkspace.className = `result-workspace ${result.label}`;
  elements.emptyResult.classList.add("hidden");
  elements.resultContent.classList.remove("hidden");
  elements.resultMark.textContent = marks[result.label] || "?";
  elements.resultLabel.textContent = labels[result.label] || "Uncertain result";
  elements.resultAction.textContent = actions[result.label] || actions.uncertain;
  elements.probabilityValue.textContent = `${probability}%`;
  elements.probabilityBar.style.width = `${probability}%`;
  elements.legitimateMarker.style.left = `${legitimateMax}%`;
  elements.phishingMarker.style.left = `${phishingMin}%`;
  elements.thresholdLabel.textContent = `Review band ${legitimateMax}-${phishingMin}%`;
  elements.inferenceValue.textContent = `${Number(result.inference_ms).toFixed(1)} ms`;
  elements.modelValue.textContent = modelName(result.model_mode, result.model_version);
  elements.limitationText.textContent = result.limitation;

  elements.signalsList.replaceChildren();
  for (const signal of result.risk_signals || []) {
    const item = document.createElement("li");
    item.textContent = signal.description;
    elements.signalsList.appendChild(item);
  }
  elements.signalsSection.classList.toggle("hidden", !(result.risk_signals || []).length);
  setStep(2);
  elements.resultWorkspace.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function analyse(event) {
  event.preventDefault();
  const payload = {
    subject: elements.subject.value.trim(),
    body: elements.body.value.trim(),
    source: "paste-mode"
  };
  if (!payload.subject && !payload.body) {
    setMessage("Enter an email subject or message body, or load a safe example.");
    elements.subject.focus();
    return;
  }

  setMessage();
  setStep(1);
  setBusy(true);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || `Analysis failed (${response.status}).`);
    renderResult(result);
  } catch (error) {
    setMessage(error instanceof TypeError
      ? "The local model service is unavailable. Start Mail Risk Lab and try again."
      : String(error.message || error));
  } finally {
    setBusy(false);
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error();
    const health = await response.json();
    elements.serviceState.className = "service-state online";
    elements.serviceText.textContent = `${modelName(health.model_mode, health.model_version)} ready`;
    elements.serviceState.title = `${health.model_name}; ${health.privacy}`;
  } catch (_error) {
    elements.serviceState.className = "service-state offline";
    elements.serviceText.textContent = "Service offline";
  }
}

for (const button of document.querySelectorAll(".sample-button")) {
  button.addEventListener("click", () => loadSample(button.dataset.sample));
}
elements.body.addEventListener("input", updateCount);
elements.form.addEventListener("submit", analyse);
elements.reset.addEventListener("click", reset);
checkHealth();
