(function exposeExtraction(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.MailRiskExtraction = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function createExtraction() {
  "use strict";

  const SUBJECT_SELECTORS = ["h2.hP", "[data-thread-perm-id] h2", "h2[data-legacy-thread-id]"];
  const BODY_SELECTORS = [".a3s.aiL", "[data-message-id] .a3s", ".adn .a3s"];

  function visible(element) {
    if (!element) return false;
    if (typeof element.getBoundingClientRect !== "function") return true;
    const box = element.getBoundingClientRect();
    return box.width > 0 && box.height > 0;
  }

  function firstVisible(documentRef, selectors) {
    for (const selector of selectors) {
      const candidates = Array.from(documentRef.querySelectorAll(selector));
      const candidate = candidates.find(visible);
      if (candidate) return candidate;
    }
    return null;
  }

  function clean(value) {
    return String(value || "")
      .replace(/\u00a0/g, " ")
      .replace(/[ \t]+/g, " ")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  function extractEmailFromDocument(documentRef) {
    const subjectElement = firstVisible(documentRef, SUBJECT_SELECTORS);
    const visibleBodies = [];
    for (const selector of BODY_SELECTORS) {
      for (const candidate of Array.from(documentRef.querySelectorAll(selector))) {
        if (visible(candidate) && !visibleBodies.includes(candidate)) {
          visibleBodies.push(candidate);
        }
      }
      if (visibleBodies.length) break;
    }

    const bodyElement = visibleBodies.at(-1) || null;
    const subject = clean(subjectElement && subjectElement.textContent);
    const body = clean(bodyElement && (bodyElement.innerText || bodyElement.textContent));
    if (!subject && !body) {
      return {
        ok: false,
        code: "NO_OPEN_EMAIL",
        message: "Open a controlled Gmail test email, then scan again."
      };
    }
    return {
      ok: true,
      subject: subject.slice(0, 1000),
      body: body.slice(0, 50000),
      source: "controlled-gmail-test",
      extraction: "visible-open-message-only"
    };
  }

  return { extractEmailFromDocument, clean, visible };
});

