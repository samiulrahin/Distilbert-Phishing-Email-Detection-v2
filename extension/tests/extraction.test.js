"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { clean, extractEmailFromDocument } = require("../lib/extraction.js");

function element(text, shown = true) {
  return {
    textContent: text,
    innerText: text,
    getBoundingClientRect: () => ({ width: shown ? 100 : 0, height: shown ? 20 : 0 })
  };
}

function fakeDocument(mapping) {
  return { querySelectorAll: (selector) => mapping[selector] || [] };
}

test("clean normalises whitespace without changing message meaning", () => {
  assert.equal(clean("  Hello\u00a0  team\n\n\nUpdate  "), "Hello team\n\nUpdate");
});

test("extracts only a visible Gmail subject and body", () => {
  const documentRef = fakeDocument({
    "h2.hP": [element("Security notice")],
    ".a3s.aiL": [element("Hidden previous message", false), element("Visible message body")]
  });
  assert.deepEqual(extractEmailFromDocument(documentRef), {
    ok: true,
    subject: "Security notice",
    body: "Visible message body",
    source: "controlled-gmail-test",
    extraction: "visible-open-message-only"
  });
});

test("returns a bounded error when no message is open", () => {
  const result = extractEmailFromDocument(fakeDocument({}));
  assert.equal(result.ok, false);
  assert.equal(result.code, "NO_OPEN_EMAIL");
});
