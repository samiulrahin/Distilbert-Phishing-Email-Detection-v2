"use strict";

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message && message.action === "EXTRACT_VISIBLE_EMAIL") {
    sendResponse(MailRiskExtraction.extractEmailFromDocument(document));
  }
});

