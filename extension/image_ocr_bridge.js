console.log("Lore image OCR bridge (background) loaded");

// image_ocr.js (content script) only handles dwell-detection on images —
// the actual fetch + OCR happens here / in the offscreen document, for two
// reasons proven out by real testing:
//   1. A content script's Worker() pointing at a chrome-extension:// URL
//      gets rejected by the page's origin, even with web_accessible_resources
//      declared — the offscreen document (a privileged extension page)
//      doesn't have this problem.
//   2. A content script's fetch() of a cross-origin image is bound by the
//      PAGE's own CORS policy, no exceptions — host_permissions only lets
//      privileged contexts (background, offscreen document) bypass CORS.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type !== "OCR_IMAGE") return false; // not for us

  (async () => {
    try {
      await ensureOffscreenDocument();
      const response = await chrome.runtime.sendMessage({ type: "EXTRACT_IMAGE_TEXT", url: message.url });
      sendResponse(response);
    } catch (err) {
      // Without this, a failure here would leave sendResponse uncalled and
      // the content script would just see a generic "message port closed"
      // error with no indication of what actually went wrong.
      console.error("Lore image OCR bridge: failed to reach offscreen document —", err);
      sendResponse({ ok: false, error: err.message });
    }
  })();

  return true; // keep the message channel open for the async sendResponse above
});
