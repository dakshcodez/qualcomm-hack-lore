console.log("Lore background script loaded");

// PDF capture: Chrome's built-in PDF viewer does not allow content scripts
// to run inside it at all, so pdf_capture.js watches tab navigation from
// here instead. The actual pdf.js parsing happens in an offscreen document
// (offscreen.html/offscreen.js) since pdf.js needs a real DOM that this
// service worker doesn't have. See pdf_capture.js for details.
//
// Image OCR: image_ocr.js (content script) detects dwelled-on images and
// asks image_ocr_bridge.js to actually fetch + OCR them, for the same
// "needs a privileged context" reason as PDFs. Both share one offscreen
// document via offscreen_manager.js.
importScripts("offscreen_manager.js", "pdf_capture.js", "image_ocr_bridge.js");

const PC_ENDPOINT = "http://localhost:8000/index"; // update with real PC IP later

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "CAPTURE_PAGE") {
    sendToLore(message.data);
  }
});

async function sendToLore(pageData) {
  // Backend's IndexRequest schema is {text, url, title}; capture sites
  // (content_script.js, pdf_capture.js, image_ocr.js) all produce {url, title, content}.
  const body = { text: pageData.content, url: pageData.url, title: pageData.title };
  try {
    const response = await fetch(PC_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    if (response.ok) {
      console.log("Lore: indexed", pageData.title);
    }
  } catch (err) {
    console.error("Lore: failed to reach PC —", err.message);
  }
}