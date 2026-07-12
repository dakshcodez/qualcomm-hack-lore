console.log("Lore offscreen manager loaded");

// Shared by pdf_capture.js and image_ocr_bridge.js — both need the same
// single offscreen document (pdf.js and Tesseract.js both need a real
// document/window, which the service worker doesn't have) and must not
// race each other trying to create it twice.
let offscreenReadyPromise = null;

async function ensureOffscreenDocument() {
  if (offscreenReadyPromise) return offscreenReadyPromise;

  offscreenReadyPromise = chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["DOM_PARSER"],
    justification: "pdf.js and Tesseract.js both need a real document/window; the service worker has neither"
  }).catch(err => {
    if (!String(err.message).includes("single offscreen")) throw err; // already exists — fine, reuse it
  });

  return offscreenReadyPromise;
}
