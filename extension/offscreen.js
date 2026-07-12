console.log("Lore offscreen document loaded (pdf.js runs here — real document, unlike the service worker)");

pdfjsLib.GlobalWorkerOptions.workerSrc = chrome.runtime.getURL("libs/pdf.worker.js");

// Lazily-created OCR worker, shared across PDF pages AND normal-page images
// — only spun up on first actual use.
let ocrWorkerPromise = null;
function getOcrWorker() {
  if (!ocrWorkerPromise) {
    console.log("Lore offscreen: starting Tesseract worker");
    ocrWorkerPromise = Tesseract.createWorker("eng", 1, {
      workerPath: chrome.runtime.getURL("libs/tesseract/worker.min.js"),
      corePath: chrome.runtime.getURL("libs/tesseract/core"),
      langPath: chrome.runtime.getURL("libs/tesseract/lang"),
      gzip: true,
      // Tesseract.js normally wraps worker creation in a Blob/importScripts
      // trick. That blob gets an opaque origin that our extension's
      // web_accessible_resources rules don't recognize, so its internal
      // importScripts call gets blocked. Disabling this makes it load the
      // worker script directly instead — the same approach pdf.worker.js
      // already uses successfully.
      workerBlobURL: false,
      logger: () => {}
    });
  }
  return ocrWorkerPromise;
}

// A stuck render/recognize call should fail loudly after a while, not hang
// forever with zero indication of what's wrong.
function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms))
  ]);
}

// Renders each PDF page to a canvas and OCRs it. Used as a fallback for
// scanned/image-only PDFs that have no real text layer for pdf.js to read.
//
// KNOWN LIMITATION: some scanned PDFs (observed with fax-style scans, likely
// CCITT/JBIG2-encoded) make pdf.js's page.render() hang indefinitely —
// confirmed via testing to survive a 5-minute timeout with no completion and
// no timeout error either. That means it's not an async wait but a
// synchronous infinite loop inside pdf.js's own decoder, which blocks the
// whole JS thread — no Promise-based timeout (including the one below) can
// ever fire in that case, because the timer callback can't run until the
// thread is free. A real fix would mean running pdf.js rendering in a
// separate, forcibly-terminable Worker; that's a substantially bigger
// change than this hackathon build takes on. If a PDF hangs here, it simply
// won't be captured — treated as an accepted gap, not a bug to chase further.
// The timeout below still has value for the (more common) case of a render
// that's merely slow, or genuinely async-stuck, rather than sync-looping.
async function ocrPdfPages(pdf) {
  const worker = await getOcrWorker();
  let fullText = "";

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    console.log("Lore offscreen OCR: starting page", pageNum, "of", pdf.numPages);

    const page = await pdf.getPage(pageNum);
    // scale 1.5, not 2 — OCR accuracy gains above ~150-200 DPI are marginal,
    // but compute cost scales with pixel count (quadratically with scale),
    // so this cuts per-page time substantially for large scanned pages.
    const viewport = page.getViewport({ scale: 1.5 });
    // OffscreenCanvas rather than document.createElement("canvas") — not
    // tied to the document's paint cycle, which an offscreen document never
    // participates in. Doesn't fix the known hang above, but is still the
    // more correct rendering target for a non-visible document.
    const canvas = new OffscreenCanvas(viewport.width, viewport.height);

    await withTimeout(
      page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise,
      45000,
      `render of page ${pageNum}`
    );
    console.log("Lore offscreen OCR: rendered page", pageNum, "— starting recognition");

    const { data: { text } } = await withTimeout(
      worker.recognize(canvas),
      120000,
      `recognize of page ${pageNum}`
    );
    console.log("Lore offscreen OCR: finished page", pageNum, "of", pdf.numPages, "—", text.trim().length, "chars");
    fullText += text.trim() + "\n\n";
  }

  return fullText.trim();
}

// pdf_capture.js (in the background service worker) sends EXTRACT_PDF_TEXT
// messages here because pdf.js needs a real document/window to run — the
// service worker has neither a DOM nor (reliably) nested Workers.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type !== "EXTRACT_PDF_TEXT") return false; // not for us

  extractPdfText(message.url)
    .then(text => sendResponse({ ok: true, text }))
    .catch(err => {
      // Log the full error here too — this console belongs to the offscreen
      // document, a separate DevTools target from both the page and the
      // service worker, so this is the only place these details show up.
      console.error("Lore offscreen: extraction/OCR failed —", err);
      sendResponse({ ok: false, error: (err && err.message) || String(err) });
    });

  return true; // keep the message channel open for the async sendResponse above
});

async function extractPdfText(url) {
  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

  let fullText = "";
  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const textContent = await page.getTextContent();
    fullText += textContent.items.map(item => item.str).join(" ") + "\n\n";
  }
  fullText = fullText.trim();

  if (fullText.length > 0) return fullText;

  // No real text layer at all — likely a scanned/image-only PDF. Fall back
  // to rendering each page as an image and running OCR on it.
  console.log("Lore offscreen: no text layer found, falling back to OCR on rendered pages —", url);
  return await ocrPdfPages(pdf);
}

// image_ocr_bridge.js (background) sends EXTRACT_IMAGE_TEXT messages here
// for the same reason PDF extraction happens here: fetching a cross-origin
// image and creating a Tesseract Worker both need a privileged context that
// a content script running inside a foreign page doesn't have.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type !== "EXTRACT_IMAGE_TEXT") return false; // not for us

  extractImageText(message.url)
    .then(text => sendResponse({ ok: true, text }))
    .catch(err => {
      console.error("Lore offscreen: image OCR failed —", err, message.url);
      sendResponse({ ok: false, error: (err && err.message) || String(err) });
    });

  return true; // keep the message channel open for the async sendResponse above
});

async function extractImageText(url) {
  const response = await fetch(url);
  const blob = await response.blob();
  const worker = await getOcrWorker();
  const { data: { text } } = await worker.recognize(blob);
  return text.trim();
}
