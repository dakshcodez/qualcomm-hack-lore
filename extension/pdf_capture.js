console.log("Lore PDF capture (background) loaded");

// pdf.js needs a real document/window to run (it touches DOM APIs the
// service worker doesn't have, and can't reliably spawn nested Workers
// either), so actual parsing happens in offscreen.html/offscreen.js via
// Chrome's Offscreen Documents API. This file stays focused on tab
// tracking, dwell gating, and batching — it just asks the offscreen
// document to do the extraction and awaits the answer.

const DWELL_MS = 30000; // 30s flat dwell, proxy for the 1.5s-per-element read check content_script.js does on normal pages
const MIN_NEW_CHARS_TO_SEND = 300; // same batching floor used in content_script.js

// Per-tab state for in-progress PDF captures, keyed by tabId.
const pdfTabs = new Map();

function looksLikePdfUrl(url) {
  return !!url && /\.pdf(?:[?#]|$)/i.test(url);
}

// Many sites (arxiv.org/pdf/1706.03762, for example) serve PDFs from URLs
// with no ".pdf" extension at all. The URL check above is just a fast path
// — when it doesn't match, confirm via the actual Content-Type header
// instead of assuming it's not a PDF.
async function isPdfUrl(url) {
  if (looksLikePdfUrl(url)) return true;
  if (!/^https?:\/\//i.test(url)) return false;
  try {
    const response = await fetch(url, { method: "HEAD" });
    const contentType = response.headers.get("content-type") || "";
    return contentType.includes("application/pdf");
  } catch {
    return false;
  }
}

// ── STEP 1: Detect PDF tabs from tab navigation events ──────────────────
// Chrome's built-in PDF viewer blocks content script injection entirely, so
// instead of watching the DOM, we watch chrome.tabs for a tab finishing a
// navigation and check whether it landed on a PDF.
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status !== "complete" || !tab.url) return;

  if (!(await isPdfUrl(tab.url))) {
    if (pdfTabs.has(tabId)) flushPdfTab(tabId); // navigated away from a tracked PDF
    return;
  }

  const existing = pdfTabs.get(tabId);
  if (existing && existing.url === tab.url) return; // already tracking this exact PDF
  if (existing) flushPdfTab(tabId); // tab reused for a different PDF — flush the old one first

  startPdfCapture(tabId, tab.url, tab.title);
});

chrome.tabs.onRemoved.addListener((tabId) => {
  if (pdfTabs.has(tabId)) flushPdfTab(tabId); // tab closed before the dwell timer fired
});

function startPdfCapture(tabId, url, title) {
  const state = {
    url,
    title: title || url,
    extractedContent: null, // stays null until extraction succeeds
    dwellElapsed: false,
    sent: false,
    dwellTimer: null
  };
  pdfTabs.set(tabId, state);

  console.log("Lore PDF: tracking", url);

  extractPdfText(url)
    .then(text => {
      if (!text) {
        // Scanned/image-only PDFs have no text layer — fail loudly, don't
        // silently POST an empty body.
        console.warn("Lore PDF: no extractable text (likely a scanned/image-only PDF) —", url);
        return;
      }
      state.extractedContent = text;
      console.log("Lore PDF: extracted", text.length, "chars from", state.title);
      trySendNormal(tabId);
    })
    .catch(err => {
      console.error("Lore PDF: text extraction failed —", err && err.message, url);
    });

  state.dwellTimer = setTimeout(() => {
    state.dwellElapsed = true;
    trySendNormal(tabId);
  }, DWELL_MS);
}

// ── STEP 2: Delegate the actual PDF parsing to the offscreen document ───
// The tab's own PDF viewer already downloaded this file, but content
// scripts can't reach into its rendering internals, and pdf.js can't run
// directly in the service worker. So: make sure an offscreen document
// exists (see offscreen_manager.js, shared with image OCR), then ask it to
// fetch + parse the PDF and hand back the text.
async function extractPdfText(url) {
  await ensureOffscreenDocument();

  const response = await chrome.runtime.sendMessage({ type: "EXTRACT_PDF_TEXT", url });
  if (!response || !response.ok) {
    throw new Error(response ? response.error : "no response from offscreen document");
  }
  return response.text;
}

// "Normal" send path — only fires once BOTH the dwell timer has elapsed AND
// extraction succeeded AND the result clears the same batching floor used
// for normal pages. Mirrors content_script.js's checkAndSendNewContent().
function trySendNormal(tabId) {
  const state = pdfTabs.get(tabId);
  if (!state || state.sent || !state.dwellElapsed || !state.extractedContent) return;
  if (state.extractedContent.length < MIN_NEW_CHARS_TO_SEND) return;
  sendCapture(state);
  state.sent = true;
}

// ── STEP 3: Flush-on-leave ───────────────────────────────────────────────
// Mirrors content_script.js's flushRemainingContent(): if the user leaves
// (navigates away or closes the tab) before the dwell timer/extraction/
// threshold all line up naturally, send whatever we've got anyway so short
// PDFs aren't lost, then stop tracking this tab.
function flushPdfTab(tabId) {
  const state = pdfTabs.get(tabId);
  if (!state) return;
  clearTimeout(state.dwellTimer);
  if (!state.sent && state.extractedContent) {
    sendCapture(state);
    state.sent = true;
  }
  pdfTabs.delete(tabId);
}

// ── STEP 4: Funnel through the exact same POST logic as normal pages ─────
// content_script.js reaches sendToLore() (defined in background.js) via
// chrome.runtime.sendMessage. We're already inside the background service
// worker, so we call sendToLore() directly instead of round-tripping a
// message to ourselves — same data shape, same function, same POST logic.
// background.js does not need to know or care this came from a PDF.
function sendCapture(state) {
  console.log("Lore PDF sending capture:", state.title, "-", state.extractedContent.length, "chars");
  sendToLore({
    url: state.url,
    title: state.title,
    content: state.extractedContent
  });
}
