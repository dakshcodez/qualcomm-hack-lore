console.log("Lore image OCR content script loaded");

// ── STEP 1: Find images worth bothering with ────────────────────────────
// Skip tiny icons/avatars/tracking pixels — OCR on those is wasted work and
// almost never has real text in it.
const MIN_IMG_DIMENSION = 150;

function isWorthOcr(img) {
  // SVGs are vector graphics, not raster images — Tesseract's image decoder
  // can't read them at all, so don't even try.
  if (/\.svg(?:[?#]|$)/i.test(img.src)) return false;
  return img.naturalWidth >= MIN_IMG_DIMENSION && img.naturalHeight >= MIN_IMG_DIMENSION;
}

// Same one-time snapshot approach as content_script.js: images added to the
// page later (infinite scroll, lazy-loaded content) are not picked up. That
// mirrors an existing limitation of the normal-page capture, not a new one.
const candidateImages = Array.from(document.images).filter(img => img.complete && isWorthOcr(img));

console.log("Lore image OCR: found", candidateImages.length, "image(s) worth checking on this page");

if (candidateImages.length > 0) {

  // ── STEP 2: Track dwell time per image, same 1.5s threshold as text ────
  const DWELL_MS = 1500;
  const MIN_NEW_CHARS_TO_SEND = 300; // same batching floor as content_script.js

  const readImages = new Set();      // images visible 1.5s+ = "read"
  const sentImages = new Set();      // images whose OCR text has been sent
  const ocrResults = new Map();      // image el -> extracted text (once OCR finishes)
  const ocrInFlight = new Set();     // images currently being OCR'd, to avoid duplicate work
  const ocrDone = new Set();         // images we've already finished with (success, no-text, or failure) — never retry these
  const visibleSince = new Map();

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      const el = entry.target;
      if (entry.isIntersecting) {
        visibleSince.set(el, Date.now());
      } else {
        const shownAt = visibleSince.get(el);
        if (shownAt && (Date.now() - shownAt) >= DWELL_MS) {
          readImages.add(el);
          queueOcr(el);
        }
        visibleSince.delete(el);
      }
    });
  }, { threshold: 0.5 });

  candidateImages.forEach(img => observer.observe(img));

  function finalizeVisibleImages() {
    visibleSince.forEach((shownAt, el) => {
      if (Date.now() - shownAt >= DWELL_MS) {
        readImages.add(el);
        queueOcr(el);
      }
    });
  }

  // ── STEP 3: Delegate the actual fetch + OCR to the background ───────────
  // A content script can't reliably create a Tesseract Worker across
  // origins, and its fetch() of a cross-origin image is bound by the
  // page's own CORS policy with no exceptions — both need a privileged
  // context. image_ocr_bridge.js (background) and offscreen.js do the real
  // work; this just asks for it and waits for the answer.
  async function queueOcr(img) {
    if (ocrInFlight.has(img) || ocrDone.has(img)) return;
    ocrInFlight.add(img);
    try {
      const response = await chrome.runtime.sendMessage({ type: "OCR_IMAGE", url: img.src });
      if (!response || !response.ok) {
        throw new Error(response ? response.error : "no response from background");
      }

      const trimmed = response.text.trim();
      ocrDone.add(img); // one attempt only — success, empty, or failure below all count as "done"
      if (trimmed.length === 0) {
        // Most images have no text at all — this is expected, not an error.
        console.log("Lore image OCR: no text found in", img.src);
        return;
      }

      ocrResults.set(img, trimmed);
      console.log("Lore image OCR: extracted", trimmed.length, "chars from", img.src);
      checkAndSendNewContent();
    } catch (err) {
      ocrDone.add(img); // don't retry forever on images Tesseract can't read (e.g. SVGs slipping through, corrupt files)
      console.error("Lore image OCR: failed on", img.src, "—", err.message);
    } finally {
      ocrInFlight.delete(img);
    }
  }

  // ── STEP 4: Batch sending — same floor/interval/flush shape as content_script.js ──
  let isTabVisible = !document.hidden;

  document.addEventListener("visibilitychange", () => {
    isTabVisible = !document.hidden;
    if (document.hidden) {
      flushRemainingContent();
    }
  });

  function checkAndSendNewContent() {
    if (!isTabVisible) return;
    finalizeVisibleImages();

    const unsent = Array.from(ocrResults.entries()).filter(([img]) => !sentImages.has(img));
    if (unsent.length === 0) return;

    const newContent = unsent.map(([, text]) => text).join("\n\n");
    if (newContent.length >= MIN_NEW_CHARS_TO_SEND) {
      sendCapture(newContent);
      unsent.forEach(([img]) => sentImages.add(img));
    }
  }

  setInterval(checkAndSendNewContent, 5000);

  function flushRemainingContent() {
    finalizeVisibleImages();

    const unsent = Array.from(ocrResults.entries()).filter(([img]) => !sentImages.has(img));
    if (unsent.length === 0) return;

    const remainingContent = unsent.map(([, text]) => text).join("\n\n");
    if (remainingContent.length > 0) {
      sendCapture(remainingContent);
      unsent.forEach(([img]) => sentImages.add(img));
    }
  }

  window.addEventListener("beforeunload", flushRemainingContent);

  // ── STEP 5: Send through the exact same pipeline as normal-page text ───
  function sendCapture(content) {
    const skipDomains = ["google.com/search", "mail.google.com"];
    if (skipDomains.some(d => window.location.href.includes(d))) return;

    console.log("Lore image OCR sending batch:", document.title, "-", content.length, "chars");

    chrome.runtime.sendMessage({
      type: "CAPTURE_PAGE",
      data: {
        url: window.location.href,
        title: document.title,
        content: content
      }
    });
  }
}
