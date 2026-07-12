console.log("Lore content script loaded");

// ── STEP 1: Find all readable elements on the page ────────────────────
// We watch more than just <p> tags because many sites (list pages, roundups,
// etc.) put real content in <li>, headings, or <article>/<blockquote> blocks.
const paragraphs = Array.from(
  document.querySelectorAll("p, li, h1, h2, h3, article, blockquote")
).filter(p => p.innerText.trim().length > 40); // ignore tiny/junk elements

console.log("Lore found", paragraphs.length, "elements on this page");

// ── STEP 2: Track which elements have actually been read ───────────────
const readParagraphs = new Set();   // elements confirmed as "read" (visible 1.5s+)
const sentParagraphs = new Set();   // elements already sent to the PC
const visibleSince = new Map();     // when each element became visible

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    const el = entry.target;
    if (entry.isIntersecting) {
      visibleSince.set(el, Date.now());
    } else {
      const shownAt = visibleSince.get(el);
      if (shownAt && (Date.now() - shownAt) >= 1500) {
        readParagraphs.add(el); // visible 1.5+ sec = counts as read
      }
      visibleSince.delete(el);
    }
  });
}, { threshold: 0.5 }); // must be at least 50% visible to count

paragraphs.forEach(p => observer.observe(p));

function finalizeVisibleParagraphs() {
  // Catches elements still on screen at the moment we check/send —
  // otherwise they'd never get marked "read" since they never scrolled away.
  visibleSince.forEach((shownAt, el) => {
    if (Date.now() - shownAt >= 1500) {
      readParagraphs.add(el);
    }
  });
}

// ── STEP 3: Batch sending — triggered by accumulated content, not a clock ──
const MIN_NEW_CHARS_TO_SEND = 300; // floor: send once unsent content reaches this size
let isTabVisible = !document.hidden;

document.addEventListener("visibilitychange", () => {
  isTabVisible = !document.hidden;
  if (document.hidden) {
    flushRemainingContent(); // user tabbed away — send whatever's left, even if small
  }
});

function checkAndSendNewContent() {
  if (!isTabVisible) return;
  finalizeVisibleParagraphs();

  // Only elements read but not yet sent
  const unsent = Array.from(readParagraphs).filter(p => !sentParagraphs.has(p));
  if (unsent.length === 0) return;

  // Join FULL element text — never sliced or truncated mid-sentence
  const newContent = unsent.map(p => p.innerText.trim()).join("\n\n");

  if (newContent.length >= MIN_NEW_CHARS_TO_SEND) {
    sendCapture(newContent);
    unsent.forEach(p => sentParagraphs.add(p)); // mark sent so we don't resend
  }
}

setInterval(checkAndSendNewContent, 5000); // re-check every 5 seconds

// ── STEP 4: Flush whatever's left when the user leaves the page ───────────
// Covers short pages that never cross the 300-char floor naturally.
function flushRemainingContent() {
  finalizeVisibleParagraphs();

  const unsent = Array.from(readParagraphs).filter(p => !sentParagraphs.has(p));
  if (unsent.length === 0) return;

  const remainingContent = unsent.map(p => p.innerText.trim()).join("\n\n");
  if (remainingContent.length > 0) {
    sendCapture(remainingContent);
    unsent.forEach(p => sentParagraphs.add(p));
  }
}

window.addEventListener("beforeunload", flushRemainingContent);

// ── STEP 5: Send a batch to background.js ──────────────────────────────
function sendCapture(content) {
  const skipDomains = ["google.com/search", "mail.google.com"];
  if (skipDomains.some(d => window.location.href.includes(d))) return;

  console.log("Lore sending batch:", document.title, "-", content.length, "chars");

  chrome.runtime.sendMessage({
    type: "CAPTURE_PAGE",
    data: {
      url: window.location.href,
      title: document.title,
      content: content
    }
  });
}