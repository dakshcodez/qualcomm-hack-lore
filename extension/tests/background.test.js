"use strict";

// Verifies the Chrome extension -> PC /index integration wired up in
// background.js: the fetch target must be the real POST /index route, and
// the outgoing body must match the backend's IndexRequest schema
// ({text, url, title}), regardless of which capture site (normal page,
// PDF, image OCR) produced the {url, title, content} shape.
//
// background.js is a MV3 service worker script (no exports, uses
// importScripts + global `chrome`), so it's loaded into a vm sandbox with
// those globals stubbed rather than required as a module.

const assert = require("node:assert/strict");
const { test } = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function loadBackgroundScript({ fetchImpl }) {
  const capturedListeners = [];

  const sandbox = {
    console,
    importScripts: () => {}, // offscreen_manager.js/pdf_capture.js/image_ocr_bridge.js not under test here
    chrome: {
      runtime: {
        onMessage: {
          addListener: (fn) => capturedListeners.push(fn),
        },
      },
    },
    fetch: fetchImpl,
  };
  vm.createContext(sandbox);

  const source = fs.readFileSync(path.join(__dirname, "..", "background.js"), "utf8");
  vm.runInContext(source, sandbox, { filename: "background.js" });

  assert.equal(capturedListeners.length, 1, "expected background.js to register exactly one onMessage listener");
  return { onMessage: capturedListeners[0], sandbox };
}

function fakeFetch(calls, { ok = true } = {}) {
  return async (url, options) => {
    calls.push({ url, options });
    return { ok };
  };
}

test("sends captured page content to the real /index endpoint", async () => {
  const calls = [];
  const { onMessage } = loadBackgroundScript({ fetchImpl: fakeFetch(calls) });

  onMessage({
    type: "CAPTURE_PAGE",
    data: { url: "https://example.com/article", title: "Article Title", content: "some page text" },
  });

  // sendToLore is async and not awaited by the listener; let its microtask run.
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://localhost:8000/index");
});

test("maps {url, title, content} onto the backend's {text, url, title} schema", async () => {
  const calls = [];
  const { onMessage } = loadBackgroundScript({ fetchImpl: fakeFetch(calls) });

  onMessage({
    type: "CAPTURE_PAGE",
    data: { url: "https://example.com/page", title: "Page Title", content: "the extracted body text" },
  });
  await new Promise((resolve) => setImmediate(resolve));

  const body = JSON.parse(calls[0].options.body);
  assert.deepEqual(body, {
    text: "the extracted body text",
    url: "https://example.com/page",
    title: "Page Title",
  });
  assert.equal(calls[0].options.method, "POST");
  assert.equal(calls[0].options.headers["Content-Type"], "application/json");
});

test("PDF and image-OCR capture paths funnel through the same mapping (no 'content' field leaks through)", async () => {
  const calls = [];
  const { onMessage } = loadBackgroundScript({ fetchImpl: fakeFetch(calls) });

  // pdf_capture.js's sendCapture() and image_ocr.js's sendCapture() both
  // post this exact {type: CAPTURE_PAGE, data: {url, title, content}} shape.
  onMessage({
    type: "CAPTURE_PAGE",
    data: { url: "https://example.com/doc.pdf", title: "doc.pdf", content: "pdf extracted text" },
  });
  await new Promise((resolve) => setImmediate(resolve));

  const body = JSON.parse(calls[0].options.body);
  assert.ok(!("content" in body), "raw 'content' field must not leak into the request body");
  assert.equal(body.text, "pdf extracted text");
});

test("ignores messages that are not CAPTURE_PAGE", async () => {
  const calls = [];
  const { onMessage } = loadBackgroundScript({ fetchImpl: fakeFetch(calls) });

  onMessage({ type: "SOME_OTHER_MESSAGE", data: {} });
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(calls.length, 0);
});

test("a failed fetch (PC unreachable) does not throw unhandled", async () => {
  const { onMessage } = loadBackgroundScript({
    fetchImpl: async () => {
      throw new Error("network error");
    },
  });

  assert.doesNotThrow(() => {
    onMessage({
      type: "CAPTURE_PAGE",
      data: { url: "https://example.com", title: "T", content: "some content" },
    });
  });
  await new Promise((resolve) => setImmediate(resolve));
});
