# Lore

**A private, multi-device semantic memory system.** Lore indexes everything you personally read and write — files, PDFs, web pages, even text baked into images — into a local vector database on your own machine. Ask it a question by voice or text from your phone, and it answers in natural language with the exact source, in seconds, without a single byte leaving hardware you own.

Built for the **Snapdragon Multiverse Hackathon** (Qualcomm India, Bengaluru — July 11–12, 2026).

> "Lore is private by architecture, not by policy — every computation happens on hardware you own."

---

## Table of contents

- [The problem](#the-problem)
- [The solution](#the-solution)
- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Browser extension features](#browser-extension-features)
- [Repository layout](#repository-layout)
- [Tech stack](#tech-stack)
- [Current status](#current-status)
- [Getting started](#getting-started)
- [Acknowledgments](#acknowledgments)

---

## The problem

Every day you run into hundreds of pieces of information worth remembering: a paper you skimmed, a Slack thread, a PDF someone sent you, a paragraph on a webpage you meant to come back to, notes you wrote at 1am and never looked at again. None of it is actually lost — it's sitting somewhere on your machine or in your browser history — but *finding it again* means digging through folders, scrolling tabs, or trying to remember the right search term for something you only half-remember. Search engines can't help, because the thing you're looking for was never public. It was yours.

The usual fix is to hand that problem to a cloud AI assistant — upload your notes, sync your files, let a hosted model index everything. That works, but it means your personal documents, medical notes, financial records, and half-formed ideas all leave your machine and live on someone else's server, subject to someone else's retention policy, someone else's breach, someone else's terms of service.

## The solution

Lore runs quietly in the background on your own PC. It watches the files you save, the pages you actually read (not just visit), and the PDFs and images you spend time on, and turns all of it into searchable meaning — using a semantic embedding model running on the machine's own NPU. Nothing is uploaded anywhere during indexing.

When you want something back, you don't open a search bar and guess at keywords. You just ask, out loud, from your phone. Lore finds the right document by meaning, not exact words, and answers in a sentence — with the source file, its location, and the exact excerpt it came from, so you can verify it instantly rather than trusting a black box.

The moment this is built for: someone asks you a question mid-conversation. You pull out your phone, say it out loud. Three seconds later, the answer is on your screen, pulled from something you wrote or read weeks ago, with its source right there — no Google, no internet connection, nothing indexed anywhere but the laptop in your bag.

## How it works

```
file saved / page read 30s+ / image dwelled on
        │
        ▼
 extract → chunk → embed (NPU) → store in LanceDB          ← happens continuously, in the background
        │
        ⋮
 you speak a question into your phone
        │
        ▼
 transcribe (Sarvam AI) → embed the query → search LanceDB for the nearest chunks
        │
        ▼
 send query + candidates to Cloud AI 100 → rerank → generate a grounded answer
        │
        ▼
 phone shows: the answer, plus every source it came from
```

Indexing and querying are deliberately different machines doing deliberately different jobs — see [Architecture](#architecture) for why that split matters, not just how it's wired.

## Architecture

Three devices, three distinct jobs. Removing any one of them breaks the system — that's a deliberate design constraint, not an accident of how the hackathon hardware was handed out.

```
┌─────────────────────┐        ┌──────────────────────┐        ┌────────────────────────┐
│   Mobile app         │        │  Snapdragon X PC       │        │  Qualcomm Cloud AI 100  │
│   (Kotlin/Compose)   │        │  (the brain)            │        │  (the intelligence)     │
│                      │  LAN  │                          │  LAN   │                          │
│  Voice/text query ───┼───────►│  FastAPI  /query          │───────►│  Rerank + generate       │
│  Sarvam AI STT       │        │  FastAPI  /index           │       │  via Cirrascale's        │
│  Answer + sources ◄──┼────────│  watchdog file watcher      │◄──────│  Imagine SDK             │
│                      │        │  EmbeddingGemma            │       │  (Llama-3.1-8B, etc.)    │
│                      │        │  (ONNX Runtime + QNN EP)   │       │                          │
│                      │        │  LanceDB (vector store)      │       │                          │
└─────────────────────┘        └──────────────────────────────┘       └──────────────────────────┘
                                            ▲
                                            │ POST /index
                                 ┌──────────┴───────────┐
                                 │  Browser extension     │
                                 │  (MV3, captures pages,  │
                                 │  PDFs, and images you   │
                                 │  actually read)         │
                                 └──────────────────────────┘
```

- **PC — the brain.** An always-on background indexer (`pc/indexer/`) plus a local FastAPI server (`pc/api/`). Watches the filesystem, extracts and chunks text, embeds it with EmbeddingGemma 300M on the NPU via ONNX Runtime's QNN execution provider, and stores vectors in LanceDB. This machine never stops indexing while a query is in flight — that's the whole reason the next device exists.
- **Cloud AI 100 — the intelligence.** Query-time reranking and answer generation (`cloud/`), reached over the local network via Cirrascale's Imagine SDK. Running an LLM on the same machine that's continuously embedding new content would stall indexing every time someone searches; splitting the work across two accelerators means neither blocks the other.
- **Mobile app — the interface.** The only thing the user actually touches (`mobile/`). Records a voice query, transcribes it via Sarvam AI, sends it to the PC over WiFi, and renders the answer with its sources. It's the device that's always in your pocket — the PC isn't.
- **Browser extension — the second sense.** A Manifest V3 extension (`extension/`) that feeds actively-read web pages, PDFs, and OCR'd image text into the same indexing pipeline via `POST /index`, so the things you read *outside* your filesystem end up just as searchable as the things you save to it.

## Browser extension features

- **Dwell-time page capture** — `content_script.js` only indexes a page once you've actually read it (30s+ on-page), not just visited it, so tab-hoarding doesn't pollute your knowledge base.
- **In-browser PDF text extraction** — `pdf_capture.js` pulls text straight out of PDFs opened in-browser via `pdf.js`, no download-and-reindex round trip needed.
- **On-device OCR for images** — `image_ocr.js` runs Tesseract.js (via an offscreen document) to pull readable text out of images you dwell on, so screenshots and image-based content are searchable too.
- **Single shared ingestion path** — everything captured (pages, PDFs, OCR'd images) funnels through the same `POST /index` contract as the filesystem indexer, so it lands in the same LanceDB table with no separate pipeline to maintain.
- **Zero third-party upload** — capture, extraction, and OCR all happen locally in the browser/offscreen context; the only network call is to your own PC's FastAPI server on the LAN.

## Repository layout

```
lore/
├── pc/                      Python — indexing engine + FastAPI backend
│   ├── indexer/             watcher, extractor, chunker, embedder, vector_store, profiler
│   ├── api/                 FastAPI app, routes, schemas, logging, cloud_client
│   └── scripts/             CLI entry points + NPU quantization/export tooling
├── cloud/                   Cloud AI 100 client: reranking + LLM generation via Imagine SDK
├── mobile/                  Native Android app (Kotlin + Jetpack Compose)
├── extension/               Chrome MV3 extension (text/PDF/image capture)
├── tests/                   Python test suite (pytest)
├── scratch/                 One-off NPU/ONNX debugging scripts (not part of the pipeline)
├── CLAUDE.md                Full project brief and phase-by-phase spec
├── SNAPDRAGON_PC_SETUP.md   From-scratch setup guide for the real ARM64 hardware
└── README.md                You are here
```

## Tech stack

Python (FastAPI, ONNX Runtime + QNN, LanceDB) on the PC; Kotlin + Jetpack Compose on Android; a Manifest V3 browser extension (pdf.js, Tesseract.js) for web/PDF/image capture; Cirrascale's Imagine SDK for Cloud AI 100 inference; Sarvam AI for speech-to-text. Tested with pytest and Node's built-in test runner.

## Current status

Honest snapshot of what's real versus what's aspirational, as of this writing:

**Working end-to-end, verified against real hardware/services:**
- Full request path: mobile app → WiFi → FastAPI → LanceDB search → real Cloud AI 100 (`Llama-3.1-8B` via Imagine SDK) → answer + sources rendered on-device, verified on a physical Android phone.
- Real QNN Execution Provider registration in `embedder.py`, and a real Gemma tokenizer path (`sentencepiece`) that auto-detects whether the loaded ONNX model expects tokenized input or a flat vector.
- Browser extension's text/PDF/image capture pipeline, all funneling through the same `/index` contract.
- Duplicate/stale-row prevention on re-indexing, and search history that's actually persisted on-device (not mock data).

**Known gaps:**
- `sentencepiece`/`transformers`/`torch` aren't yet declared in `pc/requirements*.txt` — install manually until that's added.
- The pre-exported EmbeddingGemma ONNX model needs re-exporting with standard ops (`pc/scripts/export_vanilla_onnx.py`) before QNN will actually use the NPU instead of silently falling back to CPU — validated on CPU so far, real on-NPU testing is still pending on the Snapdragon hardware.
- No authentication on the FastAPI endpoints — fine for a local-WiFi-only demo, not for anything beyond that.
- Image indexing via `jina-clip` is a placeholder schema only, deferred to a future phase.

## Getting started

Full step-by-step setup (Python env, model quantization, Cloud AI 100 wiring, mobile build, browser extension) lives in [`SNAPDRAGON_PC_SETUP.md`](SNAPDRAGON_PC_SETUP.md) — that's the from-scratch guide for the real ARM64 Snapdragon hardware, and covers everything needed to run each component elsewhere too. [`CLAUDE.md`](CLAUDE.md) has the full project brief if you want the complete phase-by-phase spec this was built against.

## Acknowledgments

Built at the Snapdragon Multiverse Hackathon by Qualcomm India. Sarvam AI (Speech-to-Text) and OnePlus (mobile hardware) are the event's official partners; indexing and embedding run on Snapdragon X NPU hardware via ONNX Runtime's QNN execution provider, and query-time inference runs on Qualcomm Cloud AI 100 via Cirrascale's Imagine SDK.
