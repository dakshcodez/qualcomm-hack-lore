# Running Lore on the Snapdragon X PC

This walks through setting up and running the indexer (`pc/indexer/`) and
backend API (`pc/api/`) from a completely fresh Snapdragon X Series
Copilot+ PC that has nothing on it but `git`. Everything below is
PowerShell on Windows.

Read **"Known gap: real embeddings aren't wired up yet"** before you get
to the real-model section — it'll save you a confusing crash.

---

## 0. Clone the repo (if you haven't yet)

```powershell
git clone <your-repo-url>
cd qualcomm-hack-lore
```

## 1. Install Python — must be the ARM64-native build

This is the single most important gotcha on this hardware: if Python is
running under x64 emulation (Prism), the QNN execution provider will
never load, and onnxruntime will silently run everything on CPU. Windows
11 on a Copilot+ PC ships `winget`, so:

```powershell
winget install -e --id Python.Python.3.12
```

Close and reopen PowerShell after installing so `PATH` picks it up.

**Verify you actually got the ARM64 build:**

```powershell
python -c "import platform; print(platform.machine())"
```

This must print `ARM64`. If it prints `AMD64`, you installed (or PATH is
pointing at) the x64 build — uninstall it and reinstall specifically the
ARM64 Python from the Microsoft Store or python.org's ARM64 installer.

## 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Your prompt should now show `(.venv)`.

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r pc\requirements-dev.txt
```

This installs everything in `pc\requirements.txt` (watchdog, PyMuPDF,
python-docx/pptx, openpyxl, lancedb, onnxruntime, fastapi, uvicorn, ...)
plus `pytest`/`httpx` for running the test suite. `pc\requirements.txt`
deliberately does **not** include `onnxruntime-qnn` (it's an ARM64-only
wheel, uninstallable anywhere but here) — install it separately, and swap
out the plain CPU `onnxruntime` package first since the two provide the
same `onnxruntime` import and will conflict if both are installed:

```powershell
pip uninstall onnxruntime -y
pip install onnxruntime-qnn
```

If `pip install onnxruntime-qnn` fails to find a wheel for your Python
version, try Python 3.11 instead of 3.12 (repeat step 1/2 with
`Python.Python.3.11`) — QNN wheel availability lags behind new CPython
releases.

**Sanity check the provider is even visible to onnxruntime:**

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

You want to see `'QNNExecutionProvider'` in that list. If you only see
`CPUExecutionProvider`, stop here and fix the install before continuing —
nothing downstream will use the NPU otherwise.

## 4. Run the test suite as a baseline sanity check

Everything in this repo except the real embedding model and the Cloud AI
100 hardware call is pure Python + onnxruntime + lancedb with no
platform-specific behavior, so the full suite should pass identically on
this Windows ARM64 machine:

```powershell
pytest
```

You should see all tests pass (144 at the time of writing). If something
fails here, fix it before moving on — everything past this point builds
on this working.

---

## 5. Quick end-to-end smoke check (no real model needed yet)

Before dealing with the real EmbeddingGemma model, validate the whole
pipeline — including whether the NPU actually gets used — with a tiny
synthetic model:

```powershell
mkdir models -Force
python pc\scripts\build_smoke_model.py --output models\smoke.onnx
```

Run the indexer against it in one terminal:

```powershell
$env:EMBEDDING_MODEL_PATH = "$PWD\models\smoke.onnx"
$env:LANCEDB_PATH = "$PWD\lancedb"
$env:EMBEDDING_DIM = "768"
$env:WATCH_ROOT = "$env:USERPROFILE\Documents"
$env:LOG_FILE = "$PWD\lore.log"
python pc\scripts\run_indexer.py
```

You should see `Watching C:\Users\<you>\Documents for changes...`. Drop a
`.txt` file into that folder from a different terminal/Explorer and
confirm the process logs an "Indexed N chunk(s) from ..." line.

**Environment variables don't persist across terminal windows** — every
new PowerShell window needs the same five `$env:` lines set again (or set
them once under System Properties → Environment Variables for a
persistent setup).

Immediately after starting, check `lore.log` for the provider that was
actually selected:

```powershell
Select-String "active provider" lore.log
```

You want `active provider=QNNExecutionProvider`. If it says
`CPUExecutionProvider`, see the troubleshooting section — the pipeline
still works, it's just not using the NPU.

Stop the indexer with `Ctrl+C` when done.

---

## 6. Run the API

In a terminal with the same five env vars set (point `LANCEDB_PATH` at
the same directory as the indexer so they share data):

```powershell
uvicorn pc.api.main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is required so the phone can reach it over WiFi — `127.0.0.1`
only accepts connections from the PC itself. Windows Firewall will prompt
the first time; allow it for **Private networks**.

**Sanity check with curl** (use `curl.exe` explicitly — plain `curl` is
aliased to `Invoke-WebRequest` in Windows PowerShell 5.1 and doesn't
accept the same flags):

```powershell
curl.exe http://localhost:8000/health

curl.exe -X POST http://localhost:8000/index -H "Content-Type: application/json" -d '{\"text\": \"neural embeddings research paper\", \"url\": \"https://example.com/a\", \"title\": \"Test Doc\"}'

curl.exe -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{\"text\": \"neural embeddings\", \"modality\": \"text\"}'
```

(Note the escaped `\"` — PowerShell's quoting rules for `-d` differ from
bash's; if that's fiddly, use `Invoke-RestMethod` instead:
`Invoke-RestMethod -Method Post -Uri http://localhost:8000/query -ContentType "application/json" -Body '{"text": "neural embeddings"}'`.)

You should get back `{"status":"ok"}` from `/health` and a JSON
answer/sources body from `/query`.

**Find the PC's local IP** (to hardcode in the mobile app, per CLAUDE.md):

```powershell
ipconfig
```

Look for the `IPv4 Address` under your active WiFi adapter.

---

## 7. Known gap: real embeddings aren't wired up yet

Read this before spending time on the real EmbeddingGemma model.

`pc/indexer/embedder.py`'s NPU→GPU→CPU provider selection and fallback
logic is real and works against any ONNX model with a single float32
input tensor. But the text-to-tensor step (`_default_preprocess`, a
bag-of-hashed-words function) is an explicitly-labeled placeholder, not
the real EmbeddingGemma tokenizer — it exists only so the pipeline has
something to test against without downloading gated model weights or a
tokenizer dependency.

**The real EmbeddingGemma ONNX export will not have the same input
signature** (it'll take tokenized `input_ids`/`attention_mask` integer
tensors, not one flat float32 vector), so pointing `EMBEDDING_MODEL_PATH`
at a real exported model as-is will fail or silently produce meaningless
output. Before real semantic search works, someone needs to:

1. Export EmbeddingGemma 300M to ONNX (see step 8 below).
2. Write a real `preprocess_fn` for `Embedder` (see `pc/indexer/embedder.py`'s
   `preprocess_fn` constructor argument) that tokenizes text with the
   actual Gemma tokenizer and feeds whatever input tensors the real
   export actually expects.

Until that's done, the smoke-test model from step 5 is the only
known-good way to validate the pipeline on this hardware — it proves the
NPU/GPU/CPU selection, LanceDB writes, and API wiring all work; it does
not prove real search quality.

## 8. Getting and quantizing the real model (best effort — verify as you go)

This part I can't fully verify without live access to Hugging Face, so
treat it as a starting point, not a copy-paste guarantee:

```powershell
pip install optimum[exporters]
optimum-cli export onnx --model google/embeddinggemma-300m models\embeddinggemma_fp32
```

Search Hugging Face for "EmbeddingGemma 300M" by Google to confirm the
exact repo id and check its model card for any export caveats — some
architectures need extra `optimum-cli` flags or aren't supported by the
default exporter yet.

Once you have a real fp32 ONNX export **and** a real `preprocess_fn`
producing calibration inputs matching that model's actual input
signature (see step 7), quantize it for QNN with the script already in
this repo:

```powershell
python pc\scripts\quantize_embedding_model.py `
  --input models\embeddinggemma_fp32\model.onnx `
  --calibration-data models\calibration_data `
  --output models\embedding_gemma_300m_qnn_int8.onnx
```

`--calibration-data` is a directory of `.npy` files, each a precomputed
input array matching the model's real input tensor shape — see the
docstring at the top of `pc/scripts/quantize_embedding_model.py` for the
exact contract.

---

## 9. Monitoring NPU usage

Three ways to check the NPU is actually doing the work, from quick to thorough:

**a) Windows Task Manager** — Performance tab → NPU. On a Copilot+ PC
this shows live NPU utilization the same way it shows CPU/GPU. Watch it
while indexing a batch of files or issuing several `/query` requests —
you should see it spike.

**b) The app's own logs** — `Embedder.__init__` logs the active provider
once at startup, and `InferenceProfiler` logs a `WARNING` any time a call
silently falls back off the top-preference provider:

```powershell
Select-String "active provider" lore.log
Select-String "fell back" lore.log
```

No "fell back" lines and `active provider=QNNExecutionProvider` means
every embedding call in that session ran on the NPU.

**c) A quick Python one-liner for a numeric breakdown** — once you have a
running `Embedder` instance (e.g. in a REPL, or add a temporary print in
`run_indexer.py`), `embedder.profiler.summary()` gives you
`{"backends": {"NPU": {"count": N, "percentage": X}, ...}, "any_fallback": bool, "avg_latency_ms": ...}`
for everything embedded in that session — the same data the demo can
cite if a judge asks about NPU usage.

---

## 10. Troubleshooting

- **QNN never appears in `get_available_providers()`** — you likely
  still have both `onnxruntime` and `onnxruntime-qnn` installed (they
  conflict), or Python is the x64 build under emulation. Re-check steps
  1 and 3.
- **`active provider=CPUExecutionProvider` even though QNN is
  available** — the ONNX model itself may not be QNN-compatible
  (unquantized fp32 models can fail to load on some QNN EP configs even
  if CPU-runnable). Confirm you're pointing at a quantized model, or fall
  back to the smoke-test model to isolate whether it's a model problem or
  an install problem.
- **`.venv\Scripts\Activate.ps1` refuses to run** — execution policy;
  see step 2's bypass command.
- **Phone can't reach the API** — confirm both devices are on the same
  WiFi (or the PC's hotspot, per CLAUDE.md's setup notes), that uvicorn
  was started with `--host 0.0.0.0` not `127.0.0.1`, and that you allowed
  the Windows Firewall prompt for Private networks.
- **LanceDB "already locked" / weird errors** — don't point two
  processes (e.g. the indexer and a manual test script) at the same
  `LANCEDB_PATH` while both are writing; stop one before starting the other.
- **`pip install onnxruntime-qnn` can't find a matching wheel** — try a
  different Python 3.x minor version (3.11 is a safe bet); QNN wheel
  support for brand-new CPython releases can lag.
