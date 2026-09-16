# How to test ABREngine

A step-by-step guide to verifying everything yourself, from a clean checkout.
Nothing here requires knowledge of the internals.

Each section says **what you run**, **what you should see**, and **what it proves**.

---

## Contents

- [0. What you need installed](#0-what-you-need-installed)
- [1. One-time setup](#1-one-time-setup)
- [2. The five-minute check](#2-the-five-minute-check)
- [3. Full verification, step by step](#3-full-verification-step-by-step)
  - [3.1 Python side: export the model](#31-python-side-export-the-model-from-the-pt-checkpoint)
  - [3.2 Python side: PyTorch vs ONNX](#32-python-side-does-onnx-agree-with-pytorch)
  - [3.3 JavaScript: typecheck, lint, build](#33-javascript-typecheck-lint-and-build)
  - [3.4 JavaScript: the test suite](#34-javascript-the-test-suite)
  - [3.5 The browser end-to-end test](#35-the-browser-end-to-end-test)
  - [3.6 Performance](#36-performance)
- [4. Testing by hand in a browser](#4-testing-by-hand-in-a-browser)
- [5. Testing against your own stream](#5-testing-against-your-own-stream)
- [6. Testing in your own application](#6-testing-in-your-own-application)
- [7. What to look for — reading the results](#7-what-to-look-for--reading-the-results)
- [8. Troubleshooting](#8-troubleshooting)
- [9. Test inventory](#9-test-inventory)

---

## 0. What you need installed

| Tool | Version | Needed for | Check with |
|---|---|---|---|
| **Node.js** | 18 or newer | everything JavaScript | `node --version` |
| **npm** | 9 or newer | dependencies | `npm --version` |
| **Python** | 3.9 or newer | re-exporting the model | `python --version` |
| **ffmpeg** | any recent | generating a local test stream | `ffmpeg -version` |
| **Google Chrome** | any recent | the browser end-to-end test | — |

**You do not need Python at all** unless you want to re-export the model or
re-generate the test fixtures. The `.onnx` file and the fixtures are committed,
so the JavaScript tests run standalone.

Installing ffmpeg:

```bash
# macOS
brew install ffmpeg
# Debian / Ubuntu
sudo apt-get install -y ffmpeg
# Windows
winget install ffmpeg
```

---

## 1. One-time setup

```bash
git clone <your-repo-url> abrengine-rl
cd abrengine-rl

# JavaScript dependencies (workspace root installs the package too)
npm install

# A browser for the automated end-to-end test
npx playwright install chrome
```

> **Why `chrome` and not `chromium`?** Playwright's bundled Chromium is built
> without proprietary codecs, and videojs-http-streaming refuses to load at all
> unless the browser can decode H.264+AAC in MediaSource. `npx playwright install
> chrome` fetches real Google Chrome, which can. See
> [§8](#8-troubleshooting) if you cannot install it.

Optional — only if you want to touch the Python side:

```bash
pip install -r requirements.txt
pip install onnx onnxruntime
```

---

## 2. The five-minute check

If you only run one thing, run this:

```bash
npm run build && npm test
```

**What you should see** — the last few lines:

```
 Test Files  9 passed (9)
      Tests  177 passed (177)
```

and, in the middle of the output:

```
parity over 296 fixtures: maxΔprob=7.749e-7 maxΔvalue=1.563e-2 maxRelValue=4.860e-7
```

**What it proves.** The JavaScript package builds, and the ONNX model running
under JavaScript reproduces the PyTorch checkpoint's output on 296 recorded
states — with the **same argmax on every single one**. That is the claim that
this package runs *your* controller and not an approximation of it.

---

## 3. Full verification, step by step

### 3.1 Python side: export the model from the `.pt` checkpoint

Skip this unless you changed the checkpoint or the research code — the outputs
are committed.

```bash
python export/export_onnx.py
```

**What you should see:**

```
✓ ONNX model    → packages/abrengine/models/ac3-controller.onnx  (1045.8 KiB)
✓ Manifest      → packages/abrengine/models/ac3-controller.json
  op types: Concat, Constant, Conv, Flatten, Gather, Gemm, Relu, Slice, Softmax, Unsqueeze
```

**What it proves.** The `.pt` checkpoint exports to ONNX cleanly at opset 17, and
every operator in the graph is a core ONNX op that ONNX Runtime Web supports on
its WASM backend. No custom ops, no unsupported PyTorch constructs.

Then regenerate the test fixtures:

```bash
python export/make_fixtures.py        # 296 (state → PyTorch output) pairs
python export/make_state_fixtures.py  # 98 StateBuilder input/output steps
```

Or all three at once: `npm run export-model`.

---

### 3.2 Python side: does ONNX agree with PyTorch?

```bash
python export/validate_parity.py
```

**What you should see:**

```
fixtures                 : 296
fixture vs live PyTorch  : max |Δp| = 0.000e+00
batch vs per-sample ONNX : max |Δp| = 0.000e+00
PyTorch vs ONNX probs    : max |Δ| = 7.749e-07  mean |Δ| = 8.914e-09  max rel = 5.445e-06
PyTorch vs ONNX value    : max |Δ| = 1.562e-02  mean |Δ| = 7.409e-05  max rel = 5.811e-07
argmax agreement         : 296/296
PASS
```

**What it proves.** The conversion is numerically faithful in Python, before
JavaScript is involved at all. If this fails, the problem is in the export, not
in the JS package.

> **About that `1.562e-02`.** It is one deliberately out-of-distribution fixture
> where the critic outputs V(s) ≈ 189 596. One float32 ULP at that magnitude is
> already ~1.6e-2, so the *relative* error of 5.8e-7 is the meaningful number.
> The tolerance is `atol + rtol·|value|` for exactly this reason. The **policy**
> output — the thing that picks the bitrate — agrees to 7.7e-7.

---

### 3.3 JavaScript: typecheck, lint, and build

```bash
npm run typecheck   # tsc --noEmit, strict mode
npm run lint        # eslint
npm run build       # tsup → dist/
```

**What you should see:** no output from the first two (silence is success), and
a size table from the build:

```
ESM dist/index.js      1.66 KB
ESM dist/videojs.js   21.66 KB
CJS dist/index.cjs     7.23 KB
CJS dist/videojs.cjs  20.01 KB
DTS dist/index.d.ts   17.66 KB
```

**What it proves.** The public API is fully typed with no `any` leaking into
declarations, and the package emits working ESM, CommonJS and `.d.ts` outputs.

---

### 3.4 JavaScript: the test suite

```bash
npm test
```

Or one file at a time, from `packages/abrengine`:

```bash
cd packages/abrengine
npx vitest run test/parity.test.ts          # PyTorch → ONNX → JS
npx vitest run test/state-builder.test.ts   # the feature-building port
npx vitest run test/ladder.test.ts          # action ↔ rendition mapping
npx vitest run test/validate.test.ts        # input validation and repair
npx vitest run test/engine.test.ts          # engine behaviour and fallbacks
npx vitest run test/adapter-videojs.test.ts # the Video.js adapter, mocked player
npx vitest run test/integration.test.ts     # real model + mocked player
npx vitest run test/manifest.test.ts        # constants have not drifted
npx vitest run test/reward.test.ts          # the QoE metric
```

Two of these deserve individual attention.

**`state-builder.test.ts`** replays the exact `update()` sequences that the
Python `StateBuilder` received during real simulator rollouts, and asserts the
resulting 6×8 tensor matches Python **exactly** — not within a tolerance, bit for
bit, across 98 steps. This is what catches a flipped shift direction, a wrong
divisor, or a row written at the wrong index. Those would all sail past the ONNX
parity test, because that test starts from a state tensor it is *handed*.

**`integration.test.ts`** drives the real ONNX model through simulated streaming
sessions. One of its tests is named
`holds a high rung even on a collapsing link (known model limitation)` — that is
deliberate, and [§7](#7-what-to-look-for--reading-the-results) explains it.

---

### 3.5 The browser end-to-end test

This is the one that backs the claim "it works with Video.js".

```bash
# Generate a real 6-rendition HLS stream (about 30 s, needs ffmpeg)
npm run make-stream

# Run it
npm run test:browser
```

**What you should see:**

```
── model + adapter bring-up ──
  ✓ ONNX model loads in the browser — onnxruntime-web
  ✓ engine reaches ready — ready
  ✓ model load time is reasonable — 1181ms
  ✓ adapter attached to the real VhsHandler — {"hasPlaylistController":true,…}
  ✓ adapter reads per-segment stats from the main segment loader

── playback ──
  ✓ video actually played — t=14.0s
  ✓ no player errors
  ✓ the trained model produced decisions — 3/4 decisions from the model
  ✓ observations carry real measured segment stats — 949562B in 0.246s
  ✓ inference latency is playback-safe — p50=0.90ms p95=1.00ms
  ✓ selectPlaylist is served from the decision cache
  ✓ 500 synchronous selectPlaylist calls trigger zero inferences
  ✓ VHS applied the engine decisions — 3 applied
  ✓ VHS switched rendition under the model (not the default ABR)
      — renditions played: v2@1264000 -> v4@2914000 -> v1@814000

── browser vs Node numerical parity ──
  ✓ browser inference matches Node inference bit-for-bit — max |Δp| = 0.000e+0

── fallback ──
  ✓ playback continues after the model starts failing — t 0.0s → 5.0s
  ✓ the failure is reported, not swallowed — inference-error ×5
  ✓ decisions fall back to the player default
  ✓ selectPlaylist keeps answering and delegates to the player default
  ✓ still no player errors
  ✓ no uncaught console errors

23/23 checks passed
END-TO-END PASS
```

**What it proves.** Nothing in that run is mocked. Real Chromium, real Video.js
8.24 and VHS 3.17.5 from `node_modules`, real MediaSource appends, real video
decoded and rendered, the real exported ONNX model, and the built package from
`dist/`. The line that matters most is the rendition sequence: VHS genuinely
switched quality levels **because the trained controller told it to**.

To watch it happen, run it headed:

```bash
PWDEBUG=1 npm run test:browser
```

The three environment variables it understands:

| Variable | Effect |
|---|---|
| `CHROMIUM_PATH` | Path to a specific browser binary |
| `CHROMIUM_NO_SANDBOX=1` | Adds `--no-sandbox` (needed in some containers) |
| `PWDEBUG=1` | Runs headed with the Playwright inspector |

---

### 3.6 Performance

```bash
npm run bench
```

**What you should see** (numbers vary by machine):

```
Model
  ac3-controller.onnx        1045.8 KiB
  gzipped                     967.1 KiB
  parameters                 265,863 float32

Built bundles (the package only; runtimes are peer deps)
  index.js                      1.7 KiB   gz 0.8 KiB
  videojs.js                   21.7 KiB   gz 6.1 KiB
  shared chunks ( 2)           42.3 KiB   gz 12.3 KiB

Session load (cold, onnxruntime-web wasm, 1 thread)
  first (includes wasm init): 309ms

Inference latency over 3000 decisions
  session.run   p50 0.151ms  p95 0.247ms  p99 1.671ms
  decide()      p50 0.169ms  p95 0.278ms  p99 2.151ms
  adapter overhead (decide − run): p50 0.018ms

Duty cycle
  one decision per 4s segment at p50 0.169ms
  = 0.0042% of wall-clock time spent in ABR
```

**What it proves.** ABR decisions cost about 0.17 ms and happen once per ~4-second
segment. That is roughly four thousandths of one percent of playback time. The
model is not a performance concern.

---

## 4. Testing by hand in a browser

```bash
npm run make-stream   # once, if you have not already
npm run build
npm run example
```

Then open **http://localhost:8080**. Four pages:

### `vanilla-videojs` — does it work at all?

The whole integration is six lines. Press play.

**What to check:**
1. The status line reads `engine ready via onnxruntime-web`.
2. The log shows `→ model selected v…` lines appearing as segments download.
3. Press **Disable custom ABR**. Log says the Video.js default is back in charge.
4. Press it again to re-enable. Decisions resume.

Playback must never stutter or error in any of those states.

### `telemetry` — what is the model actually thinking?

The most useful page. It shows, per decision:

- the full 6×8 state tensor with each row labelled and its units
- the softmax over all six actions, as a bar chart, with the argmax marked
- the critic's V(s)
- inference latency, running p50/p95

**What to check:** the throughput row should track your real network, the buffer
row should track the player's buffer ÷ 10, and the argmax should be the action
with the tallest bar. If the state tensor is full of zeros after the first few
segments, something upstream is not reporting segment stats.

### `fallback` — does it fail safely?

Four buttons. The important one is **Break inference**, which replaces the live
ONNX session with one that throws.

**What to check, in order:**
1. Press **Break inference**.
2. Press **Force a decision tick** a few times.
3. The log must show `ERROR inference-error: …` — the failure is *reported*.
4. The log must show `decision player-default(inference-error) → [player default]`.
5. **The video must keep playing.** No stutter, no error banner.
6. Press **Repair**, then **Force a decision tick**. Model decisions resume.

Also press **Feed an invalid observation** — NaN buffer, NaN bitrate, negative
rebuffer. It must return a fallback rather than throwing.

### `custom-model` — can I use my own model?

Shows the four ways to supply a controller. The file picker swaps in any `.onnx`
you point it at, live, without a page reload.

---

## 5. Testing against your own stream

Edit any example and change one line:

```js
player.src({ src: 'https://your-cdn.example/master.m3u8', type: 'application/x-mpegURL' });
```

For DASH, add `videojs-contrib-dash` or use a `.mpd` source that VHS handles, and
set `type: 'application/dash+xml'`.

To exercise the controller properly, throttle the network while it plays:
Chrome DevTools → **Network** → throttling dropdown → *Slow 4G*, then back to
*No throttling*. Watch the `telemetry` page: the throughput row should collapse
and recover, and you should see the policy react.

> Be aware of the model limitation in [§7](#7-what-to-look-for--reading-the-results)
> before drawing conclusions from what you see here.

---

## 6. Testing in your own application

```bash
cd packages/abrengine
npm run build
npm pack                       # → abrengine-0.1.0.tgz
```

Then in your app:

```bash
npm install /path/to/abrengine-0.1.0.tgz
npm install onnxruntime-web    # peer dependency
```

```js
import videojs from 'video.js';
import { AbrEngine } from 'abrengine';
import { VideoJSAbrAdapter } from 'abrengine/videojs';

const player = videojs('video');
const abr = new AbrEngine({ model: 'ac3', telemetry: true });
const adapter = new VideoJSAbrAdapter({ player, abr });
await adapter.initialize();

player.src({ src: '/your/stream.m3u8', type: 'application/x-mpegURL' });

abr.on('decision', (e) => console.log(e.decision.source, e.decision.representationId));
```

**Checklist for your app:**

- [ ] `abr.status === 'ready'` after `initialize()`
- [ ] `adapter.active === true` once a source is loaded
- [ ] `decision` events arrive roughly once per segment, not 4× per second
- [ ] `decision.source === 'model'` (not `'fallback'`)
- [ ] The Network tab shows `ac3-controller.onnx` fetched once, then cached
- [ ] Quality visibly changes when you throttle the network

If decisions are all `fallback`, listen for `error` — the `reason` field names
the problem exactly.

---

## 7. What to look for — reading the results

### The numbers that matter

| Measurement | Expected | Where |
|---|---|---|
| PyTorch vs ONNX, policy | max abs Δ < 1e-5 | `validate_parity.py`, `parity.test.ts` |
| argmax agreement | 296 / 296 | same |
| StateBuilder port | bit-exact, 98 steps | `state-builder.test.ts` |
| Browser vs Node | identical | `test:browser` |
| Inference latency | p95 under a few ms | `bench`, `test:browser` |
| Player errors | zero, always | `test:browser`, the examples |

### The limitation you need to know about

**The bundled `ac3` controller does not reliably back off when the network
collapses.**

Measured over a 56-point sweep of (throughput × buffer) states, its argmax is
action 4 — 2850 kbps on the training ladder — in **51 of them**, including states
with 0.1 Mbps of measured throughput and an empty buffer. Driven from 8 Mbps down
to 0.67 Mbps with the buffer draining to zero, it returns to 2850 kbps within two
steps and holds there with 98.5 % confidence.

**This is the trained model, not the JavaScript port.** Feed the identical state
to `checkpoints/abrengine_final.pt` in PyTorch and you get the identical argmax.
The parity suite agrees to 7.7e-7 across 296 fixtures. It is also consistent with
the evaluation in the research README: 1.89 s of rebuffering per episode against
the buffer-based heuristic's 0.43 s, and a policy described there as "locked at
2850 kbps for the entire episode".

Verify it yourself:

```bash
cd packages/abrengine
npx vitest run test/integration.test.ts -t "collapsing link"
```

**What to do about it.** There is an opt-in guardrail, off by default because
enabling it overrides the trained policy's action:

```js
const abr = new AbrEngine({
  model: 'ac3',
  safety: {
    enabled: true,          // default false
    bufferFloorSec: 8,      // only intervene below this buffer level
    throughputFactor: 0.9,  // cap at 0.9 × measured throughput
  },
});
```

Every intervention appears as `decision.safetyClamp`, so you can measure how
often the model is being overridden rather than having it hidden:

```js
let clamped = 0, total = 0;
abr.on('decision', (e) => {
  total++;
  if (e.decision.safetyClamp) clamped++;
});
// If clamped/total is high, the model is not contributing much.
```

The longer-term fix is retraining — on real network traces rather than the
synthetic generator in `src/env.py`, and with a rebuffer penalty tuned so the
policy learns to descend. The package is ready for that: re-run
`npm run export-model` and everything downstream picks up the new weights, with
the manifest test failing loudly if your normalisation constants changed.

---

## 8. Troubleshooting

### `npm run test:browser` fails at browser launch

```
browserType.launch: Executable doesn't exist at …
```

Run `npx playwright install chrome`. If your network blocks Google's download
host, point at a browser you already have:

```bash
CHROMIUM_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" npm run test:browser
# Linux
CHROMIUM_PATH=/usr/bin/google-chrome npm run test:browser
```

### `MEDIA_ERR_SRC_NOT_SUPPORTED` / "No compatible source was found"

Your browser cannot decode H.264 + AAC in MediaSource, and
videojs-http-streaming refuses to register its source handler at all in that
case — regardless of what codecs your stream actually uses:

```js
// @videojs/http-streaming, videojs-http-streaming.cjs.js
const supportsNativeMediaSources = () =>
  browserSupportsCodec('avc1.4d400d,mp4a.40.2', true);
if (supportsNativeMediaSources()) {
  videojs.getTech('Html5').registerSourceHandler(VhsSourceHandler, 0);
}
```

This affects Playwright's bundled Chromium and most Linux distro `chromium`
packages. Two ways out:

1. **Use a browser with the codecs** — real Chrome, Edge, Firefox or Safari.
2. **Use the VP9 stream and the codec shim** the examples ship:

   ```bash
   npm run make-stream -- ./test-e2e/stream vp9
   ```

   `examples/codec-shim.js` answers `true` for that one probe string so VHS
   registers, and nothing else is faked — MediaSource really appends the
   segments and the browser really decodes them. In a browser that has H.264 the
   shim is a no-op. You do **not** need it in your own application.

### No `decision` events at all

Most likely the per-segment tick is not arriving. VHS raises `bandwidthupdate`
on the **tech**, not the player, and Video.js does not forward it — the adapter
subscribes to both plus the segment loader directly. Check:

```js
console.log(adapter.active);        // must be true
console.log(adapter.counters);      // observations / decisions should climb
player.tech(true).trigger('bandwidthupdate');   // force one
```

If `adapter.active` is false, the adapter has not found a VhsHandler — either the
player has no source yet, or the source is not going through VHS.

### `No ONNX runtime available`

```bash
npm install onnxruntime-web
```

It is an optional peer dependency, deliberately not bundled — it is several MB
and many applications already have their own copy.

### The model 404s

`model: 'ac3'` resolves through `new URL('../models/…', import.meta.url)`. Vite,
webpack 5, Rollup and Node handle that; some setups do not. Serve the file
yourself and be explicit:

```js
new AbrEngine({ model: { type: 'url', url: '/models/ac3-controller.onnx' } })
```

### Decisions are all `fallback`

Listen for the reason:

```js
abr.on('error', (e) => console.error(e.reason, e.message));
```

| `reason` | Meaning |
|---|---|
| `not-initialised` | Normal for the first tick; `initialize()` has not finished |
| `model-load-failed` | The `.onnx` could not be fetched or parsed |
| `runtime-unavailable` | Neither ORT package could be imported |
| `inference-error` | The model threw |
| `inference-timeout` | Slower than `inference.timeoutMs` (default 250 ms) |
| `invalid-observation` | The adapter could not read usable player state |
| `empty-ladder` | No enabled renditions |
| `disabled` | You called `abr.disable()` |

### Tests pass but quality never changes during playback

VHS applies its own buffer guards on top of your decision — `shouldSwitchToMedia_`
will refuse an upswitch when the forward buffer is below its low-water line. The
engine chose correctly; the player declined. Check `adapter.on('apply')` — if
`applied` is `true` but the rendition does not change, it is VHS's guard, not the
engine.

---

## 9. Test inventory

| File | Tests | Covers |
|---|---:|---|
| `test/parity.test.ts` | 4 | PyTorch → ONNX → JS on 296 golden fixtures |
| `test/state-builder.test.ts` | 14 | Bit-exact port of `StateBuilder`, 98 recorded steps |
| `test/ladder.test.ts` | 19 | Action ↔ rendition mapping, all three strategies, boundaries |
| `test/validate.test.ts` | 26 | Observation validation, repair, malformed input |
| `test/engine.test.ts` | 42 | Lifecycle, decisions, every fallback, telemetry, rate limiting |
| `test/adapter-videojs.test.ts` | 40 | VHS surface, observation building, selector install/restore, teardown |
| `test/integration.test.ts` | 11 | Real ONNX model end to end, model limitation, safety guard |
| `test/manifest.test.ts` | 11 | Constants have not drifted from the research code |
| `test/reward.test.ts` | 10 | QoE metric matches `src/env.py` |
| **Total** | **177** | |
| `test-e2e/run.mjs` | 23 checks | Real browser, real Video.js, real playback |

Python side:

| File | Covers |
|---|---|
| `export/validate_parity.py` | PyTorch vs ONNX in Python, before JS is involved |
| `export/make_fixtures.py` | Regenerates the 296 parity fixtures |
| `export/make_state_fixtures.py` | Regenerates the 98 StateBuilder fixtures |

---

## Quick reference

```bash
npm install && npx playwright install chrome   # setup

npm run build          # build the package
npm test               # 177 unit + integration tests
npm run typecheck      # strict TypeScript
npm run lint           # eslint
npm run make-stream    # generate a local HLS stream (needs ffmpeg)
npm run test:browser   # 23 real-browser end-to-end checks
npm run test:all       # unit + browser
npm run bench          # performance report
npm run example        # serve the examples at :8080

npm run export-model   # re-export .pt → .onnx + regenerate fixtures (Python)
npm run validate-model # PyTorch vs ONNX parity check (Python)
```
