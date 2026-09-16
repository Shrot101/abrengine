# abrengine

Run a trained reinforcement-learning ABR controller as the adaptive-bitrate
algorithm in JavaScript video players.

The controller is a Pensieve-style A3C actor-critic, trained in PyTorch, exported
to ONNX, and executed in the browser by ONNX Runtime Web. The package ships a
framework-independent core plus a Video.js / VHS adapter; other players are
adapters, not rewrites.

```bash
npm install abrengine onnxruntime-web
```

```js
import videojs from 'video.js';
import { AbrEngine } from 'abrengine';
import { VideoJSAbrAdapter } from 'abrengine/videojs';

const player = videojs('video');
const abr = new AbrEngine({ model: 'ac3' });
const adapter = new VideoJSAbrAdapter({ player, abr });
await adapter.initialize();

player.src({ src: '/stream/master.m3u8', type: 'application/x-mpegURL' });
```

That is the whole integration. The trained controller now picks the bitrate for
every segment, and if it cannot, Video.js's own algorithm takes over silently.

---

## Contents

1. [What this is](#1-what-this-is)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [Supported environments](#4-supported-environments)
5. [Model and runtime requirements](#5-model-and-runtime-requirements)
6. [Video.js integration](#6-videojs-integration)
7. [Configuration](#7-configuration)
8. [Using your own model](#8-using-your-own-model)
9. [Telemetry](#9-telemetry)
10. [Fallback behaviour](#10-fallback-behaviour)
11. [Other players](#11-other-players)
12. [Development](#12-development)
13. [Testing](#13-testing)
14. [Building and publishing](#14-building-and-publishing)
15. [Limitations](#15-limitations)

---

## 1. What this is

Adaptive bitrate selection is a sequential decision problem: before each segment,
the player must choose a quality level given the network it has seen, the buffer
it holds, and the sizes of what it is about to download. Conventional players use
hand-tuned heuristics — throughput averaging, buffer-occupancy rules, MPC.

This package replaces that heuristic with a policy network trained by
reinforcement learning to maximise a log-scale QoE reward:

```
r = log(bitrate / bitrate_min) − 4.3 · rebuffer_seconds − 1.0 · |Δ log bitrate|
```

The model has 265 863 parameters, is 1.02 MB on disk, and takes about 0.17 ms per
decision. Decisions happen once per segment (~4 s), so ABR costs roughly 0.004 %
of playback wall-clock time.

**The research code stays untouched.** `src/` — the environment, the network, the
training loop, the evaluation — is unchanged. This package consumes the exported
checkpoint; it does not reimplement the algorithm.

---

## 2. Architecture

```
                    ┌──────────────────────────────────────────┐
                    │            Your application              │
                    └───────────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │        VideoJSAbrAdapter                 │
                    │  (abrengine/videojs)                     │
                    │                                          │
                    │  reads   VHS state  → AbrObservation     │
                    │  writes  AbrDecision → selectPlaylist    │
                    │                                          │
                    │  ── contains no ABR logic ──             │
                    └───────────────────┬──────────────────────┘
                                        │  AbrObservation
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │            AbrEngine                     │
                    │  (abrengine)                             │
                    │                                          │
                    │  validate → StateBuilder → ONNX policy   │
                    │           → argmax → ladder mapping      │
                    │           → safety guard (opt-in)        │
                    │                                          │
                    │  knows nothing about Video.js or the DOM │
                    └───────────────────┬──────────────────────┘
                                        │  Float32Array[48]
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │  onnxruntime-web  (WASM)                 │
                    │  ac3-controller.onnx — 1.02 MB           │
                    │  in → state [1,6,8]                      │
                    │  out → action_probs [1,6], value [1,1]   │
                    └──────────────────────────────────────────┘
```

The core never imports Video.js, the DOM, or an inference runtime at module load
time. The runtime is imported lazily inside `initialize()`.

### The decision loop

VHS calls `selectPlaylist()` **synchronously**, and calls it far more often than
segments complete — a 250 ms timer when `bufferBasedABR` is on, plus every
`bandwidthupdate` and every fullscreen change. ONNX inference is asynchronous, and
the controller was trained to decide once per chunk. So the adapter runs two
decoupled loops:

```
 segment completes                       VHS asks for a playlist
 (tech 'bandwidthupdate')                (checkABR_ → selectPlaylist)
         │                                          │
  buildObservation()                       read cached decision
         │                                          │
  engine.decide() ── async ──► cache ───────────────┘
                                          (synchronous, ~0 cost)
```

The cache carries a TTL. A decision older than `decisionTtlMs` is stale, and the
player's own selector runs instead — so a wedged inference can never pin the
player to one rendition.

### Model input

The 6×8 state tensor is built exactly as `src/train.py::StateBuilder` builds it:

| Row | Contents | Units | Where written |
|---|---|---|---|
| 0 | throughput history | Mbps ÷ `THROUGHPUT_NORM` (1) | shifted left, newest at index 7 |
| 1 | download-time history | seconds | shifted left, newest at index 7 |
| 2 | next-segment size per action | bytes ÷ `CHUNK_NORM` (1e6) | indices 0–5 |
| 3 | forward buffer | seconds ÷ `BUFFER_NORM` (10) | index 7 only |
| 4 | remaining segments | ÷ `NUM_CHUNKS` (48) | index 7 only |
| 5 | previous action index | ÷ (A_DIM−1) = 5 | index 7 only |

Rows 3–5 are never shifted, and indices 0–6 of those rows stay zero for the whole
session — that is what training did, and `test/state-builder.test.ts` asserts the
port reproduces it bit for bit against fixtures recorded from Python.

---

## 3. Installation

```bash
npm install abrengine
npm install onnxruntime-web        # peer dependency, browser + Node
npm install video.js               # peer dependency, only for the adapter
```

`onnxruntime-web` is a **peer** dependency on purpose: it is several megabytes,
many applications already have a copy, and bundling it would multiply your bundle
size for no benefit.

### Why `onnxruntime-web` and not `onnxruntime-node` in Node

`onnxruntime-node` downloads native binaries from `api.nuget.org` in a postinstall
script — a real install-time failure on locked-down CI and corporate networks.
ORT-Web's WASM backend runs unmodified under Node ≥18 and is validated here to
7.7e-7 against PyTorch. For a 265k-parameter model the native runtime's speed
advantage is irrelevant.

`onnxruntime-node` is still supported: `inference: { runtime: 'onnxruntime-node' }`.

---

## 4. Supported environments

| Environment | Status | Notes |
|---|---|---|
| Chrome / Edge (desktop, Android) | **Tested** | End-to-end, real playback |
| Node.js ≥ 18 | **Tested** | Full test suite, ORT-Web WASM backend |
| Firefox | Expected to work | Same WASM + MSE surface; not yet run end to end |
| Safari (macOS, iOS) | Expected to work with caveats | VHS defaults to *native* HLS on Safari; set `vhs: { overrideNative: true }` to route through MSE, or the adapter has nothing to control |
| Older browsers without WASM | Not supported | Falls back cleanly to the player default |

"Tested" means the automated end-to-end suite runs real playback there. Anything
labelled "expected" has not been verified, and this README will not claim
otherwise.

Bundlers: Vite, webpack 5, Rollup and esbuild all resolve the package's `exports`
map and the `new URL(…, import.meta.url)` model reference. For anything else, pass
an explicit model URL (§8).

---

## 5. Model and runtime requirements

The bundled model is exported by `export/export_onnx.py`:

| Property | Value |
|---|---|
| File | `models/ac3-controller.onnx` |
| Size | 1 045.8 KiB (967 KiB gzipped) |
| Parameters | 265 863 float32 |
| Opset | 17 |
| Operators | `Concat, Constant, Conv, Flatten, Gather, Gemm, Relu, Slice, Softmax, Unsqueeze` |
| Input | `state`, float32 `[batch, 6, 8]`, dynamic batch |
| Outputs | `action_probs` float32 `[batch, 6]`; `state_value` float32 `[batch, 1]` |

Every operator is a core ONNX op fully supported by ORT Web's WASM backend. There
are no custom operators and no dynamic shapes beyond the batch axis.

A sidecar `models/ac3-controller.json` records the source checkpoint's SHA-256,
the normalisation constants, the bitrate ladder and the action semantics — all
read directly from the research code at export time. `test/manifest.test.ts`
asserts the TypeScript constants still agree with it, so a retrain that changes a
divisor cannot silently corrupt every state tensor.

**Execution providers.** `['wasm']` is the default and the only one validated for
numerical parity. `webgpu` and `webgl` are accepted but unvalidated; at 265k
parameters, GPU dispatch overhead exceeds the compute.

**Threads.** Default 1. Multi-threaded WASM requires cross-origin isolation
(COOP/COEP headers), which most sites do not have, and this model gains nothing
from threads.

---

## 6. Video.js integration

Verified against **video.js 8.24.0** and **@videojs/http-streaming 3.17.5** by
reading the shipped source, not from memory.

### The extension point

VHS exposes an assignable `selectPlaylist` on its handler:

```js
Object.defineProperties(this, {
  selectPlaylist: {
    get() { return this.playlistController_.selectPlaylist; },
    set(fn) { this.playlistController_.selectPlaylist = fn.bind(this); }
  },
});
```

`PlaylistController.checkABR_()` calls it with no arguments and expects a playlist
object back, or a falsy value to keep the current one. The adapter captures the
incumbent implementation — normally `Vhs.STANDARD_PLAYLIST_SELECTOR`
(`lastBandwidthSelector`) — before overriding it. That capture is what makes
`fallback: 'player-default'` genuinely the player's default rather than a
reimplementation of it, and it is restored on `destroy()`.

### What the adapter reads

| Need | Surface | Kind |
|---|---|---|
| the ladder | `vhs.representations()` | public |
| current rendition | `vhs.playlists.media()` | public |
| throughput estimate | `vhs.bandwidth` (bits/s) | public |
| per-segment stats | `pc.mainSegmentLoader_.{mediaBytesTransferred,mediaTransferDuration}` | private, with a `vhs.stats` fallback |
| segment-completed tick | `mainSegmentLoader_.on('bandwidthupdate')` → `tech.trigger('bandwidthupdate')` | event |
| force a switch | `pc.fastQualityChange_()` | private, optional |

Two details worth knowing:

- **`bandwidthupdate` fires on the *tech*, not the player**, and Video.js does not
  forward it. Listening only on the player yields zero ticks. The adapter binds
  to the tech *and* directly to the segment loader.
- **`vhs.stats.mediaBytesTransferred` sums every segment loader** (main + audio +
  subtitles). With demuxed audio it over-counts video bytes, so the adapter
  prefers `mainSegmentLoader_` and reports which source it is using via
  `adapter.on('attach')`.

### Options

```js
new VideoJSAbrAdapter({
  player,
  abr,
  decisionTtlMs: 30_000,          // stale decisions defer to the player default
  defaultSegmentDurationSec: 4,   // used when the manifest declares no target duration
  applyImmediately: false,        // true calls fastQualityChange_ (flushes the buffer)
  resetOnSeek: true,              // clear the model's 8-step history across a seek
  pollIntervalMs: 0,              // extra timer ticks; usually unnecessary
  attachTimeoutMs: 15_000,        // how long to keep looking for a VhsHandler
  debug: false,
});
```

`initialize()` resolves as soon as the **engine** is ready. It deliberately does
not wait for VHS, because a VhsHandler only exists once the player has a source
and plenty of ordinary code sets the source after wiring up plugins. The adapter
attaches opportunistically and fires `attach` when it does; `adapter.active` tells
you the current state, and `waitUntilAttached()` is there if you need to block.

---

## 7. Configuration

```ts
const abr = new AbrEngine({
  model: 'ac3',

  inference: {
    runtime: 'auto',                  // 'onnxruntime-web' | 'onnxruntime-node'
    executionProviders: ['wasm'],
    wasmThreads: 1,
    wasmPaths: undefined,             // where ORT's .wasm assets live
    timeoutMs: 250,                   // abandon a slow inference and fall back
    warmup: true,                     // one dummy pass during initialize()
  },

  semantics: {
    trainingLadderKbps: [300, 750, 1200, 1850, 2850, 4300],
    ladderMapping: 'nearest-bitrate', // | 'proportional-rank' | 'identity'
    bufferNormSec: 10,
    chunkNormBytes: 1e6,
    throughputNorm: 1,
    totalChunksNorm: 48,
    segmentDurationSec: 4,
  },

  fallback: 'player-default',         // | 'throughput' | 'buffer' | 'lowest' | 'hold' | fn
  throughputSafetyFactor: 0.9,

  safety: { enabled: false },         // see §15

  minDecisionIntervalMs: 0,           // rate-limit model evaluations
  decisionTtlMs: 30_000,

  telemetry: false,                   // true | { enabled, includeTensors, historySize }
  debug: false,
  startDisabled: false,
});
```

Every value in `semantics` defaults to what the model was trained with. Changing
one changes model semantics, and each field's doc comment says so.

### Ladder mapping

The model's action space is fixed at six slots meaning `[300, 750, 1200, 1850,
2850, 4300]` kbps. Real streams almost never have that ladder, so something must
map between them — and that is a semantic choice, made explicit rather than
buried:

- **`nearest-bitrate`** (default) — the action names a target bitrate; pick the
  enabled rendition closest to it **in log space**. Log space because the reward
  the policy was trained on is `log(BR / BR_min)`: an equal *ratio* is an equal
  perceptual step. Ties break downward.
- **`proportional-rank`** — the action names a *position*; slot `i` of 6 maps to
  rank `round(i·(n−1)/5)` of your `n` renditions. Use this when your ladder spans
  a very different range and absolute matching would pin the policy to one end.
- **`identity`** — the index is a ladder index, clamped. Only sensible when your
  ladder *is* the training ladder.

---

## 8. Using your own model

Four sources, all first-class:

```js
// 1. Bundled — resolved via new URL('../models/…', import.meta.url)
new AbrEngine({ model: 'ac3' })

// 2. By URL — the right choice for a CDN, where you control cache headers
//    and can roll the model forward independently of your JS bundle
new AbrEngine({ model: { type: 'url', url: 'https://cdn.example/ac3-v2.onnx' } })

// 3. From bytes you already have
new AbrEngine({ model: { type: 'buffer', buffer: arrayBuffer } })

// 4. Your own inference session — TFJS, WebNN, a mock, a shared session
new AbrEngine({
  model: {
    type: 'session',
    session: {
      async run(state /* Float32Array[48] */) {
        return { actionProbs: Float32Array, stateValue: number };
      },
    },
  },
})
```

### Model distribution: what we chose and why

| Strategy | Verdict |
|---|---|
| **Bundled in the npm package** | **Chosen as the default.** 1.02 MB is well within reason for an npm tarball, it works offline, it versions atomically with the code, and it needs no infrastructure. |
| **Downloaded from a configurable URL** | **Fully supported, and recommended at scale.** Your CDN, your cache headers, model updates without a package release. |
| **Supplied by the application** | Supported (`{ type: 'buffer' }`). For apps that already manage assets. |
| **Separate model package** | Not done. One model at 1 MB does not justify a second package and a version-compatibility matrix. Revisit at three or more models. |
| **Embedded in the JS bundle** | **Rejected.** Base64 inflates 1.02 MB to ~1.4 MB of JavaScript that must be parsed on every load and cannot be cached separately from the code. |

The bundled file is never inlined — it is fetched lazily, so the browser caches
it independently and a code deploy does not invalidate it.

### After you retrain

```bash
npm run export-model     # re-export .pt → .onnx and regenerate fixtures
npm run validate-model   # PyTorch vs ONNX parity
npm test                 # everything downstream
```

If your retrain changed a normalisation constant or the ladder,
`test/manifest.test.ts` fails loudly. That is the point: silently building state
tensors with the wrong divisor is the worst possible failure mode, because
nothing crashes — the model just quietly behaves like a different model.

---

## 9. Telemetry

Off by default. When off, the engine does not allocate the event payloads at all.

```js
const abr = new AbrEngine({
  model: 'ac3',
  telemetry: { enabled: true, includeTensors: true, historySize: 100 },
});

abr.on('decision', (e) => {
  e.timestamp;              // Date.now()
  e.decision.source;        // 'model' | 'fallback' | 'player-default'
  e.decision.reason;        // why, when not 'model'
  e.decision.actionIndex;   // 0..5, the model's action
  e.decision.representationId;
  e.decision.model.actionProbs;  // the full softmax
  e.decision.model.stateValue;   // the critic's V(s)
  e.decision.safetyClamp;        // non-null if the guardrail intervened
  e.bufferSec;
  e.throughputBps;
  e.downloadSec;
  e.segmentBytes;
  e.selectedBitrateBps;
  e.availableBitratesBps;
  e.inferenceMs;
  e.totalMs;
  e.modelInput;             // Float32Array[48], with includeTensors
  e.modelOutput;            // Float32Array[6],  with includeTensors
});

abr.on('error', (e) => console.error(e.reason, e.message));  // always emitted
abr.on('ready', (e) => console.log(e.runtime, e.loadMs, e.modelBytes));
abr.on('observation', (e) => console.log(e.repairs));        // fields that were repaired
abr.on('state', (e) => console.log('enabled:', e.enabled));

abr.history();  // the retained ring buffer
```

`error` events fire regardless of the telemetry setting. Failures are never
silent.

Measured cost: `decide()` p50 is 0.169 ms with telemetry on, 0.131 ms with it off.

---

## 10. Fallback behaviour

`decide()` **never rejects and never throws.** Every failure produces a decision
with `source: 'fallback'` (or `'player-default'`), a `reason`, and an `error`
event. A player integration can call it without a try/catch and without ever
risking playback.

| Failure | `reason` |
|---|---|
| model still loading | `not-initialised` |
| the `.onnx` could not be fetched or parsed | `model-load-failed` |
| no ONNX runtime importable | `runtime-unavailable` |
| inference threw | `inference-error` |
| inference exceeded `timeoutMs` | `inference-timeout` |
| observation unusable | `invalid-observation` |
| no enabled renditions | `empty-ladder` |
| model returned something that is not a distribution | `invalid-model-output` |
| you called `disable()` | `disabled` |
| the cached decision went stale | `decision-stale` |

Strategies:

| `fallback` | Behaviour |
|---|---|
| **`'player-default'`** (default) | Decline. The adapter re-runs the selector the player had before this package overrode it — the genuine default, not a copy of it. |
| `'throughput'` | Highest rendition under `estimate × throughputSafetyFactor`. |
| `'buffer'` | The buffer-occupancy heuristic from `src/test.py`, generalised to any ladder size. |
| `'lowest'` | Always the lowest enabled rendition. |
| `'hold'` | Keep the current rendition. |
| a function | `(observation, reason) => representationId \| null`. Returning `null` declines. |

A fallback function that throws is caught and treated as a decline.

---

## 11. Other players

The core has no player dependency. An integration implements:

```ts
interface PlayerAbrAdapter {
  readonly name: string;
  initialize(): Promise<void>;
  getObservation(): AbrObservation | null;   // never throws; null when not ready
  applyDecision(decision: AbrDecision): boolean;
  destroy(): void;                            // idempotent; restores the player
  readonly active: boolean;
}
```

The Video.js adapter is ~450 lines, of which roughly 300 are reading VHS state and
comments explaining why. Building an hls.js or Shaka adapter means answering four
questions:

1. **What is the once-per-segment tick?** (hls.js: `FRAG_BUFFERED`;
   Shaka: `player.addEventListener('adaptation')` plus its network filters.)
2. **How do I read per-segment bytes and download time?**
3. **How do I enumerate the ladder with stable ids?**
4. **How do I override the player's own selection, synchronously or otherwise?**

Answer those and the adapter is mechanical. Ship it as
`abrengine/hlsjs` alongside `abrengine/videojs`.

---

## 12. Development

```bash
npm install            # workspace root
npm run build          # tsup → dist/ (ESM + CJS + .d.ts + sourcemaps)
npm run typecheck      # tsc --noEmit, strict
npm run lint           # eslint
npm run format         # prettier
npm run example        # serve the examples at :8080
```

Layout:

```
packages/abrengine/
  src/
    index.ts               public core API
    videojs.ts             the 'abrengine/videojs' entry point
    types/                 units, observation, decision, config, telemetry, adapter
    core/                  engine, state-builder, ladder, validate, reward, safety
    model/                 manifest, ONNX runtime resolution, model source
    fallback/              fallback strategies
    adapters/videojs/      the Video.js integration
    utils/                 emitter, clock
  models/                  ac3-controller.onnx + its manifest
  test/                    177 tests
```

---

## 13. Testing

See **[TESTING.md](../../TESTING.md)** at the repository root for a full
step-by-step guide. Short version:

```bash
npm test               # 177 unit + integration tests
npm run make-stream    # generate a local HLS stream (needs ffmpeg)
npm run test:browser   # 23 real-browser end-to-end checks
npm run bench          # performance report
```

The two tests that carry the most weight:

- **`test/parity.test.ts`** — 296 states recorded from PyTorch, replayed through
  the ONNX model in JavaScript. Max policy deviation 7.7e-7; argmax agrees on
  296/296. This is what justifies claiming the package runs *your* controller.
- **`test/state-builder.test.ts`** — 98 recorded `StateBuilder.update()` steps,
  compared **bit for bit** against Python. This catches feature-ordering and
  normalisation bugs that the ONNX parity test cannot see, because that test
  starts from a state tensor it is handed.

---

## 14. Building and publishing

```bash
npm run build
npm pack              # inspect the tarball first
npm publish
```

Before publishing:

- [ ] `name` in `package.json` — currently `abrengine`; scope it if you prefer
      (`@you/abrengine`), and update the README examples to match
- [ ] `version` — semver; bump the minor when model semantics change
- [ ] `repository.url` — points at `Shrot101/abrengine`
- [ ] `LICENSE` — MIT; the model weights are covered by the repository's licence
- [ ] `files` — `dist`, `models`, `README.md`, `LICENSE` only
- [ ] `npm pack --dry-run` should show ~1.1 MB, dominated by the model

The `exports` map gives you `abrengine` and `abrengine/videojs`, plus
`abrengine/models/ac3-controller.onnx` so applications can copy or preload the
asset directly.

---

## 15. Limitations

### The bundled model does not reliably back off

Over a 56-point sweep of (throughput × buffer) states, the `ac3` controller's
argmax is action 4 — 2850 kbps — in **51 of them**, including states with 0.1 Mbps
of measured throughput and an empty buffer. Driven from 8 Mbps down to 0.67 Mbps
with the buffer draining to zero, it returns to 2850 kbps within two steps and
holds there with 98.5 % confidence.

**This is the trained model, not the port.** The identical state fed to
`checkpoints/abrengine_final.pt` in PyTorch produces the identical argmax, and the
parity suite agrees to 7.7e-7 across 296 fixtures. It is consistent with the
research evaluation, which reports 1.89 s of rebuffering per episode against the
buffer-based heuristic's 0.43 s and describes the policy as "locked at 2850 kbps
for the entire episode".

The likely cause is the training environment: `src/env.py` generates synthetic
bandwidth from a clipped normal around 3 Mbps, which rarely produces the sustained
collapses a policy would need to learn descent from. Retraining on real FCC or
HSDPA traces is the fix.

Until then there is an **opt-in guardrail**, off by default because enabling it
overrides the trained policy's action:

```js
const abr = new AbrEngine({
  model: 'ac3',
  safety: { enabled: true, bufferFloorSec: 8, throughputFactor: 0.9 },
});
```

It only ever *lowers* a selection, only when the buffer is below the floor, and
only against a *measured* throughput — never a guess. Every intervention appears
as `decision.safetyClamp`, so you can measure how often the model is being
overridden rather than having it hidden.

### Live streams

The model's "remaining chunks" input encodes how close the episode is to ending —
a concept that does not exist for a live edge. The engine pins that input to 1.0
("plenty left"), which is inside the training distribution and true for live, but
it is an approximation. VOD is the exercised path.

### The training ladder is fixed

Six actions, fixed bitrates. Streams with a very different ladder go through the
mapping in §7, which preserves intent but is not the same as a model trained on
your ladder.

### `next segment size` is estimated

Real HLS/DASH manifests rarely declare per-segment byte sizes, so the adapter uses
`bitrate × duration / 8` — which is exactly how `src/env.py` synthesises chunk
sizes, so it matches training. If your player can supply measured sizes, pass them
in `observation.nextSegmentSizesBytes` and they are used instead.

### Safari

VHS uses native HLS on Safari by default, where there is no `selectPlaylist` to
override. Set `html5: { vhs: { overrideNative: true } }` to route through MSE.
Not yet verified end to end.

### VHS can decline

Returning a playlist is a request, not a command: `shouldSwitchToMedia_` applies
buffer guards on top. `adapter.on('apply')` reporting `applied: true` while the
rendition does not change means VHS declined — the engine chose correctly.

---

## Licence

MIT. See [LICENSE](./LICENSE).

Built on the ideas in *Neural Adaptive Video Streaming with Pensieve*
(Mao, Netravali, Alizadeh — SIGCOMM 2017).
