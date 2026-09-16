# Architecture report

What the research code does, what the JavaScript package does, and exactly where
the line between them falls.

Written after reading every file in `src/`, the `.pt` checkpoints, and the shipped
source of video.js 8.24.0 / @videojs/http-streaming 3.17.5.

---

## 1. The existing research implementation

### 1.1 Repository as found

```
abrengine-rl/
├── README.md                       research write-up, results, plots
├── requirements.txt                torch, numpy, matplotlib, tqdm, pandas
├── src/
│   ├── env.py       (8.2 KB)       streaming simulator
│   ├── model.py     (6.7 KB)       the actor-critic network
│   ├── train.py     (12.7 KB)      A3C training loop + StateBuilder
│   └── test.py      (8.8 KB)       evaluation against four baselines
├── checkpoints/
│   ├── abrengine_final.pt          1.3 MB — weights only
│   └── abrengine_ep{500…5000}.pt   3.2–3.5 MB — weights + optimiser state
└── results/                        three PNG plots
```

No tests, no configuration files, no packaging. That is normal and fine for
research code; it is stated here because the deployment work had to add all of it.

### 1.2 Two naming corrections

**The algorithm is A3C, not AC3.** `train.py` implements single-process
rollout-based advantage actor-critic. The public model name in the package is
`ac3-controller` because that is what was asked for, but the documentation says
what it actually is.

**There is no AC3 dataset.** `env.py::_load_trace` accepts a trace directory, but
`train.py`'s default is `trace_files = [None]` — the synthetic generator:

```python
t  = np.cumsum(rng.uniform(0.5, 2.0, 300))
bw = np.abs(rng.normal(3.0, 1.5, 300)).clip(0.3, 10.0)
```

The shipped controller was trained on synthetic bandwidth drawn from a clipped
normal around 3 Mbps, not on FCC or HSDPA traces. This matters for §7.

**`abrengine_final.pt` and `abrengine_ep5000.pt` hold bit-identical weights.**
Verified by tensor comparison. The final checkpoint is the episode-5000 model
without the optimiser state. The package exports from `_final.pt`.

### 1.3 The environment (`src/env.py`)

A gym-style single-session simulator. No RL framework, no media files.

| Constant | Value | Meaning |
|---|---|---|
| `VIDEO_CHUNK_LEN` | 4.0 | seconds per chunk |
| `BITRATES` | `[300, 750, 1200, 1850, 2850, 4300]` | kbps |
| `NUM_CHUNKS` | 48 | chunks per episode (~3 min) |
| `BUFFER_THRESH` | 60.0 | seconds; the player sleeps above this |
| `LINK_RTT` | 80 | ms added to every download |
| `PACKET_PAYLOAD_PORTION` | 0.95 | payload fraction of transferred bytes |

Chunk sizes are synthesised deterministically (`rng = default_rng(42)`):

```python
base = bitrate_kbps * 1000 * VIDEO_CHUNK_LEN / 8      # bits → bytes
size = base * (1 + 0.1 * standard_normal(NUM_CHUNKS))  # ±10 %
```

`step(action)` walks the bandwidth trace to compute the download time, drains the
buffer as time passes, accumulates rebuffering when the buffer empties, adds the
new chunk, caps at `BUFFER_THRESH`, and returns the QoE reward:

```python
q_t    = log(BITRATES[a] / BITRATES[0])
q_prev = log(BITRATES[last] / BITRATES[0])
reward = q_t - 4.3 * rebuf - 1.0 * abs(q_t - q_prev)
```

### 1.4 The network (`src/model.py`)

`ActorCritic`, faithful to Pensieve §5.3:

```
state (B, 6, 8)
  ├─ row 0        → Conv1d(1,128,k=4) → ReLU → Flatten   (B, 640)
  ├─ row 1        → Conv1d(1,128,k=4) → ReLU → Flatten   (B, 640)
  ├─ row 2[:6]    → Conv1d(1,128,k=4) → ReLU → Flatten   (B, 384)
  ├─ row 3[-1]    → Linear(1,128) → ReLU                 (B, 128)
  ├─ row 4[-1]    → Linear(1,128) → ReLU                 (B, 128)
  └─ row 5[-1]    → Linear(1,128) → ReLU                 (B, 128)
                       │
                    concat (B, 2048)
                       │
                Linear(2048,128) → ReLU
                    ├─ actor  Linear(128,6) → Softmax
                    └─ critic Linear(128,1)
```

265 863 float32 parameters. Weights are orthogonally initialised with gain √2,
except the actor head at gain 0.01 for a near-uniform initial policy.

At evaluation time (`test.py::policy_abrengine`) the action is `argmax`, not a
sample. The deployed engine does the same — production ABR should be
deterministic.

### 1.5 The state representation (`src/train.py::StateBuilder`)

This is the single most important thing to port correctly, because a mistake here
produces a model that runs perfectly and behaves like a different model.

```python
tp = (chunk_bytes * 8.0 / 1e6) / max(delay_s, 1e-6)     # Mbps

state[0, :-1] = state[0, 1:];  state[0, -1] = tp / THROUGHPUT_NORM   # 1.0
state[1, :-1] = state[1, 1:];  state[1, -1] = delay_s
for i in range(6): state[2, i] = next_chunk_sizes[i] / CHUNK_NORM     # 1e6
state[3, -1] = buffer_s / BUFFER_NORM                                 # 10.0
state[4, -1] = remaining / NUM_CHUNKS                                 # 48
state[5, -1] = bitrate_idx / (NUM_BITRATES - 1)                       # 5
```

Three properties are load-bearing and easy to get wrong:

1. **Rows 0 and 1 shift left; newest goes last.** Rows 3, 4 and 5 are *never*
   shifted — only index 7 is written, and indices 0–6 stay zero for the entire
   session. The model reads only `[:, 3:6, -1]`, so shifting or zeroing them
   would still "work" while changing the input distribution.
2. **Row 2 is written at indices 0–5 only.** Indices 6–7 stay zero forever. The
   model slices `[:, 2, :6]`, so they are unused — but keeping them zero keeps the
   tensor byte-identical to training.
3. **The episode starts from all zeros.**

`test/state-builder.test.ts` asserts the TypeScript port reproduces Python's
tensor **bit for bit** across 98 recorded steps.

### 1.6 A discrepancy worth recording

`train.py` and `test.py` differ in one place. Training does *not* call
`builder.update(obs)` after `env.reset()`, so the first decision sees the all-zero
state. Evaluation *does*, so its first state has `remaining/48 = 1.0` in row 4.

The package follows **`test.py`**, because that is the code path the reported
results come from and it is the inference path. Documented here so the choice is
visible rather than accidental.

### 1.7 Training and evaluation

Rollout-based A3C, 8-step rollouts, γ = 0.99, Adam at 1e-4, gradient clip 0.5,
value-loss coefficient 0.5, entropy weight annealed 0.5 → 0.1 at 0.9995 per
episode. Advantages are normalised per rollout. `test.py` compares against
random, lowest, highest, and a buffer-occupancy heuristic.

---

## 2. The complete decision path, as traced

```
env.step(action)
  └─ obs { delay, buffer_size, chunk_size, next_chunk_sizes,
           video_chunk_remain, bitrate_action, rebuf }
        │
StateBuilder.update(obs)
  └─ normalise, shift the history, write the scalars → state (6, 8) float32
        │
ActorCritic.forward(state)
  └─ 3 conv branches + 3 FC branches → concat(2048) → FC(128)
        ├─ actor  → softmax over 6
        └─ critic → V(s)
        │
argmax(action_probs) → action index 0..5
        │
BITRATES[action] → the kbps to download next
```

In deployment the ends change and the middle does not:

```
VHS player state                      (replaces env.step)
  └─ AbrObservation
        │
StateBuilder (TypeScript port)        (identical arithmetic)
        │
ONNX graph                            (the exported ActorCritic)
        │
argmax → action index                 (identical)
        │
ladder mapping → a VHS playlist       (replaces BITRATES[action])
```

---

## 3. Python-specific vs generic

| Component | Nature | Disposition |
|---|---|---|
| `env.py` simulator | Research only | Python only. Never needed at inference. |
| `env.py` reward | **Generic** | Ported to `core/reward.ts` for QoE scoring; not used to decide. |
| `env.py` BITRATES | **Generic** | The model's action semantics. In the manifest. |
| `model.py` architecture | PyTorch | Exported to ONNX. Not reimplemented. |
| `model.py` weights | **The asset** | Exported to ONNX verbatim. |
| `train.py` loop | Research only | Python only. |
| `train.py` StateBuilder | **Generic** | **Ported to TypeScript, bit-exact.** |
| `train.py` normalisation | **Generic** | In the manifest and the TS constants. |
| `test.py` argmax policy | **Generic** | The engine's decision rule. |
| `test.py` buffer heuristic | **Generic** | Ported as the `'buffer'` fallback. |
| `test.py` plotting | Research only | Python only. |

**Reusable as-is:** the checkpoint, the reward, the ladder, the normalisation
constants, the argmax rule, the buffer heuristic.
**Must be rewritten in JS:** `StateBuilder`, action→rendition mapping, the
inference wrapper, the whole player integration.
**Stays Python:** simulator, training, plotting, evaluation harness.

---

## 4. Running the model in JavaScript

### Options evaluated

| Approach | Verdict |
|---|---|
| **ONNX export + ONNX Runtime Web** | **Chosen.** Exports cleanly at opset 17, ten core operators, 1.02 MB, validated to 7.7e-7. |
| ONNX Runtime Web (browser) | Chosen for browsers. WASM backend, single-threaded, no COOP/COEP requirement. |
| ONNX Runtime Node | Supported, not the default. Its postinstall downloads native binaries from `api.nuget.org` — a real install failure on restricted networks (it failed outright in this project's CI container). ORT-Web's WASM backend runs in Node anyway. |
| TorchScript | **Rejected.** No JavaScript runtime executes TorchScript. ExecuTorch and torch.js are not production paths for a browser. |
| Hand-written WASM / WebGPU kernels | **Rejected.** Would mean reimplementing conv1d + gemm and re-proving numerical equivalence, for a model where inference is already 0.15 ms. |
| TensorFlow.js via ONNX→TF conversion | **Rejected.** A second lossy conversion hop, a larger runtime, no benefit. |

### Export results

Every question that had to be answered, answered:

- **Does it export?** Yes. `torch.onnx.export`, opset 17, TorchScript exporter.
- **Unsupported operators?** None. `Concat, Constant, Conv, Flatten, Gather, Gemm,
  Relu, Slice, Softmax, Unsqueeze` — all core ops with full ORT Web support.
- **Dynamic shapes?** Only the batch axis, declared explicitly. Everything else is
  static.
- **Must preprocessing be reproduced in JS?** Yes — `StateBuilder`. Nothing was
  folded into the graph; the graph input is the same `(6, 8)` tensor the PyTorch
  module consumes.
- **Do outputs match?** Yes, to 7.7e-7 on the policy across 296 fixtures, with
  argmax agreement 296/296. See §5.
- **Fast enough?** p50 0.151 ms, p95 0.247 ms, against a ~4 s decision interval.
- **Size?** 1 045.8 KiB (967 KiB gzipped).
- **Browser?** Verified end to end in Chromium with real playback.
- **Node?** Verified; the full suite runs there.
- **Bundlers?** Vite, webpack 5, Rollup, esbuild — via the `exports` map and
  `new URL(…, import.meta.url)`.
- **Bundle, download, or supply?** Bundled by default, fetched lazily; URL and
  buffer sources fully supported.

---

## 5. Validation

Three independent comparisons, all automated:

| Comparison | Fixtures | Max |Δ| policy | argmax |
|---|---:|---|---|
| PyTorch → ONNX (Python ORT) | 296 | 7.75e-7 | 296/296 |
| PyTorch → ONNX (Node, ORT-Web WASM) | 296 | 7.75e-7 | 296/296 |
| Node → browser (same input) | 1 probe | 0 (identical) | — |

Plus the state-building port, compared bit-for-bit against Python across 98
recorded `update()` steps from real simulator rollouts.

Fixtures come from three sources so the state distribution is not only what the
policy likes: rollouts driven by the model itself, rollouts driven by the buffer
heuristic, a deterministic cycling policy, and eight hand-built boundary states
(all-zeros, saturated buffer, dead link, rising and collapsing bandwidth, and
three out-of-distribution magnitudes).

The one 1.56e-2 absolute deviation is on a deliberately out-of-distribution
fixture where the critic outputs V(s) ≈ 189 596 — one float32 ULP at that
magnitude is ~1.6e-2. Relative error there is 5.8e-7. The tolerance is
`atol + rtol·|value|` for exactly that reason.

---

## 6. Video.js / VHS integration

Verified by reading `node_modules/@videojs/http-streaming/dist/videojs-http-streaming.cjs.js`.

### The extension point

```js
// line ~30357
Object.defineProperties(this, {
  selectPlaylist: {
    get() { return this.playlistController_.selectPlaylist; },
    set(fn) { this.playlistController_.selectPlaylist = fn.bind(this); }
  },
});
```

`checkABR_()` calls it with no arguments and expects a playlist object back:

```js
checkABR_(reason = 'abr') {
  const nextPlaylist = this.selectPlaylist();
  if (nextPlaylist && this.shouldSwitchToMedia_(nextPlaylist)) {
    this.switchMedia_(nextPlaylist, reason);
  }
}
```

### Four facts that shaped the adapter

1. **`selectPlaylist` is synchronous; ONNX inference is not.** Hence the
   cached-decision design.
2. **It is called far more often than segments complete.**
   `startABRTimer_` installs `setInterval(() => this.checkABR_(), 250)` when
   `bufferBasedABR` is on, plus every `bandwidthupdate` and fullscreen change.
   Running a once-per-chunk policy at 4 Hz would not be the trained behaviour.
3. **`bandwidthupdate` fires on the *tech*, not the player**, and Video.js does not
   forward it. This cost a real debugging cycle: listening only on the player gave
   exactly zero ticks. The adapter binds to the tech *and* the segment loader.
4. **VHS can decline.** `shouldSwitchToMedia_` applies its own buffer guards.
   Returning a playlist is a request, not a command.

### Surfaces used

| Need | Surface | Kind |
|---|---|---|
| override selection | `vhs.selectPlaylist` | public |
| the ladder | `vhs.representations()` | public |
| current rendition | `vhs.playlists.media()` | public |
| throughput estimate | `vhs.bandwidth` | public |
| per-segment stats | `pc.mainSegmentLoader_.{mediaBytesTransferred,mediaTransferDuration}` | private, `vhs.stats` fallback |
| segment tick | `tech.on('bandwidthupdate')` | event |
| immediate switch | `pc.fastQualityChange_()` | private, optional |

`vhs.stats.*` sums main + audio + subtitle loaders (`sumLoaderStat`), so it
over-counts video bytes with demuxed audio. The adapter prefers the main loader
and reports which source it used.

### One environment gotcha

VHS registers its source handler only when the browser can decode H.264 + AAC in
MediaSource:

```js
const supportsNativeMediaSources = () =>
  browserSupportsCodec('avc1.4d400d,mp4a.40.2', true);
if (supportsNativeMediaSources()) {
  videojs.getTech('Html5').registerSourceHandler(VhsSourceHandler, 0);
}
```

That is a claim about the browser, not about the stream. Codec-stripped Chromium
builds fail it and VHS never loads, even for VP9/Opus content they decode fine.
The test harness and examples ship a narrowly-scoped shim for that one probe;
see `examples/codec-shim.js`.

---

## 7. Technical risks

| Risk | Severity | Status |
|---|---|---|
| **The policy is near-degenerate** — argmax is action 4 in 51 of 56 sampled states, including 0.1 Mbps with an empty buffer | **High** | Confirmed as a model property, not a port bug. Opt-in safety guardrail added; retraining on real traces is the real fix. |
| **Trained on synthetic bandwidth**, not real traces | High | Documented. Directly implicated in the above. |
| Ladder mismatch between training and real streams | Medium | Three explicit mapping strategies; `nearest-bitrate` default. |
| Live streams have no "remaining chunks" | Medium | Pinned to 1.0, inside the training distribution. VOD is the exercised path. |
| Next-segment sizes are estimated | Low | `bitrate × duration / 8` is exactly how the research env synthesises them. |
| VHS internals could change | Low | Every private access is probed with a public fallback; `attach` reports what was found. |
| ORT Web bundle size | Low | Peer dependency, never bundled. |
| Float32 accumulation drift | Negligible | Measured at 7.7e-7; the port is bit-exact upstream of the graph. |
| Safari native HLS | Medium | Needs `overrideNative: true`; not yet verified end to end. |

---

## 8. What was added, and why

Nothing in `src/` or `checkpoints/` was modified.

| Added | Why |
|---|---|
| `export/export_onnx.py` | The `.pt` → `.onnx` pipeline plus a manifest generated from the research constants |
| `export/make_fixtures.py` | 296 golden (state → PyTorch output) pairs |
| `export/make_state_fixtures.py` | 98 recorded `StateBuilder.update()` steps |
| `export/validate_parity.py` | PyTorch vs ONNX, in Python |
| `packages/abrengine/` | The npm package: core, adapter, types, 177 tests |
| `test-e2e/` | Real-browser end-to-end suite + an ffmpeg HLS stream generator |
| `examples/` | Four runnable examples and a zero-config dev server |
| `scripts/bench.mjs` | Performance measurement |
| `docs/`, `TESTING.md` | This report and the testing guide |
| Root `package.json` | npm workspaces, one command per task |

---

## 9. Acceptance criterion

> A developer who knows nothing about the internal research code should be able
> to install the npm package, instantiate the ABR engine, attach it to Video.js,
> play an HLS/DASH stream, and have the trained controller make the ABR decisions
> instead of the default algorithm, with a safe fallback if it cannot.

**Met, and verified in a real browser.** `npm run test:browser` plays a real
6-rendition HLS stream in Chromium through real Video.js and VHS, and reports:

```
✓ VHS switched rendition under the model (not the default ABR)
    — renditions played: v2@1264000 -> v4@2914000 -> v1@814000
✓ inference latency is playback-safe — p50=0.90ms p95=1.00ms
✓ 500 synchronous selectPlaylist calls trigger zero inferences
✓ playback continues after the model starts failing
✓ the failure is reported, not swallowed — inference-error ×5
23/23 checks passed
```

The integration itself is six lines.

The one honest caveat is §7's first row: the mechanism works end to end, but the
*trained policy* is weak, and that is a training problem rather than a deployment
one. The package makes it visible and gives you a guardrail rather than hiding it.
