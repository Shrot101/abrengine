# Deployment layer — what was added to this repository

The research code in `src/` and the checkpoints in `checkpoints/` are
**unchanged**. Everything below was added alongside them.

```
abrengine-rl/
├── src/                      ← UNCHANGED research code
├── checkpoints/              ← UNCHANGED trained models
├── results/                  ← UNCHANGED plots
├── README.md                 ← UNCHANGED research write-up
│
├── package.json              npm workspaces; one command per task
├── TESTING.md                step-by-step guide to verifying everything
├── DEPLOYMENT.md             this file
├── LICENSE
│
├── docs/
│   └── ARCHITECTURE.md       full architecture report
│
├── export/                   Python → ONNX pipeline
│   ├── export_onnx.py        .pt → .onnx + a manifest generated from src/
│   ├── make_fixtures.py      296 golden (state → PyTorch output) pairs
│   ├── make_state_fixtures.py 98 recorded StateBuilder.update() steps
│   ├── validate_parity.py    PyTorch vs ONNX, in Python
│   └── fixtures/             the generated fixtures (committed)
│
├── packages/abrengine/       the npm package
│   ├── src/
│   │   ├── index.ts          public core API
│   │   ├── videojs.ts        the 'abrengine/videojs' entry point
│   │   ├── types/            units, observation, decision, config, telemetry, adapter
│   │   ├── core/             engine, state-builder, ladder, validate, reward, safety
│   │   ├── model/            manifest, runtime resolution, model source
│   │   ├── fallback/         fallback strategies
│   │   ├── adapters/videojs/ the Video.js integration
│   │   └── utils/            emitter, clock
│   ├── models/
│   │   ├── ac3-controller.onnx   1.02 MB, exported from abrengine_final.pt
│   │   └── ac3-controller.json   manifest: hashes, constants, ladder, semantics
│   ├── test/                 177 tests
│   ├── README.md             the package README
│   └── package.json, tsconfig.json, tsup.config.ts, vitest.config.ts, eslint.config.js
│
├── test-e2e/                 real-browser end-to-end suite
│   ├── run.mjs               23 checks against real Chromium + Video.js
│   ├── page.html             the test page
│   ├── make-stream.sh        generates a 6-rendition HLS stream with ffmpeg
│   └── stream/               the generated stream (gitignored)
│
├── examples/                 four runnable examples + a dev server
└── scripts/
    └── bench.mjs             performance measurement
```

## Commands

```bash
npm install                # install everything

npm run build              # build the package
npm test                   # 177 unit + integration tests
npm run typecheck          # strict TypeScript
npm run lint               # eslint
npm run format             # prettier

npm run make-stream        # generate a local HLS stream (needs ffmpeg)
npm run test:browser       # 23 real-browser end-to-end checks
npm run test:all           # unit + browser
npm run bench              # performance report
npm run example            # serve the examples at :8080

npm run export-model       # re-export .pt → .onnx + regenerate fixtures (Python)
npm run validate-model     # PyTorch vs ONNX parity check (Python)
```

## Design decisions, in brief

| Decision | Reasoning |
|---|---|
| **ONNX, not TorchScript** | No JavaScript runtime executes TorchScript. ONNX exports cleanly at opset 17 with ten core operators. |
| **`onnxruntime-web` in Node too** | `onnxruntime-node` downloads native binaries from `api.nuget.org` at install time — a real failure on restricted networks. ORT-Web's WASM backend runs in Node and is validated to 7.7e-7. |
| **Graph input is the raw `(6,8)` tensor** | Nothing folded into the graph. The ONNX module consumes exactly what the PyTorch module consumed, so semantics cannot drift during conversion. |
| **`StateBuilder` ported, not approximated** | Verified bit-for-bit against Python across 98 recorded steps. A wrong divisor here produces a model that runs fine and behaves like a different model. |
| **Decisions cached; `selectPlaylist` reads the cache** | VHS calls it synchronously at up to 4 Hz; inference is async and the policy was trained to decide once per chunk. |
| **`fallback: 'player-default'` captures the real selector** | The adapter saves VHS's incumbent `selectPlaylist` before overriding, so the fallback is the player's genuine default, not a reimplementation. |
| **Model bundled but fetched lazily** | 1.02 MB is fine in a tarball; base64-inlining it would add ~1.4 MB of JavaScript that is parsed every load and cannot be cached separately. |
| **Telemetry off by default** | When off, the engine does not allocate the payloads. Measured: 0.169 ms vs 0.131 ms per decision. |
| **Safety guardrail off by default** | Enabling it overrides the trained policy's action. The default reproduces the researched behaviour exactly; the guardrail is opt-in and reports every intervention. |

## Validation summary

| Comparison | Fixtures | Max abs Δ (policy) | argmax |
|---|---:|---|---|
| PyTorch → ONNX (Python) | 296 | 7.75e-7 | 296/296 |
| PyTorch → ONNX (Node, WASM) | 296 | 7.75e-7 | 296/296 |
| Node → browser | identical input | 0 | — |
| `StateBuilder` port | 98 steps | bit-exact | — |

## Performance

| Measurement | Value |
|---|---|
| Model size | 1 045.8 KiB (967 KiB gzipped) |
| Package bundle | 1.7 KiB core + 21.7 KiB adapter + 42.3 KiB shared (12.3 KiB gz) |
| Cold session load | ~309 ms (includes WASM init) |
| Inference p50 / p95 | 0.151 ms / 0.247 ms |
| `decide()` p50 / p95 | 0.169 ms / 0.278 ms |
| Adapter overhead | 0.018 ms |
| Duty cycle at one decision per 4 s | 0.0042 % |

## Known limitation

The bundled `ac3` controller does not reliably back off when the link collapses —
its argmax is action 4 (2850 kbps) in 51 of 56 sampled (throughput × buffer)
states, including 0.1 Mbps with an empty buffer.

This is the trained model, not the port: the identical state produces the
identical argmax in PyTorch. See
[docs/ARCHITECTURE.md §7](./docs/ARCHITECTURE.md#7-technical-risks) and
[TESTING.md §7](./TESTING.md#7-what-to-look-for--reading-the-results) for the
measurement, the likely cause, and the opt-in guardrail.
