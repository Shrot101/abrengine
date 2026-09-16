/**
 * Engine configuration.
 *
 * Everything the research code hard-codes is surfaced here, defaulted to the
 * value the model was actually trained with. Overriding a normalisation
 * constant is allowed but changes model semantics, so each one says so.
 */

import type { Seconds } from './units.js';
import type { AbrObservation } from './observation.js';
import type { AbrFallbackReason } from './decision.js';
import type { AbrSafetyConfig } from '../core/safety.js';

/** How to obtain the ONNX model bytes. */
export type AbrModelSource =
  /**
   * A model shipped inside this package. Currently only `'ac3'`
   * (alias: `'ac3-controller'`) — the A3C controller from `checkpoints/abrengine_final.pt`.
   *
   * Resolved with `new URL('../models/ac3-controller.onnx', import.meta.url)`,
   * which Vite / webpack 5 / Rollup / Node all understand. If your bundler does
   * not, pass an explicit `{ type: 'url' }` source instead.
   */
  | 'ac3'
  | 'ac3-controller'
  /** Fetch the model from a URL (relative URLs resolve against the document). */
  | { readonly type: 'url'; readonly url: string; readonly init?: RequestInit }
  /** Model bytes the application already has. */
  | { readonly type: 'buffer'; readonly buffer: ArrayBuffer | Uint8Array }
  /**
   * An already-constructed inference session. Use this to share one session
   * across engines, or to plug in a runtime this package does not know about.
   */
  | { readonly type: 'session'; readonly session: AbrInferenceSession };

/**
 * The minimal inference surface the engine needs.
 *
 * `onnxruntime-web` and `onnxruntime-node` sessions both satisfy this shape via
 * the adapters in `src/model/runtime.ts`; implement it yourself to use TFJS,
 * a WebNN wrapper, a hand-written WASM kernel, or a mock in tests.
 */
export interface AbrInferenceSession {
  /**
   * Run one forward pass.
   *
   * @param state Row-major `[S_INFO * S_LEN]` float32 state tensor
   *              (shape `[1, S_INFO, S_LEN]`).
   * @returns `actionProbs` of length `A_DIM`, and the critic value.
   */
  run(state: Float32Array): Promise<{ actionProbs: Float32Array; stateValue: number }>;
  /** Release native/wasm resources. Optional. */
  release?(): Promise<void> | void;
}

/** Which inference runtime to use. */
export type AbrRuntimeKind =
  /**
   * Pick automatically: `onnxruntime-web` if importable, else `onnxruntime-node`.
   * ORT-Web's WASM backend works in Node too, so this is portable by default.
   */
  'auto' | 'onnxruntime-web' | 'onnxruntime-node';

export interface AbrInferenceConfig {
  /** @default 'auto' */
  readonly runtime?: AbrRuntimeKind;

  /**
   * ONNX Runtime execution providers, in preference order.
   *
   * Browser: `['wasm']` is the safe default and what this package is validated
   * against. `'webgpu'` and `'webgl'` are accepted but **not validated** for
   * numerical parity — this model is 265k parameters, so GPU offload costs more
   * in dispatch overhead than it saves.
   *
   * @default ['wasm']
   */
  readonly executionProviders?: readonly string[];

  /**
   * WASM thread count. This model is tiny; 1 thread avoids the
   * cross-origin-isolation requirement that multi-threading imposes in browsers.
   * @default 1
   */
  readonly wasmThreads?: number;

  /**
   * Base URL/path for ONNX Runtime's `.wasm` assets. Forwarded to
   * `ort.env.wasm.wasmPaths`. Leave unset to use the runtime's own default
   * (its CDN in the browser, its package directory in Node).
   */
  readonly wasmPaths?: string;

  /**
   * Abandon an inference that exceeds this many milliseconds and fall back.
   * @default 250
   */
  readonly timeoutMs?: number;

  /**
   * Pre-warm the session with one dummy forward pass during `initialize()`, so
   * the first real decision is not slowed by lazy kernel compilation.
   * @default true
   */
  readonly warmup?: boolean;
}

/** Built-in fallback strategies. */
export type AbrFallbackStrategy =
  /**
   * Return no decision and let the player's own ABR run this tick. This is the
   * safest option and the default: for the Video.js adapter it re-invokes the
   * `selectPlaylist` implementation that was installed *before* this package
   * overrode it (normally `Vhs.STANDARD_PLAYLIST_SELECTOR`).
   */
  | 'player-default'
  /** Highest rendition whose bitrate fits under `estimatedThroughputBps * safetyFactor`. */
  | 'throughput'
  /** Buffer-occupancy rule, the `policy_buffer_based` heuristic from `src/test.py`. */
  | 'buffer'
  /** Always the lowest enabled rendition. Maximally safe, minimally useful. */
  | 'lowest'
  /** Hold whatever the player is currently on. */
  | 'hold'
  /** Your own function. */
  | AbrFallbackFn;

export type AbrFallbackFn = (
  observation: AbrObservation,
  reason: AbrFallbackReason,
) => string | null;

/**
 * Model-facing constants. Defaults are read from the exported model manifest,
 * which is generated from the research code — see `export/export_onnx.py`.
 */
export interface AbrModelSemantics {
  /**
   * Bitrate ladder, kbps, that the policy was trained on.
   * Default `[300, 750, 1200, 1850, 2850, 4300]` from `src/env.py::BITRATES`.
   *
   * The engine maps model action indices onto the *player's* ladder; see
   * `ladderMapping`.
   */
  readonly trainingLadderKbps?: readonly number[];

  /**
   * How to map the model's A_DIM action slots onto the player's actual ladder,
   * which usually has a different size and different bitrates.
   *
   * - `'nearest-bitrate'` (default): the model's action index picks a training
   *   bitrate, and the engine chooses the enabled player rendition whose
   *   bitrate is closest to it in log space. Preserves the *bitrate intent* of
   *   the trained policy.
   * - `'proportional-rank'`: action index `i` of `A_DIM` maps to rank
   *   `round(i * (n-1) / (A_DIM-1))` of the player's `n` enabled renditions.
   *   Preserves the *relative quality position* rather than the absolute rate.
   * - `'identity'`: action index is used directly as a ladder index. Only valid
   *   when the player ladder has exactly A_DIM entries in ascending order;
   *   otherwise the index is clamped.
   *
   * @default 'nearest-bitrate'
   */
  readonly ladderMapping?: 'nearest-bitrate' | 'proportional-rank' | 'identity';

  /**
   * `BUFFER_NORM` from `src/train.py`. Buffer seconds are divided by this before
   * entering the state tensor. Changing it changes model semantics.
   * @default 10
   */
  readonly bufferNormSec?: number;

  /**
   * `CHUNK_NORM` from `src/train.py`. Next-segment sizes in bytes are divided by
   * this. Changing it changes model semantics.
   * @default 1e6
   */
  readonly chunkNormBytes?: number;

  /**
   * `THROUGHPUT_NORM` from `src/train.py`. Throughput in **Mbps** is divided by
   * this. Changing it changes model semantics.
   * @default 1
   */
  readonly throughputNorm?: number;

  /**
   * `NUM_CHUNKS` from `src/env.py`; the divisor for the "remaining chunks"
   * input. Changing it changes model semantics.
   * @default 48
   */
  readonly totalChunksNorm?: number;

  /**
   * Nominal segment duration in seconds, used when the player has not reported
   * one yet. `VIDEO_CHUNK_LEN` in the research env.
   * @default 4
   */
  readonly segmentDurationSec?: Seconds;
}

export interface AbrTelemetryConfig {
  /**
   * Emit `decision` / `observation` / `error` events. Off by default: when off,
   * the engine skips building the telemetry payloads entirely (no array copies
   * of the model output, no state snapshot).
   * @default false
   */
  readonly enabled?: boolean;

  /**
   * Include the full `Float32Array` model input and output in `decision`
   * events. Costs two small array copies per decision.
   * @default false
   */
  readonly includeTensors?: boolean;

  /**
   * Retain the last N decisions in a ring buffer readable via
   * `engine.history()`. 0 disables retention.
   * @default 0
   */
  readonly historySize?: number;
}

export interface AbrConfig {
  /** Where the model comes from. @default 'ac3' */
  readonly model?: AbrModelSource;

  /** Inference runtime settings. */
  readonly inference?: AbrInferenceConfig;

  /** Model semantics / normalisation. Defaults come from the model manifest. */
  readonly semantics?: AbrModelSemantics;

  /**
   * Minimum wall-clock gap between two *model* evaluations, milliseconds.
   *
   * The trained controller makes one decision per downloaded segment. Adapters
   * drive it on segment-completion events, so this is a safety net against a
   * player that fires those events far more often than it downloads (VHS polls
   * `selectPlaylist` at 4 Hz when `bufferBasedABR` is on).
   *
   * `0` disables the guard. @default 0
   */
  readonly minDecisionIntervalMs?: number;

  /**
   * A decision older than this is considered stale and the fallback is used
   * instead. Protects against a hung inference leaving the player pinned to an
   * old rendition. Set to `Infinity` to disable.
   * @default 30000
   */
  readonly decisionTtlMs?: number;

  /** What to do when the model cannot decide. @default 'player-default' */
  readonly fallback?: AbrFallbackStrategy;

  /**
   * Multiplier applied to the throughput estimate by the `'throughput'`
   * fallback. @default 0.9
   */
  readonly throughputSafetyFactor?: number;

  /** Telemetry / debugging. */
  readonly telemetry?: AbrTelemetryConfig | boolean;

  /**
   * Log to `console` at debug level. Independent of telemetry.
   * @default false
   */
  readonly debug?: boolean;

  /**
   * Optional safety guardrail. **Off by default**, because enabling it
   * overrides the trained policy's action.
   *
   * Strongly consider enabling it for the bundled `ac3` controller: that model
   * selects action 4 in 51 of 56 sampled (throughput x buffer) states and does
   * not reliably back off on a collapsing link. See `src/core/safety.ts` and the
   * README section "Known limitations of the bundled model".
   */
  readonly safety?: AbrSafetyConfig;

  /**
   * Start disabled: the engine initialises and loads the model but every
   * `decide()` returns the fallback until `engine.enable()` is called. Lets an
   * application A/B the trained controller against the player default at
   * runtime. @default false
   */
  readonly startDisabled?: boolean;
}
