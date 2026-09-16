/**
 * Telemetry event shapes.
 *
 * Everything here is optional work: when `telemetry.enabled` is false the
 * engine never allocates these objects.
 */

import type { AbrDecision, AbrFallbackReason } from './decision.js';
import type { AbrObservation } from './observation.js';
import type { BitsPerSecond, Bytes, Milliseconds, Seconds } from './units.js';

/** Emitted once per decision, model or fallback. */
export interface AbrDecisionEvent {
  readonly type: 'decision';
  /** `Date.now()` at emit time, milliseconds since epoch. */
  readonly timestamp: number;
  /** Monotonic timestamp, milliseconds. */
  readonly monotonicMs: number;

  readonly decision: AbrDecision;

  /** Forward buffer at decision time, seconds. */
  readonly bufferSec: Seconds;
  /** Throughput of the segment that triggered this decision, bits/s. `null` on cold start. */
  readonly throughputBps: BitsPerSecond | null;
  /** Download time of that segment, seconds. `null` on cold start. */
  readonly downloadSec: Seconds | null;
  /** Transferred size of that segment, bytes. `null` on cold start. */
  readonly segmentBytes: Bytes | null;

  /** Bitrate the player was on before this decision, bits/s. */
  readonly previousBitrateBps: BitsPerSecond | null;
  /** Bitrate this decision selects, bits/s. */
  readonly selectedBitrateBps: BitsPerSecond;
  /** Every rendition offered to the engine, bits/s, ascending. */
  readonly availableBitratesBps: readonly BitsPerSecond[];

  /** Time inside `session.run`, milliseconds. */
  readonly inferenceMs: Milliseconds;
  /** Time for the whole `decide()` call including state building, milliseconds. */
  readonly totalMs: Milliseconds;

  /**
   * The exact `[S_INFO * S_LEN]` tensor handed to the model, row-major.
   * Present only when `telemetry.includeTensors` is true.
   */
  readonly modelInput?: Float32Array;
  /** Softmax output. Present only when `telemetry.includeTensors` is true. */
  readonly modelOutput?: Float32Array;
}

/** Emitted when an observation is accepted, before inference. */
export interface AbrObservationEvent {
  readonly type: 'observation';
  readonly timestamp: number;
  readonly observation: AbrObservation;
  /** Fields that failed validation and were repaired. Empty when clean. */
  readonly repairs: readonly string[];
}

/** Emitted whenever the engine falls back. Always emitted, even with telemetry off. */
export interface AbrErrorEvent {
  readonly type: 'error';
  readonly timestamp: number;
  readonly reason: AbrFallbackReason;
  readonly message: string;
  readonly error?: unknown;
  /** `true` when the engine will keep trying; `false` when it has given up on the model. */
  readonly recoverable: boolean;
}

/** Emitted once when the model + runtime are ready. */
export interface AbrReadyEvent {
  readonly type: 'ready';
  readonly timestamp: number;
  /** Which runtime actually loaded. */
  readonly runtime: string;
  /** Execution providers actually requested. */
  readonly executionProviders: readonly string[];
  /** Time from `initialize()` to ready, milliseconds. */
  readonly loadMs: Milliseconds;
  /** Size of the model bytes, bytes. */
  readonly modelBytes: Bytes;
}

/** Emitted when `enable()` / `disable()` changes state. */
export interface AbrStateEvent {
  readonly type: 'state';
  readonly timestamp: number;
  readonly enabled: boolean;
}

export type AbrEvent =
  AbrDecisionEvent | AbrObservationEvent | AbrErrorEvent | AbrReadyEvent | AbrStateEvent;

export interface AbrEventMap {
  decision: AbrDecisionEvent;
  observation: AbrObservationEvent;
  error: AbrErrorEvent;
  ready: AbrReadyEvent;
  state: AbrStateEvent;
}
