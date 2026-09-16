/**
 * What the engine hands back to an adapter.
 */

import type { BitsPerSecond, Milliseconds } from './units.js';
import type { SafetyClamp } from '../core/safety.js';

/** Where a decision came from. Always present, so failures are never silent. */
export type AbrDecisionSource =
  /** The trained neural policy produced this. */
  | 'model'
  /** A fallback strategy produced this (see `reason`). */
  | 'fallback'
  /**
   * The engine explicitly declined to decide and the adapter should leave the
   * player's own algorithm in charge for this tick.
   */
  | 'player-default';

/** Why a fallback fired. `null` when `source === 'model'`. */
export type AbrFallbackReason =
  | 'not-initialised'
  | 'model-load-failed'
  | 'runtime-unavailable'
  | 'inference-error'
  | 'inference-timeout'
  | 'invalid-observation'
  | 'invalid-model-output'
  | 'empty-ladder'
  | 'disabled'
  | 'decision-stale';

/** The raw model output for one decision, retained for telemetry. */
export interface AbrModelOutput {
  /** Softmax over the A_DIM=6 action slots, in action-index order. */
  readonly actionProbs: readonly number[];
  /** The critic's value estimate V(s) for this state. Unbounded, dimensionless. */
  readonly stateValue: number;
  /** `argmax(actionProbs)`. This is the action the research code takes at eval time. */
  readonly actionIndex: number;
}

export interface AbrDecision {
  /** Id of the representation to download the next segment at. */
  readonly representationId: string;

  /** Declared bitrate of the chosen representation, bits per second. */
  readonly bitrateBps: BitsPerSecond;

  /** Where the decision came from. */
  readonly source: AbrDecisionSource;

  /** Populated when `source !== 'model'`. */
  readonly reason: AbrFallbackReason | null;

  /**
   * Index into the *model's* action space (0..A_DIM-1), or `null` when the
   * decision did not come from the model.
   *
   * Action semantics are preserved from training: index 0 is the lowest quality
   * of the ladder the model was trained on, index A_DIM-1 the highest.
   */
  readonly actionIndex: number | null;

  /** Raw model output, present only when `source === 'model'` and telemetry is on. */
  readonly model: AbrModelOutput | null;

  /**
   * Non-null when the optional safety guardrail lowered the model's choice.
   * `source` stays `'model'` — the model did decide — but the acted-on
   * rendition is the clamped one. Always `null` unless `config.safety.enabled`.
   */
  readonly safetyClamp: SafetyClamp | null;

  /** Wall-clock time spent inside `session.run`, milliseconds. */
  readonly inferenceMs: Milliseconds;

  /** `observation.timestampMs` of the observation that produced this decision. */
  readonly observedAtMs: number;

  /** Monotonic timestamp when the decision was finalised, milliseconds. */
  readonly decidedAtMs: number;
}
