/**
 * Optional safety guardrail — OFF BY DEFAULT.
 * ==========================================
 *
 * Why this exists
 * ---------------
 * The shipped `ac3` controller is **near-degenerate**: measured over a
 * 56-point sweep of (throughput × buffer) states, its argmax is action 4
 * (2850 kbps on the training ladder) in 51 of them, including states with
 * 0.1 Mbps of measured throughput and an empty buffer. It does not reliably
 * back off when the link collapses.
 *
 * This is a property of the trained weights, not of the JavaScript port — the
 * same states produce the same argmax in PyTorch, and the parity suite confirms
 * agreement to 7.7e-7. It is also consistent with the research evaluation in the
 * repository README, which reports ABREngine rebuffering for 1.89 s per episode
 * against the buffer-based heuristic's 0.43 s, and describes the learned policy
 * as "locked at 2850 kbps for the entire episode".
 *
 * Shipping that unguarded into a production player is a real risk, so this
 * module offers a guardrail. It is **off by default** because switching it on
 * overrides the trained policy's action, and rule one of this integration is
 * that the deployed controller behaves exactly like the researched one unless
 * the application deliberately says otherwise.
 *
 * What it does when enabled
 * -------------------------
 * After the model has chosen, and only when the observed state is genuinely
 * dangerous, the guard lowers the selection to something the measured
 * throughput can sustain. It never raises a selection. Every intervention is
 * reported through telemetry with `source: 'model'` but a non-null
 * `safetyClamp`, so you can always tell how often it fired — if it fires
 * constantly, the model is not doing useful work and you should know that
 * rather than have it hidden.
 */

import type { AbrRepresentation } from '../types/observation.js';

export interface AbrSafetyConfig {
  /**
   * Turn the guardrail on. @default false — the trained policy's action is used
   * verbatim.
   */
  readonly enabled?: boolean;

  /**
   * Only intervene when the forward buffer is below this many seconds. Above it
   * there is time to absorb a bad choice and the model is left alone.
   * @default 8
   */
  readonly bufferFloorSec?: number;

  /**
   * When intervening, cap the selection at `measuredThroughputBps × factor`.
   * @default 0.9
   */
  readonly throughputFactor?: number;

  /**
   * Never clamp below this rung index of the enabled ladder. `0` allows the
   * guard to drop all the way to the lowest rendition. @default 0
   */
  readonly minRungIndex?: number;
}

export interface ResolvedSafety {
  enabled: boolean;
  bufferFloorSec: number;
  throughputFactor: number;
  minRungIndex: number;
}

export function resolveSafety(cfg: AbrSafetyConfig | undefined): ResolvedSafety {
  return {
    enabled: cfg?.enabled === true,
    bufferFloorSec: cfg?.bufferFloorSec ?? 8,
    throughputFactor: cfg?.throughputFactor ?? 0.9,
    minRungIndex: Math.max(0, cfg?.minRungIndex ?? 0),
  };
}

/** Why the guard changed the model's choice. `null` when it did not. */
export interface SafetyClamp {
  /** The rendition the model actually chose. */
  readonly fromRepresentationId: string;
  /** The rendition the guard substituted. */
  readonly toRepresentationId: string;
  /** Forward buffer at the time, seconds. */
  readonly bufferSec: number;
  /** Measured throughput used for the cap, bits/s. */
  readonly throughputBps: number;
  /** The bitrate ceiling that was applied, bits/s. */
  readonly ceilingBps: number;
}

export interface SafetyInput {
  /** Enabled ladder, ascending by bitrate. */
  ladder: readonly AbrRepresentation[];
  /** What the model chose. */
  chosen: AbrRepresentation;
  /** Forward buffer, seconds. */
  bufferSec: number;
  /**
   * Throughput of the most recent completed segment, bits/s, or `null` when
   * there has not been one. The guard never fires without a measurement — it
   * will not act on a guess.
   */
  measuredThroughputBps: number | null;
}

/**
 * Apply the guardrail.
 *
 * @returns the representation to use, plus a `clamp` record when the guard
 *          intervened.
 */
export function applySafety(
  cfg: ResolvedSafety,
  input: SafetyInput,
): { representation: AbrRepresentation; clamp: SafetyClamp | null } {
  const { ladder, chosen, bufferSec, measuredThroughputBps } = input;

  if (
    !cfg.enabled ||
    ladder.length === 0 ||
    measuredThroughputBps === null ||
    !Number.isFinite(measuredThroughputBps) ||
    measuredThroughputBps <= 0 ||
    bufferSec >= cfg.bufferFloorSec
  ) {
    return { representation: chosen, clamp: null };
  }

  const ceiling = measuredThroughputBps * cfg.throughputFactor;
  if ((chosen.bitrateBps as number) <= ceiling) {
    return { representation: chosen, clamp: null };
  }

  // Highest rung at or below the ceiling, floored at minRungIndex.
  const floorIdx = Math.min(cfg.minRungIndex, ladder.length - 1);
  let idx = floorIdx;
  for (let i = floorIdx; i < ladder.length; i++) {
    if ((ladder[i]!.bitrateBps as number) <= ceiling) idx = i;
    else break;
  }

  const target = ladder[idx]!;
  // Only ever lower the selection.
  if ((target.bitrateBps as number) >= (chosen.bitrateBps as number)) {
    return { representation: chosen, clamp: null };
  }

  return {
    representation: target,
    clamp: {
      fromRepresentationId: chosen.id,
      toRepresentationId: target.id,
      bufferSec,
      throughputBps: measuredThroughputBps,
      ceilingBps: ceiling,
    },
  };
}
