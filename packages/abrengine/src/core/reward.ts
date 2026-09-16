/**
 * The QoE reward from `src/env.py::step`.
 *
 * ```python
 * q_t    = math.log(BITRATES[a] / BITRATES[0])
 * q_prev = math.log(BITRATES[last] / BITRATES[0])
 * reward = q_t - 4.3 * rebuf - 1.0 * abs(q_t - q_prev)
 * ```
 *
 * The engine never uses this: the controller is trained offline and runs purely
 * as a policy at inference time — there is no online learning, so no reward is
 * needed to make a decision. It is exported because scoring live sessions with
 * the *same* QoE metric the policy was trained against is the only honest way to
 * compare it to the player's default ABR in production. `examples/telemetry`
 * uses it for exactly that.
 */

import { REWARD } from '../model/manifest.js';
import type { BitsPerSecond, Seconds } from '../types/units.js';

export interface RewardTerms {
  /** `log(bitrate / minBitrate)` — the quality term. */
  quality: number;
  /** `λ · rebufferSec`, already negated-in-sign-convention (i.e. a positive penalty). */
  rebufferPenalty: number;
  /** `μ · |q_t − q_prev|` (positive penalty). */
  smoothnessPenalty: number;
  /** `quality − rebufferPenalty − smoothnessPenalty`. */
  total: number;
}

export interface RewardOptions {
  /** λ, rebuffer weight. Default 4.3, from `src/env.py`. */
  rebufferPenalty?: number;
  /** μ, smoothness weight. Default 1.0, from `src/env.py`. */
  smoothnessPenalty?: number;
}

/**
 * Log-scale QoE for one segment.
 *
 * @param bitrateBps          bitrate of the segment just played, bits/s
 * @param previousBitrateBps  bitrate of the previous segment, bits/s
 * @param minBitrateBps       lowest rung of the ladder, bits/s (the log base point)
 * @param rebufferSec         stall time attributable to this segment, seconds
 */
export function computeReward(
  bitrateBps: BitsPerSecond | number,
  previousBitrateBps: BitsPerSecond | number,
  minBitrateBps: BitsPerSecond | number,
  rebufferSec: Seconds | number,
  options: RewardOptions = {},
): RewardTerms {
  const lambda = options.rebufferPenalty ?? REWARD.rebufferPenalty;
  const mu = options.smoothnessPenalty ?? REWARD.smoothnessPenalty;

  // Guard against non-finite / non-positive bitrates so a broken observation
  // yields a finite (if meaningless) score rather than NaN poisoning a running
  // QoE total.
  const positive = (v: number): number => (Number.isFinite(v) && v > 0 ? v : 1);

  const base = positive(minBitrateBps as number);
  const q = Math.log(positive(bitrateBps as number) / base);
  const qPrev = Math.log(positive(previousBitrateBps as number) / base);

  const rebuf = Number.isFinite(rebufferSec as number) ? Math.max(rebufferSec as number, 0) : 0;
  const rebufferPenalty = lambda * rebuf;
  const smoothnessPenalty = mu * Math.abs(q - qPrev);

  return {
    quality: q,
    rebufferPenalty,
    smoothnessPenalty,
    total: q - rebufferPenalty - smoothnessPenalty,
  };
}
