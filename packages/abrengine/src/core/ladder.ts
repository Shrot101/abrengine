/**
 * Reconciling the model's action space with the player's real bitrate ladder.
 * ===========================================================================
 *
 * The trained policy has a fixed action space of A_DIM = 6 slots that mean
 * `[300, 750, 1200, 1850, 2850, 4300]` kbps — the ladder from `src/env.py`.
 * A real stream almost never has that ladder. Something has to map between them,
 * and that mapping is a *semantic* choice, so it is explicit and configurable
 * rather than buried.
 *
 * Three strategies, all of which preserve the property that action index 0 is
 * the lowest-quality intent and A_DIM-1 the highest:
 *
 * - `nearest-bitrate` — the action names a target bitrate; pick the enabled
 *   rendition closest to it **in log space**. Log space because the reward
 *   function the policy was trained against is `log(BR / BR_min)`: an equal
 *   ratio, not an equal difference, is an equal perceptual step. Ties break
 *   downward (the safer rendition).
 * - `proportional-rank` — the action names a *position* in the ladder; map
 *   slot `i` of A_DIM onto rank `round(i * (n-1) / (A_DIM-1))` of the n enabled
 *   renditions. Use this when the stream's ladder spans a very different range
 *   and absolute bitrate matching would pin the policy to one end.
 * - `identity` — use the index directly, clamped. Only sensible when the player
 *   ladder is the training ladder.
 *
 * The inverse direction (player rendition -> action index) is needed too,
 * because the "last bitrate" model input and the next-segment-size vector are
 * both expressed in action space.
 */

import type { AbrRepresentation } from '../types/observation.js';
import { A_DIM, TRAINING_LADDER_KBPS } from '../model/manifest.js';

export type LadderMapping = 'nearest-bitrate' | 'proportional-rank' | 'identity';

export interface LadderContext {
  /** Enabled renditions, ascending by bitrate. Never empty (callers check). */
  readonly ladder: readonly AbrRepresentation[];
  /** Training ladder in kbps, ascending, length A_DIM. */
  readonly trainingKbps: readonly number[];
  readonly mapping: LadderMapping;
}

const clamp = (v: number, lo: number, hi: number): number => (v < lo ? lo : v > hi ? hi : v);

/** log-distance between two positive bitrates. Symmetric, scale-free. */
const logDistance = (a: number, b: number): number =>
  Math.abs(Math.log(Math.max(a, 1)) - Math.log(Math.max(b, 1)));

export function makeLadderContext(
  ladder: readonly AbrRepresentation[],
  mapping: LadderMapping = 'nearest-bitrate',
  trainingKbps: readonly number[] = TRAINING_LADDER_KBPS,
): LadderContext {
  return { ladder, trainingKbps, mapping };
}

/**
 * Model action index -> a rendition of the player's ladder.
 *
 * @returns the chosen representation, or `undefined` if the ladder is empty.
 */
export function actionToRepresentation(
  ctx: LadderContext,
  actionIndex: number,
): AbrRepresentation | undefined {
  const n = ctx.ladder.length;
  if (n === 0) return undefined;

  const a = clamp(Math.round(actionIndex), 0, A_DIM - 1);

  switch (ctx.mapping) {
    case 'identity':
      return ctx.ladder[clamp(a, 0, n - 1)];

    case 'proportional-rank': {
      const slots = ctx.trainingKbps.length;
      if (slots <= 1) return ctx.ladder[0];
      const rank = Math.round((a * (n - 1)) / (slots - 1));
      return ctx.ladder[clamp(rank, 0, n - 1)];
    }

    case 'nearest-bitrate':
    default: {
      const targetBps =
        (ctx.trainingKbps[clamp(a, 0, ctx.trainingKbps.length - 1)] ?? 0) * 1000;
      let best = 0;
      let bestD = Infinity;
      for (let i = 0; i < n; i++) {
        const d = logDistance(ctx.ladder[i]!.bitrateBps as number, targetBps);
        // Strictly-less keeps the *lower* rendition on a tie, since the ladder
        // is ascending and we scan upward.
        if (d < bestD - 1e-12) {
          bestD = d;
          best = i;
        }
      }
      return ctx.ladder[best];
    }
  }
}

/**
 * Player rendition -> model action index. Inverse of {@link actionToRepresentation}.
 *
 * Exact for `identity`; nearest-in-log-space for `nearest-bitrate`; rank-scaled
 * for `proportional-rank`.
 */
export function representationToAction(ctx: LadderContext, representationId: string): number {
  const n = ctx.ladder.length;
  if (n === 0) return 0;

  const idx = ctx.ladder.findIndex((r) => r.id === representationId);
  if (idx < 0) return 0;

  switch (ctx.mapping) {
    case 'identity':
      return clamp(idx, 0, A_DIM - 1);

    case 'proportional-rank': {
      if (n === 1) return 0;
      return clamp(Math.round((idx * (A_DIM - 1)) / (n - 1)), 0, A_DIM - 1);
    }

    case 'nearest-bitrate':
    default: {
      const bps = ctx.ladder[idx]!.bitrateBps as number;
      let best = 0;
      let bestD = Infinity;
      for (let a = 0; a < ctx.trainingKbps.length; a++) {
        const d = logDistance(bps, (ctx.trainingKbps[a] ?? 0) * 1000);
        if (d < bestD - 1e-12) {
          bestD = d;
          best = a;
        }
      }
      return clamp(best, 0, A_DIM - 1);
    }
  }
}

/**
 * Next-segment sizes expressed in the model's action space.
 *
 * The model's row-2 input has one slot per action. For each action we resolve
 * the rendition it would select and report that rendition's next-segment size.
 * When the player supplies measured sizes we use them; otherwise we use the
 * standard estimate `bitrateBps * segmentDurationSec / 8`, which is how the
 * research environment generates chunk sizes in the first place.
 *
 * @param sizesById measured next-segment sizes in bytes, keyed by rendition id
 * @param segmentDurationSec duration of the next segment, seconds
 */
export function nextSegmentBytesByAction(
  ctx: LadderContext,
  segmentDurationSec: number,
  sizesById?: Readonly<Record<string, number>>,
): number[] {
  const out = new Array<number>(A_DIM).fill(0);
  if (ctx.ladder.length === 0) return out;

  for (let a = 0; a < A_DIM; a++) {
    const rep = actionToRepresentation(ctx, a);
    if (!rep) continue;
    const measured = sizesById?.[rep.id];
    out[a] =
      measured !== undefined && Number.isFinite(measured) && measured > 0
        ? measured
        : ((rep.bitrateBps as number) * segmentDurationSec) / 8;
  }
  return out;
}
