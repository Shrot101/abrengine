/**
 * Fallback strategies.
 *
 * A fallback returns a representation **id**, or `null` meaning "I decline —
 * leave the player's own algorithm in charge for this tick". The `null` case is
 * what `'player-default'` always does, and it is the safest possible behaviour:
 * the player keeps doing exactly what it would have done without this package.
 */

import type { AbrFallbackFn, AbrFallbackStrategy } from '../types/config.js';
import type { AbrFallbackReason } from '../types/decision.js';
import type { AbrObservation } from '../types/observation.js';
import { enabledLadder } from '../types/observation.js';

export interface FallbackContext {
  /** Multiplier applied to the throughput estimate by `'throughput'`. */
  throughputSafetyFactor: number;
}

/** Highest rendition that fits under `estimate * safetyFactor`; lowest if none fit. */
function throughputStrategy(obs: AbrObservation, ctx: FallbackContext): string | null {
  const ladder = enabledLadder(obs.representations);
  if (ladder.length === 0) return null;

  const measured =
    obs.lastSegment && (obs.lastSegment.downloadSec as number) > 0
      ? ((obs.lastSegment.sizeBytes as number) * 8) / (obs.lastSegment.downloadSec as number)
      : null;
  const estimate = (obs.estimatedThroughputBps as number | undefined) ?? measured;
  if (estimate === null || estimate === undefined || !Number.isFinite(estimate)) {
    return ladder[0]!.id;
  }

  const budget = estimate * ctx.throughputSafetyFactor;
  let chosen = ladder[0]!;
  for (const r of ladder) {
    if ((r.bitrateBps as number) <= budget) chosen = r;
    else break;
  }
  return chosen.id;
}

/**
 * The buffer-occupancy heuristic from `src/test.py::policy_buffer_based`,
 * generalised from a 6-rung ladder to an n-rung one:
 *   buffer < 5 s  -> lowest
 *   buffer > 40 s -> highest
 *   otherwise     -> linear interpolation across the ladder
 */
function bufferStrategy(obs: AbrObservation): string | null {
  const ladder = enabledLadder(obs.representations);
  if (ladder.length === 0) return null;

  const buf = obs.playback?.bufferSec as number | undefined;
  if (buf === undefined || !Number.isFinite(buf)) return ladder[0]!.id;

  const LOW = 5.0;
  const HIGH = 40.0;
  if (buf < LOW) return ladder[0]!.id;
  if (buf > HIGH) return ladder[ladder.length - 1]!.id;

  const idx = Math.min(
    Math.floor(((buf - LOW) / (HIGH - LOW)) * (ladder.length - 1)),
    ladder.length - 1,
  );
  return ladder[Math.max(idx, 0)]!.id;
}

function lowestStrategy(obs: AbrObservation): string | null {
  const ladder = enabledLadder(obs.representations);
  return ladder.length > 0 ? ladder[0]!.id : null;
}

function holdStrategy(obs: AbrObservation): string | null {
  const cur = obs.currentRepresentationId;
  if (!cur) return null;
  const rep = obs.representations.find((r) => r.id === cur);
  return rep && rep.enabled ? cur : lowestStrategy(obs);
}

/** Compile a strategy name (or user function) into a callable. */
export function compileFallback(
  strategy: AbrFallbackStrategy | undefined,
  ctx: FallbackContext,
): AbrFallbackFn {
  if (typeof strategy === 'function') return strategy;

  switch (strategy ?? 'player-default') {
    case 'player-default':
      // Decline. The adapter interprets `null` as "run the player's own selector".
      return () => null;
    case 'throughput':
      return (obs) => throughputStrategy(obs, ctx);
    case 'buffer':
      return (obs) => bufferStrategy(obs);
    case 'lowest':
      return (obs) => lowestStrategy(obs);
    case 'hold':
      return (obs) => holdStrategy(obs);
    default:
      return () => null;
  }
}

export type { AbrFallbackReason };
