/**
 * Observation validation and repair.
 *
 * The engine must never propagate a NaN into the model, and must never throw
 * into a player's event handler. So every numeric field is checked, and every
 * bad field is either repaired to a defensible value (recording the repair for
 * telemetry) or, if the observation is structurally unusable, rejected so the
 * caller falls back.
 */

import type { AbrObservation, AbrRepresentation } from '../types/observation.js';

export interface ValidationResult {
  /** `false` means the observation cannot be used at all. */
  ok: boolean;
  /** Human-readable reason when `ok` is false. */
  problem?: string;
  /** Names of fields that were out of range and got clamped/defaulted. */
  repairs: string[];
  /** Values usable by the engine, after repair. */
  clean: CleanObservation;
}

/** Post-validation, all-finite view of an observation. */
export interface CleanObservation {
  timestampMs: number;
  ladder: readonly AbrRepresentation[];
  currentRepresentationId: string;
  bufferSec: number;
  rebufferSec: number;
  currentTimeSec: number;
  durationSec: number;
  /** null on cold start */
  segmentBytes: number | null;
  downloadSec: number | null;
  segmentDurationSec: number | null;
  throughputBpsMeasured: number | null;
  lastSegmentRepresentationId: string | null;
  estimatedThroughputBps: number | null;
  nextSegmentDurationSec: number | null;
  nextSegmentSizesBytes: Readonly<Record<string, number>> | null;
  remainingSegments: number | null;
  totalSegments: number | null;
}

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** Clamp to a range, recording a repair if the value moved or was unusable. */
function num(
  raw: unknown,
  fallback: number,
  lo: number,
  hi: number,
  name: string,
  repairs: string[],
): number {
  if (!finite(raw)) {
    repairs.push(`${name}:non-finite`);
    return fallback;
  }
  if (raw < lo) {
    repairs.push(`${name}:below-min`);
    return lo;
  }
  if (raw > hi) {
    repairs.push(`${name}:above-max`);
    return hi;
  }
  return raw;
}

/** Upper bounds chosen to be absurd-but-finite, so a broken player cannot poison the state. */
const MAX_BUFFER_SEC = 3600;
const MAX_DOWNLOAD_SEC = 600;
const MAX_SEGMENT_BYTES = 4e9;
const MAX_BITRATE_BPS = 1e10;
const MAX_SEGMENTS = 1e7;

export function validateObservation(obs: AbrObservation | null | undefined): ValidationResult {
  const repairs: string[] = [];
  const empty: CleanObservation = {
    timestampMs: 0,
    ladder: [],
    currentRepresentationId: '',
    bufferSec: 0,
    rebufferSec: 0,
    currentTimeSec: 0,
    durationSec: 0,
    segmentBytes: null,
    downloadSec: null,
    segmentDurationSec: null,
    throughputBpsMeasured: null,
    lastSegmentRepresentationId: null,
    estimatedThroughputBps: null,
    nextSegmentDurationSec: null,
    nextSegmentSizesBytes: null,
    remainingSegments: null,
    totalSegments: null,
  };

  if (!obs || typeof obs !== 'object') {
    return {
      ok: false,
      problem: 'observation is null or not an object',
      repairs,
      clean: empty,
    };
  }

  // ── Ladder ─────────────────────────────────────────────────────────────
  const rawReps = Array.isArray(obs.representations) ? obs.representations : [];
  const ladder: AbrRepresentation[] = [];
  for (const r of rawReps) {
    if (!r || typeof r.id !== 'string' || r.id.length === 0) {
      repairs.push('representation:missing-id');
      continue;
    }
    const bps = r.bitrateBps as unknown;
    if (!finite(bps) || bps <= 0 || bps > MAX_BITRATE_BPS) {
      repairs.push(`representation[${r.id}]:bad-bitrate`);
      continue;
    }
    ladder.push(r);
  }
  if (ladder.length === 0) {
    return {
      ok: false,
      problem: 'no representation with a usable bitrate',
      repairs,
      clean: { ...empty, timestampMs: finite(obs.timestampMs) ? obs.timestampMs : 0 },
    };
  }

  // ── Playback ───────────────────────────────────────────────────────────
  const pb = obs.playback ?? ({} as AbrObservation['playback']);
  const bufferSec = num(pb.bufferSec, 0, 0, MAX_BUFFER_SEC, 'playback.bufferSec', repairs);
  const rebufferSec = num(
    pb.rebufferSec,
    0,
    0,
    MAX_BUFFER_SEC,
    'playback.rebufferSec',
    repairs,
  );
  const currentTimeSec = num(
    pb.currentTimeSec,
    0,
    0,
    Number.MAX_SAFE_INTEGER,
    'playback.currentTimeSec',
    repairs,
  );
  // Infinity is legitimate here (live), so it is not a repair.
  const durationSec =
    typeof pb.durationSec === 'number' && !Number.isNaN(pb.durationSec)
      ? (pb.durationSec as number)
      : 0;

  // ── Last segment ───────────────────────────────────────────────────────
  let segmentBytes: number | null = null;
  let downloadSec: number | null = null;
  let segmentDurationSec: number | null = null;
  let throughputBpsMeasured: number | null = null;
  let lastSegmentRepresentationId: string | null = null;

  const ls = obs.lastSegment;
  if (ls) {
    const b = num(ls.sizeBytes, 0, 0, MAX_SEGMENT_BYTES, 'lastSegment.sizeBytes', repairs);
    const d = num(ls.downloadSec, 0, 0, MAX_DOWNLOAD_SEC, 'lastSegment.downloadSec', repairs);
    // A zero-byte or zero-duration download tells us nothing; treat it as cold
    // start rather than feeding a 0 or Infinity throughput into the history.
    if (b > 0 && d > 0) {
      segmentBytes = b;
      downloadSec = d;
      segmentDurationSec = num(
        ls.durationSec,
        0,
        0,
        MAX_DOWNLOAD_SEC,
        'lastSegment.durationSec',
        repairs,
      );
      const tp = ls.throughputBps as unknown;
      throughputBpsMeasured = finite(tp) && tp > 0 && tp < MAX_BITRATE_BPS ? tp : null;
      lastSegmentRepresentationId =
        typeof ls.representationId === 'string' ? ls.representationId : null;
    } else {
      repairs.push('lastSegment:degenerate');
    }
  }

  // ── Misc ───────────────────────────────────────────────────────────────
  const est = obs.estimatedThroughputBps as unknown;
  const estimatedThroughputBps = finite(est) && est > 0 && est < MAX_BITRATE_BPS ? est : null;

  const nsd = obs.nextSegmentDurationSec as unknown;
  const nextSegmentDurationSec = finite(nsd) && nsd > 0 && nsd <= MAX_DOWNLOAD_SEC ? nsd : null;

  let nextSegmentSizesBytes: Record<string, number> | null = null;
  if (obs.nextSegmentSizesBytes && typeof obs.nextSegmentSizesBytes === 'object') {
    nextSegmentSizesBytes = {};
    for (const [k, v] of Object.entries(obs.nextSegmentSizesBytes)) {
      if (finite(v) && v > 0 && v <= MAX_SEGMENT_BYTES) nextSegmentSizesBytes[k] = v as number;
      else repairs.push(`nextSegmentSizesBytes[${k}]:invalid`);
    }
    if (Object.keys(nextSegmentSizesBytes).length === 0) nextSegmentSizesBytes = null;
  }

  const rs = obs.remainingSegments;
  const remainingSegments =
    rs === null || rs === undefined
      ? null
      : num(rs, 0, 0, MAX_SEGMENTS, 'remainingSegments', repairs);

  const ts = obs.totalSegments;
  const totalSegments =
    ts === null || ts === undefined
      ? null
      : num(ts, 0, 0, MAX_SEGMENTS, 'totalSegments', repairs);

  const currentRepresentationId =
    typeof obs.currentRepresentationId === 'string' && obs.currentRepresentationId.length > 0
      ? obs.currentRepresentationId
      : (repairs.push('currentRepresentationId:missing'), (ladder[0] as AbrRepresentation).id);

  return {
    ok: true,
    repairs,
    clean: {
      timestampMs: finite(obs.timestampMs) ? obs.timestampMs : 0,
      ladder,
      currentRepresentationId,
      bufferSec,
      rebufferSec,
      currentTimeSec,
      durationSec,
      segmentBytes,
      downloadSec,
      segmentDurationSec,
      throughputBpsMeasured,
      lastSegmentRepresentationId,
      estimatedThroughputBps,
      nextSegmentDurationSec,
      nextSegmentSizesBytes,
      remainingSegments,
      totalSegments,
    },
  };
}

/** Model output sanity: right length, all finite, non-negative, sums to ~1. */
export function validateModelOutput(
  probs: Float32Array | readonly number[],
  aDim: number,
): string | null {
  if (probs.length !== aDim) return `expected ${aDim} probabilities, got ${probs.length}`;
  let sum = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i] as number;
    if (!Number.isFinite(p)) return `probability[${i}] is not finite`;
    if (p < -1e-6) return `probability[${i}] is negative (${p})`;
    sum += p;
  }
  if (Math.abs(sum - 1) > 1e-2) return `probabilities sum to ${sum}, expected ~1`;
  return null;
}
