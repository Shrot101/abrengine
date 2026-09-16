/** Shared test fixtures and fakes. */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import type { AbrObservation, AbrRepresentation } from '../src/types/observation.js';
import type { AbrInferenceSession } from '../src/types/config.js';
import { bitsPerSecond, bytes, seconds } from '../src/types/units.js';
import { A_DIM, TRAINING_LADDER_KBPS } from '../src/model/manifest.js';

const here = dirname(fileURLToPath(import.meta.url));

export const PKG_ROOT = resolve(here, '..');
export const REPO_ROOT = resolve(here, '../../..');
export const MODEL_PATH = resolve(PKG_ROOT, 'models/ac3-controller.onnx');
export const MANIFEST_PATH = resolve(PKG_ROOT, 'models/ac3-controller.json');
export const FIXTURES_PATH = resolve(REPO_ROOT, 'export/fixtures/parity-fixtures.json');

export interface ParityFixture {
  id: string;
  state: number[];
  expected: { actionProbs: number[]; stateValue: number; action: number };
}

export interface ParityFile {
  schema: { stateShape: [number, number]; aDim: number };
  checkpoint: string;
  count: number;
  cases: ParityFixture[];
}

export function loadParityFixtures(): ParityFile {
  return JSON.parse(readFileSync(FIXTURES_PATH, 'utf8')) as ParityFile;
}

export function loadManifest(): Record<string, any> {
  return JSON.parse(readFileSync(MANIFEST_PATH, 'utf8')) as Record<string, any>;
}

export function loadModelBytes(): Uint8Array {
  return new Uint8Array(readFileSync(MODEL_PATH));
}

/** A ladder matching the training bitrates, so `identity` mapping is exact. */
export function trainingLadder(): AbrRepresentation[] {
  return TRAINING_LADDER_KBPS.map((kbps, i) => ({
    id: `rep${i}`,
    bitrateBps: bitsPerSecond(kbps * 1000),
    heightPx: undefined,
    enabled: true,
  }));
}

/** An arbitrary 4-rung ladder that does not match training bitrates. */
export function customLadder(): AbrRepresentation[] {
  return [400_000, 1_100_000, 2_400_000, 6_000_000].map((bps, i) => ({
    id: `c${i}`,
    bitrateBps: bitsPerSecond(bps),
    enabled: true,
  }));
}

export interface ObsOverrides {
  representations?: AbrRepresentation[];
  currentRepresentationId?: string;
  bufferSec?: number;
  rebufferSec?: number;
  currentTimeSec?: number;
  durationSec?: number;
  segmentBytes?: number | null;
  downloadSec?: number;
  segmentDurationSec?: number;
  remainingSegments?: number | null;
  estimatedThroughputBps?: number;
  paused?: boolean;
  timestampMs?: number;
}

export function makeObservation(o: ObsOverrides = {}): AbrObservation {
  const reps = o.representations ?? trainingLadder();
  const cur = o.currentRepresentationId ?? reps[0]?.id ?? 'none';
  const segBytes = o.segmentBytes === undefined ? 500_000 : o.segmentBytes;

  return {
    timestampMs: o.timestampMs ?? 1000,
    representations: reps,
    currentRepresentationId: cur,
    playback: {
      bufferSec: seconds(o.bufferSec ?? 10),
      rebufferSec: seconds(o.rebufferSec ?? 0),
      currentTimeSec: seconds(o.currentTimeSec ?? 8),
      durationSec: seconds(o.durationSec ?? 192),
      paused: o.paused ?? false,
    },
    lastSegment:
      segBytes === null
        ? null
        : {
            sizeBytes: bytes(segBytes),
            downloadSec: seconds(o.downloadSec ?? 1),
            durationSec: seconds(o.segmentDurationSec ?? 4),
            representationId: cur,
          },
    ...(o.estimatedThroughputBps !== undefined
      ? { estimatedThroughputBps: bitsPerSecond(o.estimatedThroughputBps) }
      : {}),
    nextSegmentDurationSec: seconds(o.segmentDurationSec ?? 4),
    remainingSegments: o.remainingSegments === undefined ? 40 : o.remainingSegments,
  };
}

/** A session that always returns the given probabilities. */
export function fakeSession(
  probs: number[] = [0, 0, 0, 0, 1, 0],
  stateValue = 1.5,
): AbrInferenceSession & { calls: Float32Array[] } {
  const calls: Float32Array[] = [];
  return {
    calls,
    async run(state: Float32Array) {
      calls.push(state.slice());
      return { actionProbs: Float32Array.from(probs), stateValue };
    },
  };
}

/** A session that always throws. */
export function throwingSession(message = 'boom'): AbrInferenceSession {
  return {
    async run() {
      throw new Error(message);
    },
  };
}

/** A session that never resolves — for timeout tests. */
export function hangingSession(): AbrInferenceSession {
  return {
    run() {
      return new Promise(() => {
        /* never settles */
      });
    },
  };
}

export const ACTION_SLOTS = A_DIM;
