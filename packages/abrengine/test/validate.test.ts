import { describe, expect, it } from 'vitest';

import { validateModelOutput, validateObservation } from '../src/core/validate.js';
import { A_DIM } from '../src/model/manifest.js';
import { bitsPerSecond, bytes, seconds } from '../src/types/units.js';
import type { AbrObservation } from '../src/types/observation.js';
import { makeObservation, trainingLadder } from './helpers.js';

describe('validateObservation — rejection', () => {
  it('rejects null and non-objects', () => {
    expect(validateObservation(null).ok).toBe(false);
    expect(validateObservation(undefined).ok).toBe(false);
    expect(validateObservation('nope' as unknown as AbrObservation).ok).toBe(false);
  });

  it('rejects an observation with no representations', () => {
    const r = validateObservation(makeObservation({ representations: [] }));
    expect(r.ok).toBe(false);
    expect(r.problem).toMatch(/no representation/);
  });

  it('rejects when every representation has an unusable bitrate', () => {
    const r = validateObservation(
      makeObservation({
        representations: [
          { id: 'a', bitrateBps: bitsPerSecond(Number.NaN), enabled: true },
          { id: 'b', bitrateBps: bitsPerSecond(0), enabled: true },
          { id: 'c', bitrateBps: bitsPerSecond(-1), enabled: true },
        ],
      }),
    );
    expect(r.ok).toBe(false);
    expect(r.repairs.length).toBe(3);
  });
});

describe('validateObservation — repair', () => {
  it('drops individual bad representations but keeps the good ones', () => {
    const reps = [
      ...trainingLadder(),
      { id: 'bad', bitrateBps: bitsPerSecond(Number.POSITIVE_INFINITY), enabled: true },
    ];
    const r = validateObservation(makeObservation({ representations: reps }));
    expect(r.ok).toBe(true);
    expect(r.clean.ladder).toHaveLength(A_DIM);
    expect(r.repairs).toContain('representation[bad]:bad-bitrate');
  });

  it('clamps a negative buffer to 0', () => {
    const r = validateObservation(makeObservation({ bufferSec: -5 }));
    expect(r.ok).toBe(true);
    expect(r.clean.bufferSec).toBe(0);
    expect(r.repairs).toContain('playback.bufferSec:below-min');
  });

  it('replaces a NaN buffer with 0', () => {
    const r = validateObservation(makeObservation({ bufferSec: Number.NaN }));
    expect(r.clean.bufferSec).toBe(0);
    expect(r.repairs).toContain('playback.bufferSec:non-finite');
  });

  it('clamps an absurd buffer', () => {
    const r = validateObservation(makeObservation({ bufferSec: 1e9 }));
    expect(r.clean.bufferSec).toBe(3600);
    expect(r.repairs).toContain('playback.bufferSec:above-max');
  });

  it('treats a zero-byte or zero-duration download as no download at all', () => {
    expect(
      validateObservation(makeObservation({ segmentBytes: 0 })).clean.segmentBytes,
    ).toBeNull();
    expect(
      validateObservation(makeObservation({ downloadSec: 0 })).clean.segmentBytes,
    ).toBeNull();
    expect(validateObservation(makeObservation({ downloadSec: 0 })).repairs).toContain(
      'lastSegment:degenerate',
    );
  });

  it('accepts Infinity as a live-stream duration without flagging a repair', () => {
    const r = validateObservation(makeObservation({ durationSec: Infinity }));
    expect(r.ok).toBe(true);
    expect(r.clean.durationSec).toBe(Infinity);
    expect(r.repairs.filter((x) => x.includes('duration'))).toHaveLength(0);
  });

  it('substitutes the lowest rendition when currentRepresentationId is missing', () => {
    const obs = { ...makeObservation(), currentRepresentationId: '' } as AbrObservation;
    const r = validateObservation(obs);
    expect(r.ok).toBe(true);
    expect(r.clean.currentRepresentationId).toBe('rep0');
    expect(r.repairs).toContain('currentRepresentationId:missing');
  });

  it('drops invalid entries from nextSegmentSizesBytes', () => {
    const obs = {
      ...makeObservation(),
      nextSegmentSizesBytes: { rep0: bytes(1000), rep1: bytes(Number.NaN), rep2: bytes(-1) },
    } as AbrObservation;
    const r = validateObservation(obs);
    expect(r.clean.nextSegmentSizesBytes).toEqual({ rep0: 1000 });
    expect(r.repairs).toContain('nextSegmentSizesBytes[rep1]:invalid');
  });

  it('nulls nextSegmentSizesBytes when nothing survives', () => {
    const obs = {
      ...makeObservation(),
      nextSegmentSizesBytes: { rep0: bytes(Number.NaN) },
    } as AbrObservation;
    expect(validateObservation(obs).clean.nextSegmentSizesBytes).toBeNull();
  });

  it('preserves null remainingSegments (live) rather than defaulting it', () => {
    expect(
      validateObservation(makeObservation({ remainingSegments: null })).clean.remainingSegments,
    ).toBeNull();
  });

  it('reports no repairs for a clean observation', () => {
    const r = validateObservation(makeObservation());
    expect(r.ok).toBe(true);
    expect(r.repairs).toEqual([]);
  });

  it('survives an observation with a missing playback block', () => {
    const obs = { ...makeObservation(), playback: undefined } as unknown as AbrObservation;
    const r = validateObservation(obs);
    expect(r.ok).toBe(true);
    expect(r.clean.bufferSec).toBe(0);
  });

  it('ignores a non-positive or absurd throughput estimate', () => {
    const obs = {
      ...makeObservation(),
      estimatedThroughputBps: bitsPerSecond(-1),
    } as AbrObservation;
    expect(validateObservation(obs).clean.estimatedThroughputBps).toBeNull();
  });

  it('carries a valid measured segment through unchanged', () => {
    const r = validateObservation(
      makeObservation({ segmentBytes: 123_456, downloadSec: 0.75, segmentDurationSec: 6 }),
    );
    expect(r.clean.segmentBytes).toBe(123_456);
    expect(r.clean.downloadSec).toBe(0.75);
    expect(r.clean.segmentDurationSec).toBe(6);
  });

  it('never throws on a deeply malformed object', () => {
    const junk = {
      timestampMs: 'x',
      representations: [null, 7, { id: 5 }],
      playback: { bufferSec: {} },
      lastSegment: { sizeBytes: [] },
    } as unknown as AbrObservation;
    expect(() => validateObservation(junk)).not.toThrow();
    expect(validateObservation(junk).ok).toBe(false);
  });
});

describe('validateModelOutput', () => {
  const uniform = new Array(A_DIM).fill(1 / A_DIM);

  it('accepts a well-formed softmax', () => {
    expect(validateModelOutput(Float32Array.from(uniform), A_DIM)).toBeNull();
  });

  it('rejects the wrong length', () => {
    expect(validateModelOutput(new Float32Array(3), A_DIM)).toMatch(/expected 6/);
  });

  it('rejects NaN', () => {
    const p = [...uniform];
    p[2] = Number.NaN;
    expect(validateModelOutput(p, A_DIM)).toMatch(/not finite/);
  });

  it('rejects Infinity', () => {
    const p = [...uniform];
    p[0] = Infinity;
    expect(validateModelOutput(p, A_DIM)).toMatch(/not finite/);
  });

  it('rejects a meaningfully negative probability', () => {
    const p = [...uniform];
    p[1] = -0.5;
    expect(validateModelOutput(p, A_DIM)).toMatch(/negative/);
  });

  it('tolerates a tiny negative value from float rounding', () => {
    // A real softmax can emit -1e-9 in a slot; the sum stays ~1.
    const p = new Array(A_DIM).fill(0);
    p[0] = -1e-9;
    p[1] = 0.5 + 1e-9;
    p[2] = 0.5;
    expect(validateModelOutput(p, A_DIM)).toBeNull();
  });

  it('rejects probabilities that do not sum to ~1', () => {
    expect(validateModelOutput(new Array(A_DIM).fill(1), A_DIM)).toMatch(/sum to 6/);
    expect(validateModelOutput(new Array(A_DIM).fill(0), A_DIM)).toMatch(/sum to 0/);
  });
});

describe('unit helpers', () => {
  it('keeps bits and bytes distinct at the type level and correct at runtime', () => {
    expect(bitsPerSecond(1_000_000)).toBe(1_000_000);
    expect(bytes(500)).toBe(500);
    expect(seconds(1.5)).toBe(1.5);
  });
});
