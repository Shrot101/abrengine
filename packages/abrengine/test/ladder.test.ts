import { describe, expect, it } from 'vitest';

import {
  actionToRepresentation,
  makeLadderContext,
  nextSegmentBytesByAction,
  representationToAction,
} from '../src/core/ladder.js';
import { A_DIM, TRAINING_LADDER_KBPS } from '../src/model/manifest.js';
import { bitsPerSecond } from '../src/types/units.js';
import type { AbrRepresentation } from '../src/types/observation.js';
import { customLadder, trainingLadder } from './helpers.js';

const rep = (id: string, bps: number, enabled = true): AbrRepresentation => ({
  id,
  bitrateBps: bitsPerSecond(bps),
  enabled,
});

describe('ladder mapping — nearest-bitrate', () => {
  it('is the identity when the player ladder is the training ladder', () => {
    const ctx = makeLadderContext(trainingLadder(), 'nearest-bitrate');
    for (let a = 0; a < A_DIM; a++) {
      expect(actionToRepresentation(ctx, a)?.id).toBe(`rep${a}`);
    }
  });

  it('round-trips: representation → action → representation', () => {
    const ctx = makeLadderContext(trainingLadder(), 'nearest-bitrate');
    for (let a = 0; a < A_DIM; a++) {
      const r = actionToRepresentation(ctx, a)!;
      expect(representationToAction(ctx, r.id)).toBe(a);
    }
  });

  it('picks the log-nearest rung on a ladder that does not match training', () => {
    // Player ladder: 400k, 1.1M, 2.4M, 6M. Training: 300, 750, 1200, 1850, 2850, 4300 kbps.
    const ctx = makeLadderContext(customLadder(), 'nearest-bitrate');
    // action 0 (300 kbps) -> nearest in log space is 400k
    expect(actionToRepresentation(ctx, 0)?.id).toBe('c0');
    // action 1 (750 kbps): log|750/400|=0.629, log|750/1100|=0.383 -> c1
    expect(actionToRepresentation(ctx, 1)?.id).toBe('c1');
    // action 5 (4300 kbps): log|4300/2400|=0.583, log|4300/6000|=0.333 -> c3
    expect(actionToRepresentation(ctx, 5)?.id).toBe('c3');
  });

  it('breaks ties toward the lower rendition', () => {
    // Geometric midpoint of 300 and 750 kbps is sqrt(300*750) ≈ 474.3 kbps.
    // Two rungs equidistant in log space from action 0's 300 kbps target:
    const ladder = [rep('lo', 150_000), rep('hi', 600_000)];
    const ctx = makeLadderContext(ladder, 'nearest-bitrate');
    expect(actionToRepresentation(ctx, 0)?.id).toBe('lo');
  });

  it('uses log space, not linear space', () => {
    // Target for action 0 is 300 kbps. Linear-nearest would pick 'far'
    // (|300-1000| = 700 vs |300-100| = 200 -> 'near' wins linearly too),
    // so use a case where they differ: target 1200 kbps (action 2),
    // candidates 400 kbps and 3000 kbps.
    //   linear: |1200-400| = 800  vs |1200-3000| = 1800  -> 400
    //   log:    ln(3)=1.099       vs ln(2.5)=0.916       -> 3000
    const ladder = [rep('low', 400_000), rep('high', 3_000_000)];
    const ctx = makeLadderContext(ladder, 'nearest-bitrate');
    expect(actionToRepresentation(ctx, 2)?.id).toBe('high');
  });
});

describe('ladder mapping — proportional-rank', () => {
  it('spreads A_DIM actions across an n-rung ladder', () => {
    const ctx = makeLadderContext(customLadder(), 'proportional-rank');
    // 6 actions over 4 rungs: ranks round(i*3/5) = 0,1,1,2,2,3
    expect([0, 1, 2, 3, 4, 5].map((a) => actionToRepresentation(ctx, a)!.id)).toEqual([
      'c0',
      'c1',
      'c1',
      'c2',
      'c2',
      'c3',
    ]);
  });

  it('maps the extremes to the extremes', () => {
    const ctx = makeLadderContext(customLadder(), 'proportional-rank');
    expect(actionToRepresentation(ctx, 0)!.id).toBe('c0');
    expect(actionToRepresentation(ctx, A_DIM - 1)!.id).toBe('c3');
  });

  it('handles a single-rung ladder', () => {
    const ctx = makeLadderContext([rep('only', 1e6)], 'proportional-rank');
    for (let a = 0; a < A_DIM; a++) expect(actionToRepresentation(ctx, a)!.id).toBe('only');
    expect(representationToAction(ctx, 'only')).toBe(0);
  });
});

describe('ladder mapping — identity', () => {
  it('uses the index directly', () => {
    const ctx = makeLadderContext(trainingLadder(), 'identity');
    expect(actionToRepresentation(ctx, 3)!.id).toBe('rep3');
  });

  it('clamps when the ladder is shorter than the action space', () => {
    const ctx = makeLadderContext(customLadder(), 'identity');
    expect(actionToRepresentation(ctx, 5)!.id).toBe('c3');
    expect(actionToRepresentation(ctx, 4)!.id).toBe('c3');
    expect(actionToRepresentation(ctx, 2)!.id).toBe('c2');
  });
});

describe('ladder mapping — boundaries', () => {
  it('clamps out-of-range action indices', () => {
    const ctx = makeLadderContext(trainingLadder());
    expect(actionToRepresentation(ctx, -5)!.id).toBe('rep0');
    expect(actionToRepresentation(ctx, 99)!.id).toBe(`rep${A_DIM - 1}`);
  });

  it('rounds fractional action indices', () => {
    const ctx = makeLadderContext(trainingLadder());
    expect(actionToRepresentation(ctx, 2.4)!.id).toBe('rep2');
    expect(actionToRepresentation(ctx, 2.6)!.id).toBe('rep3');
  });

  it('returns undefined for an empty ladder', () => {
    const ctx = makeLadderContext([]);
    expect(actionToRepresentation(ctx, 0)).toBeUndefined();
    expect(representationToAction(ctx, 'anything')).toBe(0);
  });

  it('returns action 0 for an unknown representation id', () => {
    const ctx = makeLadderContext(trainingLadder());
    expect(representationToAction(ctx, 'not-there')).toBe(0);
  });
});

describe('nextSegmentBytesByAction', () => {
  it('estimates bitrate × duration ÷ 8 when the player has no measured sizes', () => {
    const ctx = makeLadderContext(trainingLadder());
    const out = nextSegmentBytesByAction(ctx, 4);
    expect(out).toHaveLength(A_DIM);
    TRAINING_LADDER_KBPS.forEach((kbps, i) => {
      expect(out[i]).toBeCloseTo((kbps * 1000 * 4) / 8, 6);
    });
  });

  it('is the same formula the research env uses to synthesise chunk sizes', () => {
    // src/env.py: base = br * 1000 * VIDEO_CHUNK_LEN / 8
    const ctx = makeLadderContext(trainingLadder());
    const out = nextSegmentBytesByAction(ctx, 4);
    expect(out[0]).toBeCloseTo((300 * 1000 * 4) / 8, 6); // 150 000 bytes
    expect(out[5]).toBeCloseTo((4300 * 1000 * 4) / 8, 6); // 2 150 000 bytes
  });

  it('prefers measured sizes when supplied', () => {
    const ctx = makeLadderContext(trainingLadder());
    const out = nextSegmentBytesByAction(ctx, 4, { rep0: 1234, rep5: 999_999 });
    expect(out[0]).toBe(1234);
    expect(out[5]).toBe(999_999);
    expect(out[3]).toBeCloseTo((1850 * 1000 * 4) / 8, 6);
  });

  it('ignores non-finite or non-positive measured sizes', () => {
    const ctx = makeLadderContext(trainingLadder());
    const out = nextSegmentBytesByAction(ctx, 4, {
      rep0: Number.NaN,
      rep1: 0,
      rep2: -5,
    });
    expect(out[0]).toBeCloseTo((300 * 1000 * 4) / 8, 6);
    expect(out[1]).toBeCloseTo((750 * 1000 * 4) / 8, 6);
    expect(out[2]).toBeCloseTo((1200 * 1000 * 4) / 8, 6);
  });

  it('returns all zeros for an empty ladder', () => {
    expect(nextSegmentBytesByAction(makeLadderContext([]), 4)).toEqual(
      new Array(A_DIM).fill(0),
    );
  });
});
