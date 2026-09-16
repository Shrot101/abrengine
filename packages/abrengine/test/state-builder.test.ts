/**
 * StateBuilder port fidelity.
 *
 * Replays the exact `update()` input sequences that `src/train.py::StateBuilder`
 * was given during real rollouts, and asserts the resulting tensor matches
 * Python's **exactly** — not within a tolerance.
 *
 * Exactness is the right bar here: both implementations perform the same
 * operations, in the same order, on a float32 buffer. Any difference means a
 * genuine semantic divergence (wrong shift direction, wrong divisor, wrong
 * index), not accumulated rounding.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { StateBuilder } from '../src/core/state-builder.js';
import { A_DIM, ROW, S_INFO, S_LEN, STATE_SIZE } from '../src/model/manifest.js';
import { REPO_ROOT } from './helpers.js';

interface StateFixtureFile {
  schema: { stateShape: [number, number]; aDim: number };
  normalisation: {
    bufferNormSec: number;
    chunkNormBytes: number;
    throughputNorm: number;
    totalChunksNorm: number;
  };
  episodes: Array<{
    id: string;
    steps: Array<{
      input: {
        segmentBytes: number;
        downloadSec: number;
        bufferSec: number;
        remainingSegments: number;
        lastActionIndex: number;
        nextSegmentBytesByAction: number[];
      };
      state: number[];
    }>;
  }>;
}

const fixtures: StateFixtureFile = JSON.parse(
  readFileSync(resolve(REPO_ROOT, 'export/fixtures/state-fixtures.json'), 'utf8'),
);

describe('StateBuilder — port of src/train.py::StateBuilder', () => {
  it('agrees with the Python normalisation constants', () => {
    // The TS defaults come from src/model/manifest.ts; the fixture values come
    // from src/train.py. If these ever diverge, every state tensor is wrong.
    const b = new StateBuilder();
    b.update({
      segmentBytes: 1e6,
      downloadSec: 1,
      bufferSec: fixtures.normalisation.bufferNormSec,
      remainingSegments: fixtures.normalisation.totalChunksNorm,
      lastActionIndex: A_DIM - 1,
      nextSegmentBytesByAction: new Array(A_DIM).fill(fixtures.normalisation.chunkNormBytes),
    });
    const rows = b.toRows();
    expect(rows[ROW.BUFFER]![S_LEN - 1]).toBeCloseTo(1, 6);
    expect(rows[ROW.REMAINING]![S_LEN - 1]).toBeCloseTo(1, 6);
    expect(rows[ROW.LAST_BITRATE]![S_LEN - 1]).toBeCloseTo(1, 6);
    expect(rows[ROW.CHUNK_SIZES]!.slice(0, A_DIM).every((v) => Math.abs(v - 1) < 1e-6)).toBe(
      true,
    );
  });

  for (const ep of fixtures.episodes) {
    it(`reproduces the Python tensor at every step of '${ep.id}' (${ep.steps.length} steps)`, () => {
      const b = new StateBuilder();
      ep.steps.forEach((step, i) => {
        b.update(step.input);
        const got = Array.from(b.get());
        expect(got.length).toBe(STATE_SIZE);
        // Compare as float32: the fixture values were produced by a float32
        // numpy array, so the JS float32 buffer must hold identical bits.
        const want = Float32Array.from(step.state);
        for (let k = 0; k < STATE_SIZE; k++) {
          expect(
            got[k],
            `${ep.id} step ${i} element ${k} (row ${Math.floor(k / S_LEN)}, col ${k % S_LEN})`,
          ).toBe(want[k]);
        }
      });
    });
  }
});

describe('StateBuilder — structural invariants', () => {
  it('starts at the all-zero state the research code starts each episode from', () => {
    const b = new StateBuilder();
    expect(Array.from(b.get()).every((v) => v === 0)).toBe(true);
    expect(b.updateCount).toBe(0);
  });

  it('reset() returns to the all-zero state', () => {
    const b = new StateBuilder();
    b.update({
      segmentBytes: 1e6,
      downloadSec: 1,
      bufferSec: 20,
      remainingSegments: 10,
      lastActionIndex: 3,
      nextSegmentBytesByAction: [1, 2, 3, 4, 5, 6],
    });
    expect(Array.from(b.get()).some((v) => v !== 0)).toBe(true);
    b.reset();
    expect(Array.from(b.get()).every((v) => v === 0)).toBe(true);
    expect(b.updateCount).toBe(0);
  });

  it('shifts history left, newest last — rows 0 and 1 only', () => {
    const b = new StateBuilder();
    for (let i = 1; i <= S_LEN + 2; i++) {
      b.update({
        // throughputMbps = segmentBytes*8/1e6 / downloadSec => i Mbps with these values
        segmentBytes: (i * 1e6) / 8,
        downloadSec: 1,
        bufferSec: 0,
        remainingSegments: 0,
        lastActionIndex: 0,
        nextSegmentBytesByAction: [],
      });
    }
    const rows = b.toRows();
    // After S_LEN+2 updates the window holds the last S_LEN values, ascending.
    const expected = Array.from({ length: S_LEN }, (_, k) => 3 + k);
    rows[ROW.THROUGHPUT]!.forEach((v, k) => expect(v).toBeCloseTo(expected[k]!, 4));
    // Row 1 holds the download times, all 1 here.
    expect(rows[ROW.DOWNLOAD]!.every((v) => v === 1)).toBe(true);
  });

  it('never writes rows 3–5 outside the last index', () => {
    const b = new StateBuilder();
    b.update({
      segmentBytes: 1e6,
      downloadSec: 2,
      bufferSec: 33,
      remainingSegments: 17,
      lastActionIndex: 4,
      nextSegmentBytesByAction: [1, 2, 3, 4, 5, 6],
    });
    const rows = b.toRows();
    for (const r of [ROW.BUFFER, ROW.REMAINING, ROW.LAST_BITRATE]) {
      for (let c = 0; c < S_LEN - 1; c++) {
        expect(rows[r]![c], `row ${r} col ${c} must stay 0`).toBe(0);
      }
      expect(rows[r]![S_LEN - 1]).not.toBe(0);
    }
  });

  it('writes chunk sizes only at indices 0..A_DIM-1', () => {
    const b = new StateBuilder();
    b.update({
      segmentBytes: 1,
      downloadSec: 1,
      bufferSec: 0,
      remainingSegments: 0,
      lastActionIndex: 0,
      nextSegmentBytesByAction: [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6],
    });
    const row = b.toRows()[ROW.CHUNK_SIZES]!;
    expect(row.slice(0, A_DIM)).toEqual([1, 2, 3, 4, 5, 6]);
    for (let c = A_DIM; c < S_LEN; c++) expect(row[c]).toBe(0);
  });

  it('derives throughput exactly as the research code does', () => {
    const b = new StateBuilder();
    // tp = bytes*8/1e6 / max(delay, 1e-6)
    b.update({
      segmentBytes: 250_000,
      downloadSec: 0.5,
      bufferSec: 0,
      remainingSegments: 0,
      lastActionIndex: 0,
      nextSegmentBytesByAction: [],
    });
    // 250000*8/1e6 = 2 Mbit, over 0.5 s = 4 Mbps
    expect(b.toRows()[ROW.THROUGHPUT]![S_LEN - 1]).toBeCloseTo(4, 6);
  });

  it('clamps a zero download time with the same 1e-6 epsilon as Python', () => {
    const b = new StateBuilder();
    b.update({
      segmentBytes: 1,
      downloadSec: 0,
      bufferSec: 0,
      remainingSegments: 0,
      lastActionIndex: 0,
      nextSegmentBytesByAction: [],
    });
    const v = b.toRows()[ROW.THROUGHPUT]![S_LEN - 1]!;
    expect(Number.isFinite(v)).toBe(true);
    expect(v).toBeCloseTo((1 * 8) / 1e6 / 1e-6, 3);
  });

  it('prefers a supplied throughput over the derived one', () => {
    const b = new StateBuilder();
    b.update({
      segmentBytes: 250_000,
      downloadSec: 0.5,
      throughputMbps: 12.5,
      bufferSec: 0,
      remainingSegments: 0,
      lastActionIndex: 0,
      nextSegmentBytesByAction: [],
    });
    expect(b.toRows()[ROW.THROUGHPUT]![S_LEN - 1]).toBeCloseTo(12.5, 6);
  });

  it('get() returns the live buffer by default and a copy on request', () => {
    const b = new StateBuilder();
    const live = b.get();
    const copy = b.get(true);
    b.update({
      segmentBytes: 1e6,
      downloadSec: 1,
      bufferSec: 5,
      remainingSegments: 1,
      lastActionIndex: 1,
      nextSegmentBytesByAction: [],
    });
    expect(live[ROW.BUFFER * S_LEN + S_LEN - 1]).not.toBe(0);
    expect(copy[ROW.BUFFER * S_LEN + S_LEN - 1]).toBe(0);
  });

  it('honours custom normalisation constants', () => {
    const b = new StateBuilder({ bufferNormSec: 20, totalChunksNorm: 100 });
    b.update({
      segmentBytes: 1,
      downloadSec: 1,
      bufferSec: 20,
      remainingSegments: 100,
      lastActionIndex: 0,
      nextSegmentBytesByAction: [],
    });
    const rows = b.toRows();
    expect(rows[ROW.BUFFER]![S_LEN - 1]).toBeCloseTo(1, 6);
    expect(rows[ROW.REMAINING]![S_LEN - 1]).toBeCloseTo(1, 6);
  });

  it('exposes the expected tensor geometry', () => {
    expect(S_INFO * S_LEN).toBe(STATE_SIZE);
    expect(new StateBuilder().get().length).toBe(STATE_SIZE);
  });
});
