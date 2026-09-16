import { describe, expect, it } from 'vitest';

import { computeReward } from '../src/core/reward.js';
import { REWARD, TRAINING_LADDER_KBPS } from '../src/model/manifest.js';

const kbps = (k: number): number => k * 1000;
const MIN = kbps(TRAINING_LADDER_KBPS[0]!);

describe('computeReward — the QoE from src/env.py', () => {
  it('is zero at the lowest bitrate with no stall and no switch', () => {
    const r = computeReward(MIN, MIN, MIN, 0);
    expect(r.total).toBe(0);
    expect(r.quality).toBe(0);
  });

  it('uses log-scale quality', () => {
    // log(4300/300) = 2.6603
    const r = computeReward(kbps(4300), kbps(4300), MIN, 0);
    expect(r.quality).toBeCloseTo(Math.log(4300 / 300), 10);
    expect(r.total).toBeCloseTo(Math.log(4300 / 300), 10);
  });

  it('applies the 4.3 rebuffer penalty from the paper', () => {
    const r = computeReward(kbps(1200), kbps(1200), MIN, 1);
    expect(r.rebufferPenalty).toBeCloseTo(REWARD.rebufferPenalty, 10);
    expect(r.total).toBeCloseTo(Math.log(1200 / 300) - 4.3, 10);
  });

  it('applies the smoothness penalty on a switch', () => {
    const r = computeReward(kbps(2850), kbps(750), MIN, 0);
    const q = Math.log(2850 / 300);
    const qPrev = Math.log(750 / 300);
    expect(r.smoothnessPenalty).toBeCloseTo(Math.abs(q - qPrev), 10);
    expect(r.total).toBeCloseTo(q - Math.abs(q - qPrev), 10);
  });

  it('penalises a switch symmetrically in either direction', () => {
    const up = computeReward(kbps(2850), kbps(750), MIN, 0).smoothnessPenalty;
    const down = computeReward(kbps(750), kbps(2850), MIN, 0).smoothnessPenalty;
    expect(up).toBeCloseTo(down, 10);
  });

  it('makes one second of stall cost more than the largest possible quality gain', () => {
    // This is the property the README claims: λ=4.3 exceeds max quality 2.66.
    const maxQuality = Math.log(4300 / 300);
    expect(REWARD.rebufferPenalty).toBeGreaterThan(maxQuality);
    expect(computeReward(kbps(4300), kbps(4300), MIN, 1).total).toBeLessThan(0);
  });

  it('reproduces the exact Python expression on a worked example', () => {
    // src/env.py: reward = q_t - 4.3*rebuf - 1.0*abs(q_t - q_prev)
    const q = Math.log(1850 / 300);
    const qPrev = Math.log(300 / 300);
    const expected = q - 4.3 * 0.25 - 1.0 * Math.abs(q - qPrev);
    expect(computeReward(kbps(1850), kbps(300), MIN, 0.25).total).toBeCloseTo(expected, 12);
  });

  it('accepts custom weights', () => {
    const r = computeReward(kbps(1200), kbps(1200), MIN, 1, {
      rebufferPenalty: 10,
      smoothnessPenalty: 0,
    });
    expect(r.total).toBeCloseTo(Math.log(1200 / 300) - 10, 10);
  });

  it('treats negative rebuffer time as zero', () => {
    expect(computeReward(MIN, MIN, MIN, -5).rebufferPenalty).toBe(0);
  });

  it('does not produce NaN or -Infinity for degenerate bitrates', () => {
    for (const v of [0, -1, Number.NaN]) {
      const r = computeReward(v, v, v, 0);
      expect(Number.isFinite(r.total)).toBe(true);
    }
  });
});
