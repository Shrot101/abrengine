/**
 * End-to-end integration on the real model.
 *
 * Everything else mocks the session; this drives the actual exported ONNX graph
 * through a simulated streaming session against a mocked Video.js player, and
 * checks that the *observable behaviour* of the deployed controller matches what
 * the research evaluation reports.
 */

import { describe, expect, it } from 'vitest';

import { AbrEngine } from '../src/core/engine.js';
import { VideoJSAbrAdapter } from '../src/adapters/videojs/index.js';
import { A_DIM, TRAINING_LADDER_KBPS } from '../src/model/manifest.js';
import { loadModelBytes, makeObservation, trainingLadder } from './helpers.js';
import { MockPlayer } from './mock-videojs.js';

const realEngine = (extra: Record<string, unknown> = {}): AbrEngine =>
  new AbrEngine({
    model: { type: 'buffer', buffer: loadModelBytes() },
    telemetry: { enabled: true, historySize: 200 },
    ...extra,
  });

describe('integration — the real trained controller', () => {
  it('loads the bundled ONNX model and decides from it', async () => {
    const abr = realEngine();
    await abr.initialize();
    expect(abr.status).toBe('ready');

    const d = await abr.decide(makeObservation());
    expect(d.source).toBe('model');
    expect(d.actionIndex).toBeGreaterThanOrEqual(0);
    expect(d.actionIndex).toBeLessThan(A_DIM);
    expect(d.model!.actionProbs).toHaveLength(A_DIM);
    // Softmax output, so probabilities sum to 1.
    expect(d.model!.actionProbs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    await abr.destroy();
  });

  it('reproduces the learned "start safe, then ramp and hold" behaviour', async () => {
    // The research README describes the trained policy as: begin at the lowest
    // rung from a cold start, jump once a throughput estimate exists, then hold.
    // If the port had broken feature ordering or normalisation, this would not
    // survive.
    const abr = realEngine();
    await abr.initialize();

    const ladder = trainingLadder();
    const chosen: number[] = [];
    let current = ladder[0]!.id;
    let buffer = 0;

    for (let i = 0; i < 12; i++) {
      const d = await abr.decide(
        makeObservation({
          representations: ladder,
          currentRepresentationId: current,
          bufferSec: buffer,
          currentTimeSec: i * 4,
          remainingSegments: 48 - i,
          // A healthy ~4 Mbps link: 600 kB in 1.2 s.
          segmentBytes: i === 0 ? null : 600_000,
          downloadSec: 1.2,
          segmentDurationSec: 4,
        }),
      );
      chosen.push(d.actionIndex!);
      current = d.representationId;
      buffer = Math.min(60, buffer + 4 - 1.2);
    }

    expect(chosen[0]).toBe(0); // cold start is conservative
    expect(Math.max(...chosen.slice(1))).toBeGreaterThan(0); // it ramps up
    // and then holds: no oscillation across the last third of the session.
    const tail = chosen.slice(-4);
    expect(new Set(tail).size).toBe(1);
    await abr.destroy();
  });

  /**
   * DOCUMENTED LIMITATION — not a bug in the port.
   *
   * The shipped `ac3` controller does NOT reliably back off when the link
   * collapses. Driven from 8 Mbps down to 0.67 Mbps with the buffer draining to
   * zero, it returns to action 4 (2850 kbps) within two steps and holds it with
   * ~98.5% confidence. A 56-point sweep of (throughput x buffer) states puts the
   * argmax at action 4 in 51 of them.
   *
   * Verified to be a property of the trained weights, not the conversion: the
   * same state fed to the original PyTorch checkpoint yields the same argmax,
   * and `test/parity.test.ts` shows agreement to 7.7e-7 across 296 fixtures.
   *
   * This test pins the observed behaviour so that a future retrain which fixes
   * it makes the test fail loudly rather than silently changing what ships.
   */
  it('holds a high rung even on a collapsing link (known model limitation)', async () => {
    const abr = realEngine();
    await abr.initialize();

    let current = trainingLadder()[0]!.id;
    const chosen: number[] = [];
    for (let i = 0; i < 10; i++) {
      const d = await abr.decide(
        makeObservation({
          currentRepresentationId: current,
          bufferSec: Math.max(0, 8 - i),
          // 1 MB taking 12 s: 0.67 Mbps.
          segmentBytes: 1_000_000,
          downloadSec: 12,
        }),
      );
      current = d.representationId;
      chosen.push(d.actionIndex!);
    }
    // It settles high and stays there rather than dropping to rung 0.
    expect(chosen.slice(-4)).toEqual([4, 4, 4, 4]);
    await abr.destroy();
  });

  it('the optional safety guard fixes that, when the application opts in', async () => {
    const abr = realEngine({
      safety: { enabled: true, bufferFloorSec: 8, throughputFactor: 0.9 },
    });
    await abr.initialize();

    let current = trainingLadder()[0]!.id;
    let last!: Awaited<ReturnType<AbrEngine['decide']>>;
    for (let i = 0; i < 10; i++) {
      last = await abr.decide(
        makeObservation({
          currentRepresentationId: current,
          bufferSec: Math.max(0, 8 - i),
          segmentBytes: 1_000_000,
          downloadSec: 12,
        }),
      );
      current = last.representationId;
    }

    // 0.67 Mbps * 0.9 = 600 kbps ceiling -> the 300 kbps rung.
    expect(last.representationId).toBe('rep0');
    // The model still decided; the guard lowered the acted-on rendition, and
    // says so instead of hiding it.
    expect(last.source).toBe('model');
    expect(last.actionIndex).toBe(4);
    expect(last.safetyClamp).not.toBeNull();
    expect(last.safetyClamp!.fromRepresentationId).toBe('rep4');
    expect(last.safetyClamp!.toRepresentationId).toBe('rep0');
    await abr.destroy();
  });

  it('the safety guard stays out of the way on a healthy link', async () => {
    const abr = realEngine({ safety: { enabled: true } });
    await abr.initialize();
    let current = trainingLadder()[0]!.id;
    let last!: Awaited<ReturnType<AbrEngine['decide']>>;
    for (let i = 0; i < 8; i++) {
      last = await abr.decide(
        makeObservation({
          currentRepresentationId: current,
          bufferSec: 25,
          segmentBytes: 1_500_000,
          downloadSec: 1,
        }),
      );
      current = last.representationId;
    }
    expect(last.safetyClamp).toBeNull();
    expect(last.representationId).toBe('rep4');
    await abr.destroy();
  });

  it('drives a mocked Video.js player end to end', async () => {
    const player = new MockPlayer().setTime(0).setBuffer(0);
    const abr = realEngine();
    const adapter = new VideoJSAbrAdapter({ player, abr });
    await adapter.initialize();
    expect(adapter.active).toBe(true);

    const applied: string[] = [];
    adapter.on('apply', (e) => {
      if (e.applied) applied.push(e.decision.representationId);
    });

    for (let i = 0; i < 15; i++) {
      player.setTime(i * 4).setBuffer(Math.min(30, 4 + i * 2.5));
      await player.completeSegment(600_000, 1300);
      player.checkABR();
    }

    expect(applied.length).toBeGreaterThan(10);
    // Every applied decision named a real rendition of the player's ladder.
    for (const id of applied) {
      expect(player.playlists.map((p) => p.id)).toContain(id);
    }
    // The player was moved off the rendition it started on (pl0) by the model.
    expect(player.selections.every((s) => s !== undefined)).toBe(true);
    expect(player.selections.at(-1)!.id).not.toBe('pl0');

    const decisions = abr.history();
    // The very first tick fires on attach, before the model has finished
    // loading, so it is a `not-initialised` fallback by design. Everything
    // after that comes from the model.
    expect(decisions[0]!.decision.reason).toBe('not-initialised');
    expect(decisions.slice(1).every((d) => d.decision.source === 'model')).toBe(true);
    expect(decisions.every((d) => d.inferenceMs < 100)).toBe(true);

    adapter.destroy();
    await abr.destroy();
  });

  it('switching between custom ABR and player default is live-controllable', async () => {
    const player = new MockPlayer().setTime(0).setBuffer(10);
    const abr = realEngine();
    const adapter = new VideoJSAbrAdapter({ player, abr });
    await adapter.initialize();

    await player.completeSegment(600_000, 1200);
    const withModel = abr.history().at(-1)!.decision.source;
    expect(withModel).toBe('model');

    abr.disable();
    await player.completeSegment(600_000, 1200);
    expect(abr.history().at(-1)!.decision.reason).toBe('disabled');

    abr.enable();
    await player.completeSegment(600_000, 1200);
    expect(abr.history().at(-1)!.decision.source).toBe('model');

    adapter.destroy();
    await abr.destroy();
  });

  it('meets the inference latency budget for per-segment decisions', async () => {
    const abr = realEngine();
    await abr.initialize();

    const latencies: number[] = [];
    for (let i = 0; i < 100; i++) {
      const d = await abr.decide(makeObservation({ bufferSec: 5 + (i % 20) }));
      latencies.push(d.inferenceMs);
    }
    latencies.sort((a, b) => a - b);
    const p50 = latencies[50]!;
    const p95 = latencies[95]!;
    console.log(`inference latency: p50=${p50.toFixed(3)}ms p95=${p95.toFixed(3)}ms`);

    // A decision happens roughly once per segment (~4 s). Anything under 50 ms
    // is three orders of magnitude of headroom; this asserts the budget, not
    // the machine's exact speed.
    expect(p95).toBeLessThan(50);
    await abr.destroy();
  });

  it('produces the same decision for the same state regardless of history path', async () => {
    const a = realEngine();
    const b = realEngine();
    await Promise.all([a.initialize(), b.initialize()]);

    const seq = [
      { segmentBytes: 400_000, downloadSec: 1.0, bufferSec: 8 },
      { segmentBytes: 900_000, downloadSec: 1.4, bufferSec: 14 },
      { segmentBytes: 700_000, downloadSec: 0.9, bufferSec: 19 },
    ];
    let da, db;
    for (const s of seq) da = await a.decide(makeObservation(s));
    for (const s of seq) db = await b.decide(makeObservation(s));
    expect(db!.representationId).toBe(da!.representationId);
    expect(db!.model!.actionProbs).toEqual(da!.model!.actionProbs);

    await Promise.all([a.destroy(), b.destroy()]);
  });

  it('exposes a ladder mapping that keeps the model usable on non-training ladders', async () => {
    const abr = realEngine();
    await abr.initialize();
    const ladders = [
      [500_000, 2_000_000, 8_000_000],
      [200_000, 400_000, 800_000, 1_600_000, 3_200_000, 6_400_000, 12_800_000],
      [1_000_000],
    ];
    for (const bitrates of ladders) {
      const reps = bitrates.map((bps, i) => ({
        id: `x${i}`,
        bitrateBps: bps as never,
        enabled: true,
      }));
      const d = await abr.decide(makeObservation({ representations: reps }));
      expect(d.source).toBe('model');
      expect(bitrates).toContain(d.bitrateBps as unknown as number);
    }
    await abr.destroy();
  });

  it('keeps the training ladder in the manifest consistent with what it selects', async () => {
    const abr = realEngine();
    await abr.initialize();
    const d = await abr.decide(makeObservation());
    expect(d.bitrateBps).toBe(TRAINING_LADDER_KBPS[d.actionIndex!]! * 1000);
    await abr.destroy();
  });
});
