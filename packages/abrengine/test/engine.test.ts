import { beforeEach, describe, expect, it, vi } from 'vitest';

import { AbrEngine } from '../src/core/engine.js';
import { A_DIM, ROW, S_LEN, TRAINING_LADDER_KBPS } from '../src/model/manifest.js';
import type { AbrDecisionEvent, AbrErrorEvent } from '../src/types/telemetry.js';
import {
  customLadder,
  fakeSession,
  hangingSession,
  loadModelBytes,
  makeObservation,
  throwingSession,
  trainingLadder,
} from './helpers.js';

/** Probabilities with all mass on one action. */
const onehot = (i: number): number[] =>
  Array.from({ length: A_DIM }, (_, k) => (k === i ? 1 : 0));

describe('AbrEngine — lifecycle', () => {
  it('starts idle and becomes ready after initialize()', async () => {
    const abr = new AbrEngine({ model: { type: 'session', session: fakeSession() } });
    expect(abr.status).toBe('idle');
    await abr.initialize();
    expect(abr.status).toBe('ready');
    expect(abr.runtime).toBe('custom');
    await abr.destroy();
    expect(abr.status).toBe('destroyed');
  });

  it('initialize() is idempotent and concurrency-safe', async () => {
    const session = fakeSession();
    const abr = new AbrEngine({ model: { type: 'session', session } });
    await Promise.all([abr.initialize(), abr.initialize(), abr.initialize()]);
    expect(abr.status).toBe('ready');
    // Exactly one warm-up pass, not three.
    expect(session.calls).toHaveLength(1);
    await abr.destroy();
  });

  it('warms up by default and can be told not to', async () => {
    const warm = fakeSession();
    const cold = fakeSession();
    await new AbrEngine({ model: { type: 'session', session: warm } }).initialize();
    await new AbrEngine({
      model: { type: 'session', session: cold },
      inference: { warmup: false },
    }).initialize();
    expect(warm.calls).toHaveLength(1);
    expect(cold.calls).toHaveLength(0);
  });

  it('does not reject when the model fails to load — it records failure', async () => {
    const abr = new AbrEngine({
      model: { type: 'url', url: 'http://127.0.0.1:1/missing.onnx' },
    });
    const errors: AbrErrorEvent[] = [];
    abr.on('error', (e) => errors.push(e));
    await expect(abr.initialize()).resolves.toBeUndefined();
    expect(abr.status).toBe('failed');
    expect(errors[0]?.reason).toBe('model-load-failed');
    expect(errors[0]?.recoverable).toBe(false);
  });

  it('rejects an unknown bundled model name', async () => {
    const abr = new AbrEngine({ model: 'nope' as unknown as 'ac3' });
    const errors: AbrErrorEvent[] = [];
    abr.on('error', (e) => errors.push(e));
    await abr.initialize();
    expect(abr.status).toBe('failed');
    expect(errors[0]?.message).toMatch(/unknown bundled model/);
  });

  it('emits ready with runtime and model size', async () => {
    const abr = new AbrEngine({ model: { type: 'buffer', buffer: loadModelBytes() } });
    const ready = vi.fn();
    abr.on('ready', ready);
    await abr.initialize();
    expect(ready).toHaveBeenCalledOnce();
    const ev = ready.mock.calls[0]![0] as { runtime: string; modelBytes: number };
    expect(ev.runtime).toBe('onnxruntime-web');
    expect(ev.modelBytes).toBeGreaterThan(1_000_000);
    await abr.destroy();
  });

  it('validates configuration eagerly', () => {
    expect(() => new AbrEngine({ semantics: { trainingLadderKbps: [1, 2] } })).toThrow(
      /exactly 6 entries/,
    );
    expect(() => new AbrEngine({ semantics: { bufferNormSec: 0 } })).toThrow(/non-zero/);
    expect(() => new AbrEngine({ semantics: { chunkNormBytes: Number.NaN } })).toThrow(
      /non-zero/,
    );
  });
});

describe('AbrEngine — decisions', () => {
  let abr: AbrEngine;

  beforeEach(async () => {
    abr = new AbrEngine({
      model: { type: 'session', session: fakeSession(onehot(4)) },
      telemetry: { enabled: true, historySize: 10, includeTensors: true },
    });
    await abr.initialize();
  });

  it('maps the argmax action onto the player ladder', async () => {
    const d = await abr.decide(makeObservation());
    expect(d.source).toBe('model');
    expect(d.actionIndex).toBe(4);
    expect(d.representationId).toBe('rep4');
    expect(d.bitrateBps).toBe(TRAINING_LADDER_KBPS[4]! * 1000);
    expect(d.reason).toBeNull();
  });

  it('never rejects, whatever the input', async () => {
    await expect(abr.decide(null as never)).resolves.toBeTruthy();
    await expect(abr.decide({} as never)).resolves.toBeTruthy();
    await expect(abr.decide(makeObservation({ representations: [] }))).resolves.toBeTruthy();
  });

  it('accumulates a rolling history across decisions', async () => {
    const before = abr.snapshotState();
    expect(Array.from(before).every((v) => v === 0)).toBe(true);

    await abr.decide(makeObservation({ segmentBytes: 250_000, downloadSec: 0.5 }));
    const after = abr.snapshotState();
    // 250000*8/1e6 / 0.5 = 4 Mbps in the newest throughput slot
    expect(after[ROW.THROUGHPUT * S_LEN + S_LEN - 1]).toBeCloseTo(4, 4);
    expect(after[ROW.DOWNLOAD * S_LEN + S_LEN - 1]).toBeCloseTo(0.5, 6);
  });

  it('does not update history on a cold-start observation', async () => {
    await abr.decide(makeObservation({ segmentBytes: null }));
    expect(Array.from(abr.snapshotState()).every((v) => v === 0)).toBe(true);
  });

  it('reset() clears the accumulated history', async () => {
    await abr.decide(makeObservation());
    expect(Array.from(abr.snapshotState()).some((v) => v !== 0)).toBe(true);
    abr.reset();
    expect(Array.from(abr.snapshotState()).every((v) => v === 0)).toBe(true);
  });

  it('records the last action index from the segment that was actually downloaded', async () => {
    await abr.decide(
      makeObservation({ currentRepresentationId: 'rep3', segmentBytes: 100_000 }),
    );
    const s = abr.snapshotState();
    expect(s[ROW.LAST_BITRATE * S_LEN + S_LEN - 1]).toBeCloseTo(3 / (A_DIM - 1), 6);
  });

  it('feeds ladder-derived next-segment sizes into row 2', async () => {
    await abr.decide(makeObservation({ segmentDurationSec: 4 }));
    const s = abr.snapshotState();
    // 300 kbps * 4 s / 8 = 150 000 bytes, ÷ CHUNK_NORM 1e6 = 0.15
    expect(s[ROW.CHUNK_SIZES * S_LEN + 0]).toBeCloseTo(0.15, 5);
    expect(s[ROW.CHUNK_SIZES * S_LEN + 5]).toBeCloseTo(2.15, 5);
  });

  it('works on a ladder that does not match the training bitrates', async () => {
    const d = await abr.decide(makeObservation({ representations: customLadder() }));
    expect(d.source).toBe('model');
    // action 4 = 2850 kbps target; log-nearest of {400k, 1.1M, 2.4M, 6M} is 2.4M
    expect(d.representationId).toBe('c2');
  });

  it('never selects a disabled rendition', async () => {
    const reps = trainingLadder().map((r) => (r.id === 'rep4' ? { ...r, enabled: false } : r));
    const d = await abr.decide(makeObservation({ representations: reps }));
    expect(d.representationId).not.toBe('rep4');
    expect(reps.find((r) => r.id === d.representationId)?.enabled).toBe(true);
  });

  it('observe() ingests without deciding', async () => {
    expect(abr.observe(makeObservation({ segmentBytes: 250_000, downloadSec: 0.5 }))).toBe(
      true,
    );
    expect(abr.snapshotState()[ROW.THROUGHPUT * S_LEN + S_LEN - 1]).toBeCloseTo(4, 4);
    expect(abr.currentDecision).toBeNull();
  });

  it('observe() returns false for cold-start and invalid observations', () => {
    expect(abr.observe(makeObservation({ segmentBytes: null }))).toBe(false);
    expect(abr.observe(makeObservation({ representations: [] }))).toBe(false);
    expect(abr.observe(null as never)).toBe(false);
  });
});

describe('AbrEngine — fallback behaviour', () => {
  it('falls back before initialize() and kicks off loading', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: fakeSession(onehot(5)) },
      fallback: 'lowest',
    });
    const d = await abr.decide(makeObservation());
    expect(d.source).toBe('fallback');
    expect(d.reason).toBe('not-initialised');
    expect(d.representationId).toBe('rep0');

    // The first decision started the load, so a later one uses the model.
    await abr.initialize();
    const d2 = await abr.decide(makeObservation());
    expect(d2.source).toBe('model');
  });

  it('falls back when inference throws, and stays usable afterwards', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: throwingSession('kaboom') },
      fallback: 'buffer',
      telemetry: true,
    });
    const errors: AbrErrorEvent[] = [];
    abr.on('error', (e) => errors.push(e));
    await abr.initialize();

    const d = await abr.decide(makeObservation({ bufferSec: 45 }));
    expect(d.source).toBe('fallback');
    expect(d.reason).toBe('inference-error');
    expect(d.representationId).toBe('rep5'); // buffer > 40 -> highest
    expect(errors.some((e) => e.message.includes('kaboom'))).toBe(true);
    expect(errors.at(-1)?.recoverable).toBe(true);

    // Still serving decisions, not wedged.
    const d2 = await abr.decide(makeObservation({ bufferSec: 2 }));
    expect(d2.representationId).toBe('rep0');
  });

  it('falls back on inference timeout without hanging', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: hangingSession() },
      inference: { timeoutMs: 30, warmup: false },
      fallback: 'lowest',
    });
    await abr.initialize();
    const t0 = Date.now();
    const d = await abr.decide(makeObservation());
    expect(Date.now() - t0).toBeLessThan(2000);
    expect(d.reason).toBe('inference-timeout');
    expect(d.source).toBe('fallback');
  });

  it('rejects a model output that is not a valid distribution', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: fakeSession([1, 1, 1, 1, 1, 1]) },
      fallback: 'lowest',
      inference: { warmup: false },
    });
    await abr.initialize();
    const d = await abr.decide(makeObservation());
    expect(d.reason).toBe('invalid-model-output');
  });

  it("'player-default' declines, signalling the adapter to defer", async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: throwingSession() },
      fallback: 'player-default',
    });
    await abr.initialize();
    const d = await abr.decide(makeObservation());
    expect(d.source).toBe('player-default');
    expect(d.representationId).toBe('');
    expect(d.reason).toBe('inference-error');
  });

  it("'throughput' picks the highest rung under the safety-adjusted estimate", async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: throwingSession() },
      fallback: 'throughput',
      throughputSafetyFactor: 0.9,
    });
    await abr.initialize();
    // 2 000 000 * 0.9 = 1 800 000 -> highest rung ≤ that is 1200 kbps (rep2)
    const d = await abr.decide(makeObservation({ estimatedThroughputBps: 2_000_000 }));
    expect(d.representationId).toBe('rep2');
  });

  it("'hold' keeps the current rendition", async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: throwingSession() },
      fallback: 'hold',
    });
    await abr.initialize();
    const d = await abr.decide(makeObservation({ currentRepresentationId: 'rep3' }));
    expect(d.representationId).toBe('rep3');
  });

  it('accepts a custom fallback function and survives it throwing', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: throwingSession() },
      fallback: (obs) => obs.representations[1]!.id,
    });
    await abr.initialize();
    expect((await abr.decide(makeObservation())).representationId).toBe('rep1');

    const boom = new AbrEngine({
      model: { type: 'session', session: throwingSession() },
      fallback: () => {
        throw new Error('bad fallback');
      },
    });
    await boom.initialize();
    const d = await boom.decide(makeObservation());
    expect(d.source).toBe('player-default');
  });

  it('declines when a fallback names an unknown rendition', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: throwingSession() },
      fallback: () => 'does-not-exist',
    });
    await abr.initialize();
    expect((await abr.decide(makeObservation())).source).toBe('player-default');
  });

  it('falls back permanently once the model has failed to load', async () => {
    const abr = new AbrEngine({
      model: { type: 'url', url: 'http://127.0.0.1:1/x.onnx' },
      fallback: 'lowest',
    });
    await abr.initialize();
    for (let i = 0; i < 3; i++) {
      const d = await abr.decide(makeObservation());
      expect(d.reason).toBe('model-load-failed');
      expect(d.representationId).toBe('rep0');
    }
  });
});

describe('AbrEngine — enable/disable', () => {
  it('serves fallbacks while disabled and the model once enabled', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: fakeSession(onehot(5)) },
      startDisabled: true,
      fallback: 'lowest',
    });
    await abr.initialize();
    expect(abr.enabled).toBe(false);

    const off = await abr.decide(makeObservation());
    expect(off.source).toBe('fallback');
    expect(off.reason).toBe('disabled');

    const states: boolean[] = [];
    abr.on('state', (e) => states.push(e.enabled));
    abr.enable();
    expect(states).toEqual([true]);

    const on = await abr.decide(makeObservation());
    expect(on.source).toBe('model');
    expect(on.representationId).toBe('rep5');

    abr.disable();
    expect((await abr.decide(makeObservation())).source).toBe('fallback');
  });

  it('does not emit duplicate state events', async () => {
    const abr = new AbrEngine({ model: { type: 'session', session: fakeSession() } });
    const states: boolean[] = [];
    abr.on('state', (e) => states.push(e.enabled));
    abr.enable();
    abr.enable();
    abr.disable();
    abr.disable();
    expect(states).toEqual([false]);
  });

  it('keeps the history warm while disabled', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: fakeSession() },
      startDisabled: true,
      fallback: 'lowest',
    });
    await abr.initialize();
    await abr.decide(makeObservation({ segmentBytes: 250_000, downloadSec: 0.5 }));
    expect(abr.snapshotState()[ROW.THROUGHPUT * S_LEN + S_LEN - 1]).toBeCloseTo(4, 4);
  });
});

describe('AbrEngine — decision rate limiting', () => {
  it('reuses the last decision inside minDecisionIntervalMs', async () => {
    const session = fakeSession(onehot(2));
    const abr = new AbrEngine({
      model: { type: 'session', session },
      minDecisionIntervalMs: 10_000,
      inference: { warmup: false },
    });
    await abr.initialize();

    const a = await abr.decide(makeObservation());
    const b = await abr.decide(makeObservation());
    expect(session.calls).toHaveLength(1);
    expect(b).toBe(a);
  });

  it('still ingests observations on rate-limited ticks', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: fakeSession(onehot(2)) },
      minDecisionIntervalMs: 10_000,
      inference: { warmup: false },
    });
    await abr.initialize();
    await abr.decide(makeObservation({ segmentBytes: 125_000, downloadSec: 0.5 }));
    await abr.decide(makeObservation({ segmentBytes: 250_000, downloadSec: 0.5 }));
    const s = abr.snapshotState();
    // Both observations are in the window: 2 Mbps then 4 Mbps.
    expect(s[ROW.THROUGHPUT * S_LEN + S_LEN - 2]).toBeCloseTo(2, 4);
    expect(s[ROW.THROUGHPUT * S_LEN + S_LEN - 1]).toBeCloseTo(4, 4);
  });

  it('does not rate limit by default', async () => {
    const session = fakeSession(onehot(2));
    const abr = new AbrEngine({
      model: { type: 'session', session },
      inference: { warmup: false },
    });
    await abr.initialize();
    await abr.decide(makeObservation());
    await abr.decide(makeObservation());
    expect(session.calls).toHaveLength(2);
  });
});

describe('AbrEngine — telemetry', () => {
  it('emits nothing when telemetry is off', async () => {
    const abr = new AbrEngine({ model: { type: 'session', session: fakeSession(onehot(1)) } });
    await abr.initialize();
    const seen = vi.fn();
    abr.on('decision', seen);
    await abr.decide(makeObservation());
    expect(seen).not.toHaveBeenCalled();
  });

  it('emits a fully populated decision event when telemetry is on', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: fakeSession(onehot(3), 2.75) },
      telemetry: { enabled: true, includeTensors: true },
    });
    await abr.initialize();

    const events: AbrDecisionEvent[] = [];
    abr.on('decision', (e) => events.push(e));
    await abr.decide(
      makeObservation({
        currentRepresentationId: 'rep1',
        bufferSec: 12,
        segmentBytes: 500_000,
        downloadSec: 2,
      }),
    );

    const e = events[0]!;
    expect(e.decision.source).toBe('model');
    expect(e.decision.representationId).toBe('rep3');
    expect(e.decision.model?.stateValue).toBe(2.75);
    expect(e.decision.model?.actionProbs).toHaveLength(A_DIM);
    expect(e.bufferSec).toBe(12);
    expect(e.segmentBytes).toBe(500_000);
    expect(e.downloadSec).toBe(2);
    expect(e.throughputBps).toBeCloseTo((500_000 * 8) / 2, 3);
    expect(e.previousBitrateBps).toBe(750_000);
    expect(e.selectedBitrateBps).toBe(1_850_000);
    expect(e.availableBitratesBps).toHaveLength(A_DIM);
    expect(e.modelInput).toBeInstanceOf(Float32Array);
    expect(e.modelOutput).toBeInstanceOf(Float32Array);
    expect(e.inferenceMs).toBeGreaterThanOrEqual(0);
    expect(e.totalMs).toBeGreaterThanOrEqual(0);
  });

  it('omits tensors unless includeTensors is set', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: fakeSession(onehot(3)) },
      telemetry: { enabled: true },
    });
    await abr.initialize();
    const events: AbrDecisionEvent[] = [];
    abr.on('decision', (e) => events.push(e));
    await abr.decide(makeObservation());
    expect(events[0]!.modelInput).toBeUndefined();
    expect(events[0]!.modelOutput).toBeUndefined();
  });

  it('emits observation events with the repairs that were applied', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: fakeSession(onehot(0)) },
      telemetry: true,
    });
    await abr.initialize();
    const repairs: string[][] = [];
    abr.on('observation', (e) => repairs.push([...e.repairs]));
    await abr.decide(makeObservation({ bufferSec: -3 }));
    expect(repairs[0]).toContain('playback.bufferSec:below-min');
  });

  it('keeps a bounded history ring', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: fakeSession(onehot(0)) },
      telemetry: { enabled: true, historySize: 3 },
    });
    await abr.initialize();
    for (let i = 0; i < 7; i++) await abr.decide(makeObservation());
    expect(abr.history()).toHaveLength(3);
  });

  it('records fallback decisions in telemetry too', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: throwingSession() },
      fallback: 'lowest',
      telemetry: { enabled: true, historySize: 5 },
    });
    await abr.initialize();
    await abr.decide(makeObservation());
    expect(abr.history()[0]!.decision.source).toBe('fallback');
    expect(abr.history()[0]!.decision.reason).toBe('inference-error');
  });

  it('survives a listener that throws', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: fakeSession(onehot(0)) },
      telemetry: true,
    });
    await abr.initialize();
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    abr.on('decision', () => {
      throw new Error('listener exploded');
    });
    const seen = vi.fn();
    abr.on('decision', seen);
    await expect(abr.decide(makeObservation())).resolves.toBeTruthy();
    expect(seen).toHaveBeenCalled();
    spy.mockRestore();
  });

  it('off() removes a listener', async () => {
    const abr = new AbrEngine({
      model: { type: 'session', session: fakeSession(onehot(0)) },
      telemetry: true,
    });
    await abr.initialize();
    const fn = vi.fn();
    abr.on('decision', fn);
    abr.off('decision', fn);
    await abr.decide(makeObservation());
    expect(fn).not.toHaveBeenCalled();
  });
});
