import { describe, expect, it, vi } from 'vitest';

import { AbrEngine } from '../src/core/engine.js';
import { VideoJSAbrAdapter } from '../src/adapters/videojs/index.js';
import { VhsBridge } from '../src/adapters/videojs/vhs-bridge.js';
import {
  buildObservation,
  createTracker,
  forwardBufferSec,
} from '../src/adapters/videojs/observation.js';
import { A_DIM } from '../src/model/manifest.js';
import { fakeSession, throwingSession } from './helpers.js';
import { MockPlayer, makePlaylists } from './mock-videojs.js';

const onehot = (i: number): number[] =>
  Array.from({ length: A_DIM }, (_, k) => (k === i ? 1 : 0));

function engineSelecting(action: number): AbrEngine {
  return new AbrEngine({
    model: { type: 'session', session: fakeSession(onehot(action)) },
    telemetry: true,
  });
}

const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

describe('VhsBridge — reading the player', () => {
  it('reports usable only when the required surface is present', () => {
    const p = new MockPlayer();
    expect(new VhsBridge(p.vhs).usable()).toBe(true);
    expect(new VhsBridge(null).usable()).toBe(false);
    expect(new VhsBridge({ stats: {} }).usable()).toBe(false);
  });

  it('reads the ladder, the current rendition and the bandwidth estimate', () => {
    const p = new MockPlayer();
    const b = new VhsBridge(p.vhs);
    expect(b.representations()).toHaveLength(6);
    expect(b.currentPlaylistId()).toBe('pl0');
    expect(b.bandwidthBps()).toBe(3_000_000);
  });

  it('prefers the main segment loader over the aggregate vhs.stats counters', () => {
    const p = new MockPlayer();
    expect(new VhsBridge(p.vhs).statsSource()).toBe('main-segment-loader');
  });

  it('falls back to vhs.stats when the private loader is absent', () => {
    const p = new MockPlayer();
    const vhs = p.vhs as unknown as { playlistController_: unknown };
    vhs.playlistController_ = {};
    const b = new VhsBridge(p.vhs);
    expect(b.statsSource()).toBe('vhs-stats');
    p.vhs!.addTransfer(1000, 200);
    expect(b.counters().bytes).toBe(1000);
  });

  it('reports "none" when neither counter source exists', () => {
    expect(new VhsBridge({ representations: () => [] }).statsSource()).toBe('none');
  });

  it('resolves a playlist by id, and returns undefined for an unknown one', () => {
    const b = new VhsBridge(new MockPlayer().vhs);
    expect(b.playlistById('pl3')?.attributes?.BANDWIDTH).toBe(1_850_000);
    expect(b.playlistById('nope')).toBeUndefined();
  });

  it('detects VOD vs live', () => {
    expect(new VhsBridge(new MockPlayer().vhs).isLive()).toBe(false);
    expect(
      new VhsBridge(
        new MockPlayer({ playlists: makePlaylists([1e6], { live: true }) }).vhs,
      ).isLive(),
    ).toBe(true);
  });

  it('reports segment geometry for VOD and null for live', () => {
    expect(new VhsBridge(new MockPlayer().vhs).segmentCount()).toBe(48);
    expect(new VhsBridge(new MockPlayer().vhs).targetDurationSec()).toBe(4);
    const live = new MockPlayer({ playlists: makePlaylists([1e6], { live: true }) });
    expect(new VhsBridge(live.vhs).segmentCount()).toBeNull();
  });

  it('never throws when the handler misbehaves', () => {
    const hostile = {
      representations() {
        throw new Error('nope');
      },
      playlists: {
        media() {
          throw new Error('nope');
        },
      },
    };
    const b = new VhsBridge(hostile);
    expect(b.representations()).toEqual([]);
    expect(b.currentPlaylist()).toBeUndefined();
    expect(() => b.describe()).not.toThrow();
  });
});

describe('observation building — player state → AbrObservation', () => {
  it('computes forward buffer relative to the playhead', () => {
    const p = new MockPlayer().setTime(10).setBufferedRanges([[0, 25]]);
    expect(forwardBufferSec(p)).toBe(15);
  });

  it('returns 0 forward buffer when the playhead is outside every range', () => {
    const p = new MockPlayer().setTime(100).setBufferedRanges([[0, 25]]);
    expect(forwardBufferSec(p)).toBe(0);
  });

  it('returns null before the player is ready', () => {
    const p = new MockPlayer({ attachVhs: false });
    expect(
      buildObservation(p, new VhsBridge(null), createTracker(), {
        defaultSegmentDurationSec: 4,
        liveRemainingSegments: null,
      }),
    ).toBeNull();
  });

  it('reports no lastSegment on the first sample, then diffs the counters', () => {
    const p = new MockPlayer().setTime(4).setBuffer(12);
    const bridge = new VhsBridge(p.vhs);
    const tracker = createTracker();
    const opts = { defaultSegmentDurationSec: 4, liveRemainingSegments: null };

    const first = buildObservation(p, bridge, tracker, opts)!;
    expect(first.lastSegment).toBeNull();
    expect(first.playback.bufferSec).toBe(12);
    expect(first.currentRepresentationId).toBe('pl0');

    p.vhs!.addTransfer(600_000, 1500);
    const second = buildObservation(p, bridge, tracker, opts)!;
    expect(second.lastSegment).not.toBeNull();
    expect(second.lastSegment!.sizeBytes).toBe(600_000);
    // 1500 ms of transfer becomes 1.5 s — the ms→s conversion.
    expect(second.lastSegment!.downloadSec).toBe(1.5);
  });

  it('converts BANDWIDTH straight through as bits/s', () => {
    const p = new MockPlayer();
    const obs = buildObservation(p, new VhsBridge(p.vhs), createTracker(), {
      defaultSegmentDurationSec: 4,
      liveRemainingSegments: null,
    })!;
    expect(obs.representations.map((r) => r.bitrateBps)).toEqual([
      300_000, 750_000, 1_200_000, 1_850_000, 2_850_000, 4_300_000,
    ]);
  });

  it('marks app-disabled renditions as disabled', () => {
    const p = new MockPlayer();
    p.vhs!.representations()[2]!.enabled(false);
    const obs = buildObservation(p, new VhsBridge(p.vhs), createTracker(), {
      defaultSegmentDurationSec: 4,
      liveRemainingSegments: null,
    })!;
    expect(obs.representations.find((r) => r.id === 'pl2')!.enabled).toBe(false);
  });

  it('derives remaining segments from duration and target duration on VOD', () => {
    const p = new MockPlayer().setTime(40).setDuration(192);
    const obs = buildObservation(p, new VhsBridge(p.vhs), createTracker(), {
      defaultSegmentDurationSec: 4,
      liveRemainingSegments: null,
    })!;
    expect(obs.remainingSegments).toBeCloseTo((192 - 40) / 4, 6);
    expect(obs.totalSegments).toBe(48);
  });

  it('reports null remaining segments on live', () => {
    const p = new MockPlayer({
      playlists: makePlaylists([1e6, 2e6], { live: true }),
      live: true,
    });
    const obs = buildObservation(p, new VhsBridge(p.vhs), createTracker(), {
      defaultSegmentDurationSec: 4,
      liveRemainingSegments: null,
    })!;
    expect(obs.remainingSegments).toBeNull();
    expect(obs.diagnostics?.live).toBe(true);
  });

  it('does not report rebuffering while paused', async () => {
    const p = new MockPlayer().setTime(10).setBuffer(20);
    const bridge = new VhsBridge(p.vhs);
    const tracker = createTracker();
    const opts = { defaultSegmentDurationSec: 4, liveRemainingSegments: null };
    buildObservation(p, bridge, tracker, opts);
    p.setPaused(true);
    await new Promise((r) => setTimeout(r, 120));
    const obs = buildObservation(p, bridge, tracker, opts)!;
    expect(obs.playback.rebufferSec).toBe(0);
  });

  it('reports stall time when wall clock advances but the playhead does not', async () => {
    const p = new MockPlayer().setTime(10).setBuffer(1);
    const bridge = new VhsBridge(p.vhs);
    const tracker = createTracker();
    const opts = { defaultSegmentDurationSec: 4, liveRemainingSegments: null };
    buildObservation(p, bridge, tracker, opts);
    await new Promise((r) => setTimeout(r, 150));
    const obs = buildObservation(p, bridge, tracker, opts)!;
    expect(obs.playback.rebufferSec).toBeGreaterThan(0.05);
  });
});

describe('VideoJSAbrAdapter — installing and restoring selectPlaylist', () => {
  it('overrides selectPlaylist and restores the original behaviour on destroy', async () => {
    const p = new MockPlayer().setTime(4).setBuffer(10);
    const original = p.vhs!.selectPlaylist;

    const adapter = new VideoJSAbrAdapter({ player: p, abr: engineSelecting(4) });
    await adapter.initialize();
    expect(adapter.active).toBe(true);
    expect(p.vhs!.selectPlaylist).not.toBe(original);

    await p.completeSegment(600_000, 1200);
    expect(p.checkABR()?.id).toBe('pl4');

    adapter.destroy();
    expect(adapter.active).toBe(false);

    // VHS's `selectPlaylist` setter re-binds whatever is assigned, so the
    // restored function is a bound copy rather than the identical reference.
    // What matters is that the player's own selector is back in charge.
    const before = p.vhs!.defaultSelectorCalls;
    expect(p.checkABR()?.id).toBe('pl0');
    expect(p.vhs!.defaultSelectorCalls).toBe(before + 1);
  });

  it('delegates to the captured player default until a decision exists', async () => {
    const p = new MockPlayer();
    const adapter = new VideoJSAbrAdapter({
      player: p,
      abr: new AbrEngine({
        model: { type: 'session', session: fakeSession(onehot(5)) },
        fallback: 'player-default',
      }),
    });
    // Do not initialise: no decision has been cached yet.
    (adapter as unknown as { tryAttach(): boolean }).tryAttach();

    const before = p.vhs!.defaultSelectorCalls;
    p.checkABR();
    expect(p.vhs!.defaultSelectorCalls).toBe(before + 1);
    expect(adapter.counters.delegatedToDefault).toBeGreaterThan(0);
    adapter.destroy();
  });

  it('serves the model decision synchronously once one is cached', async () => {
    const p = new MockPlayer().setTime(4).setBuffer(10);
    const adapter = new VideoJSAbrAdapter({ player: p, abr: engineSelecting(4) });
    await adapter.initialize();

    await p.completeSegment(600_000, 1200);

    const chosen = p.checkABR();
    expect(chosen?.id).toBe('pl4'); // 2850 kbps, action 4
    expect(adapter.counters.servedFromCache).toBeGreaterThan(0);
    adapter.destroy();
  });

  it('is cheap and synchronous on the selectPlaylist hot path', async () => {
    const p = new MockPlayer().setTime(4).setBuffer(10);
    const adapter = new VideoJSAbrAdapter({ player: p, abr: engineSelecting(2) });
    await adapter.initialize();
    await p.completeSegment(600_000, 1200);

    // VHS polls at 4 Hz when bufferBasedABR is on. 1000 calls must not trigger
    // 1000 inferences — the whole point of the cached-decision design.
    const decisionsBefore = adapter.counters.decisions;
    for (let i = 0; i < 1000; i++) p.checkABR();
    expect(adapter.counters.decisions).toBe(decisionsBefore);
    expect(adapter.counters.selectPlaylistCalls).toBeGreaterThanOrEqual(1000);
    adapter.destroy();
  });

  it('falls back to the player default once a decision goes stale', async () => {
    const p = new MockPlayer().setTime(4).setBuffer(10);
    const adapter = new VideoJSAbrAdapter({
      player: p,
      abr: engineSelecting(4),
      decisionTtlMs: 20,
    });
    await adapter.initialize();
    await p.completeSegment(600_000, 1200);
    expect(p.checkABR()?.id).toBe('pl4');

    await new Promise((r) => setTimeout(r, 40));
    const before = p.vhs!.defaultSelectorCalls;
    p.checkABR();
    expect(p.vhs!.defaultSelectorCalls).toBe(before + 1);
    adapter.destroy();
  });

  it('never lets an exception escape into selectPlaylist', async () => {
    const p = new MockPlayer();
    const adapter = new VideoJSAbrAdapter({ player: p, abr: engineSelecting(1) });
    await adapter.initialize();
    // Corrupt the cache so the fast path throws.
    (adapter as unknown as { cached: unknown }).cached = {
      get playlist(): never {
        throw new Error('exploded');
      },
      decision: {},
      atMs: Date.now(),
    };
    expect(() => p.checkABR()).not.toThrow();
    adapter.destroy();
  });
});

describe('VideoJSAbrAdapter — the decision loop', () => {
  it('runs one decision per segment, not per selectPlaylist call', async () => {
    const p = new MockPlayer().setTime(4).setBuffer(10);
    const adapter = new VideoJSAbrAdapter({ player: p, abr: engineSelecting(3) });
    await adapter.initialize();

    const start = adapter.counters.decisions;
    for (let i = 0; i < 5; i++) {
      p.setTime(4 + i * 4).setBuffer(10);
      await p.completeSegment(500_000, 1000);
      for (let k = 0; k < 16; k++) p.checkABR(); // 4 s at 4 Hz
    }
    expect(adapter.counters.decisions - start).toBe(5);
    adapter.destroy();
  });

  it('reflects the player state in the observation it builds', async () => {
    const p = new MockPlayer().setTime(12).setBuffer(18);
    const abr = engineSelecting(2);
    const adapter = new VideoJSAbrAdapter({ player: p, abr });
    await adapter.initialize();

    const seen: unknown[] = [];
    abr.on('observation', (e) => seen.push(e.observation));
    await p.completeSegment(750_000, 2000);

    const obs = seen.at(-1) as {
      playback: { bufferSec: number; currentTimeSec: number };
      lastSegment: { sizeBytes: number; downloadSec: number };
      representations: unknown[];
    };
    expect(obs.playback.bufferSec).toBe(18);
    expect(obs.playback.currentTimeSec).toBe(12);
    expect(obs.lastSegment.sizeBytes).toBe(750_000);
    expect(obs.lastSegment.downloadSec).toBe(2);
    expect(obs.representations).toHaveLength(6);
    adapter.destroy();
  });

  it('drops overlapping ticks rather than queueing them', async () => {
    const p = new MockPlayer().setTime(4).setBuffer(10);
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const abr = new AbrEngine({
      model: {
        type: 'session',
        session: {
          async run() {
            await gate;
            return { actionProbs: Float32Array.from(onehot(1)), stateValue: 0 };
          },
        },
      },
      inference: { warmup: false, timeoutMs: 5000 },
    });
    const adapter = new VideoJSAbrAdapter({ player: p, abr });
    await adapter.initialize();

    p.vhs!.addTransfer(500_000, 1000);
    p.trigger('bandwidthupdate');
    await tick();
    p.trigger('bandwidthupdate');
    p.trigger('bandwidthupdate');
    await tick();
    expect(adapter.counters.skippedInFlight).toBeGreaterThanOrEqual(2);

    release();
    await tick();
    adapter.destroy();
  });

  it('emits apply events describing what happened', async () => {
    const p = new MockPlayer().setTime(4).setBuffer(10);
    const adapter = new VideoJSAbrAdapter({ player: p, abr: engineSelecting(5) });
    const applies: Array<{ applied: boolean; decision: { representationId: string } }> = [];
    adapter.on('apply', (e) => applies.push(e));
    await adapter.initialize();
    await p.completeSegment(500_000, 1000);
    expect(applies.at(-1)!.applied).toBe(true);
    expect(applies.at(-1)!.decision.representationId).toBe('pl5');
    adapter.destroy();
  });

  it('does not apply a decision that declines', async () => {
    const p = new MockPlayer().setTime(4).setBuffer(10);
    const adapter = new VideoJSAbrAdapter({
      player: p,
      abr: new AbrEngine({
        model: { type: 'session', session: throwingSession() },
        inference: { warmup: false },
        fallback: 'player-default',
      }),
    });
    const applies: Array<{ applied: boolean }> = [];
    adapter.on('apply', (e) => applies.push(e));
    await adapter.initialize();
    await p.completeSegment(500_000, 1000);
    expect(applies.at(-1)!.applied).toBe(false);

    const before = p.vhs!.defaultSelectorCalls;
    p.checkABR();
    expect(p.vhs!.defaultSelectorCalls).toBe(before + 1);
    adapter.destroy();
  });

  it('calls fastQualityChange_ only when applyImmediately is set', async () => {
    const lazy = new MockPlayer().setTime(4).setBuffer(10);
    const a1 = new VideoJSAbrAdapter({ player: lazy, abr: engineSelecting(4) });
    await a1.initialize();
    await lazy.completeSegment(500_000, 1000);
    expect(lazy.vhs!.fastQualityChanges).toHaveLength(0);
    a1.destroy();

    const eager = new MockPlayer().setTime(4).setBuffer(10);
    const a2 = new VideoJSAbrAdapter({
      player: eager,
      abr: engineSelecting(4),
      applyImmediately: true,
    });
    await a2.initialize();
    await eager.completeSegment(500_000, 1000);
    expect(eager.vhs!.fastQualityChanges.map((p) => p.id)).toContain('pl4');
    a2.destroy();
  });
});

describe('VideoJSAbrAdapter — lifecycle and resilience', () => {
  it('waits for VHS to appear and attaches when it does', async () => {
    const p = new MockPlayer({ attachVhs: false });
    const adapter = new VideoJSAbrAdapter({ player: p, abr: engineSelecting(1) });
    const started = adapter.initialize();
    expect(adapter.active).toBe(false);

    p.attachVhs();
    p.trigger('loadedmetadata');
    await started;
    expect(adapter.active).toBe(true);
    adapter.destroy();
  });

  it('re-attaches and clears history on a new source', async () => {
    const p = new MockPlayer().setTime(4).setBuffer(10);
    const abr = engineSelecting(3);
    const adapter = new VideoJSAbrAdapter({ player: p, abr });
    await adapter.initialize();
    await p.completeSegment(500_000, 1000);
    expect(Array.from(abr.snapshotState()).some((v) => v !== 0)).toBe(true);

    p.detachVhs();
    p.attachVhs();
    p.trigger('loadstart');
    expect(Array.from(abr.snapshotState()).every((v) => v === 0)).toBe(true);
    expect(adapter.active).toBe(true);
    adapter.destroy();
  });

  it('resets the engine history on seek', async () => {
    const p = new MockPlayer().setTime(4).setBuffer(10);
    const abr = engineSelecting(3);
    const adapter = new VideoJSAbrAdapter({ player: p, abr });
    await adapter.initialize();
    await p.completeSegment(500_000, 1000);
    p.trigger('seeking');
    expect(Array.from(abr.snapshotState()).every((v) => v === 0)).toBe(true);
    adapter.destroy();
  });

  it('can be told not to reset on seek', async () => {
    const p = new MockPlayer().setTime(4).setBuffer(10);
    const abr = engineSelecting(3);
    const adapter = new VideoJSAbrAdapter({ player: p, abr, resetOnSeek: false });
    await adapter.initialize();
    await p.completeSegment(500_000, 1000);
    p.trigger('seeking');
    expect(Array.from(abr.snapshotState()).some((v) => v !== 0)).toBe(true);
    adapter.destroy();
  });

  it('tears down on player dispose', async () => {
    const p = new MockPlayer();
    const adapter = new VideoJSAbrAdapter({ player: p, abr: engineSelecting(0) });
    await adapter.initialize();
    p.dispose();
    expect(adapter.active).toBe(false);
  });

  it('destroy() is idempotent and removes every listener', async () => {
    const p = new MockPlayer();
    const adapter = new VideoJSAbrAdapter({ player: p, abr: engineSelecting(0) });
    await adapter.initialize();
    adapter.destroy();
    adapter.destroy();
    expect(p.listenerCount('bandwidthupdate')).toBe(0);
    expect(p.listenerCount('loadstart')).toBe(0);
  });

  it('requires a player and an engine', () => {
    expect(() => new VideoJSAbrAdapter({ player: null, abr: engineSelecting(0) })).toThrow(
      /player.*required/,
    );
    expect(
      () => new VideoJSAbrAdapter({ player: new MockPlayer(), abr: null as never }),
    ).toThrow(/abr.*required/);
  });

  it('never breaks the player when the engine misbehaves', async () => {
    const p = new MockPlayer().setTime(4).setBuffer(10);
    const abr = engineSelecting(2);
    const adapter = new VideoJSAbrAdapter({ player: p, abr });
    await adapter.initialize();

    // A rejecting decide() is contractually impossible, but must be survivable.
    vi.spyOn(abr, 'decide').mockRejectedValue(new Error('contract violation'));
    await expect(p.completeSegment(500_000, 1000)).resolves.toBeUndefined();
    expect(() => p.checkABR()).not.toThrow();
    adapter.destroy();
  });

  it('handles a hostile VHS handler without attaching', async () => {
    const p = new MockPlayer({ attachVhs: false });
    (p as unknown as { tech(): unknown }).tech = () => ({ vhs: { stats: {} } });
    const adapter = new VideoJSAbrAdapter({
      player: p,
      abr: engineSelecting(0),
      attachTimeoutMs: 200,
    });
    await adapter.initialize();
    expect(adapter.active).toBe(false);
    expect(adapter.getObservation()).toBeNull();
    adapter.destroy();
  });
});
