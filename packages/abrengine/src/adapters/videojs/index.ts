/**
 * VideoJSAbrAdapter — plugs the engine into Video.js / VHS.
 * =========================================================
 *
 * How the override works (verified against video.js 8.24.0 / VHS 3.17.5):
 *
 * VHS exposes an assignable `selectPlaylist` on the handler:
 *
 * ```js
 * Object.defineProperties(this, {
 *   selectPlaylist: {
 *     get() { return this.playlistController_.selectPlaylist; },
 *     set(fn) { this.playlistController_.selectPlaylist = fn.bind(this); }
 *   },
 *   ...
 * });
 * ```
 *
 * `PlaylistController.checkABR_()` calls it with no arguments and expects a
 * playlist object back (or a falsy value to keep the current one). We capture
 * the previous implementation — normally `Vhs.STANDARD_PLAYLIST_SELECTOR`, i.e.
 * `lastBandwidthSelector` — before overriding, which is what makes
 * `fallback: 'player-default'` genuinely the player's default rather than a
 * re-implementation of it.
 *
 * The timing problem and its solution
 * -----------------------------------
 * `selectPlaylist()` is **synchronous** but ONNX inference is **asynchronous**,
 * and VHS calls `selectPlaylist()` on a 250 ms timer (when `bufferBasedABR` is
 * on) plus on every `bandwidthupdate` and fullscreen change — far more often
 * than segments actually complete.
 *
 * So the adapter runs two decoupled loops:
 *
 * ```
 *  segment completes                          VHS asks for a playlist
 *  (tech 'bandwidthupdate')                   (checkABR_ -> selectPlaylist)
 *          |                                             |
 *   buildObservation()                          read cached decision
 *          |                                             |
 *   engine.decide()  ── async ──> cache  ────────────────┘
 *                                                (synchronous, ~0 cost)
 * ```
 *
 * The cache has a TTL (`decisionTtlMs`). A decision older than that is treated
 * as stale and the captured default selector runs instead, so a wedged
 * inference can never pin the player to a rendition indefinitely.
 *
 * This also means the trained controller is evaluated at the cadence it was
 * trained at: once per downloaded chunk.
 */

import type { AbrDecision } from '../../types/decision.js';
import type { AbrObservation } from '../../types/observation.js';
import type { PlayerAbrAdapter } from '../../types/adapter.js';
import type { AbrEngine } from '../../core/engine.js';
import type { AbrConfig } from '../../types/config.js';
import { SEGMENT_DURATION_SEC } from '../../model/manifest.js';
import { now } from '../../utils/clock.js';
import { Emitter } from '../../utils/emitter.js';
import { VhsBridge, type VhsPlaylist } from './vhs-bridge.js';
import { buildObservation, createTracker, type SegmentTracker } from './observation.js';

export interface VideoJSAbrAdapterOptions {
  /** The Video.js player instance. */
  player: any;
  /** An `AbrEngine`. The adapter calls `initialize()` on it if not already ready. */
  abr: AbrEngine;

  /**
   * Fallback segment duration when the manifest does not declare a target
   * duration, seconds. @default 4 (the research env's `VIDEO_CHUNK_LEN`)
   */
  defaultSegmentDurationSec?: number;

  /**
   * A cached decision older than this is ignored and the player's own selector
   * runs instead, milliseconds. @default 30000
   */
  decisionTtlMs?: number;

  /**
   * Also poll on a timer, milliseconds, in addition to segment-completion
   * events. `0` disables. Off by default because the controller is meant to
   * decide once per chunk; raise it only if your source emits no
   * `bandwidthupdate`. @default 0
   */
  pollIntervalMs?: number;

  /**
   * When a decision lands between VHS ABR ticks, call
   * `PlaylistController.fastQualityChange_()` to act on it immediately.
   *
   * `false` (the default) is the conservative choice: the decision is picked up
   * by the next `selectPlaylist()` call, at most a few hundred ms later, with
   * no buffer flush. `fastQualityChange_` discards already-buffered content to
   * apply the switch instantly, which is right for a user-driven quality change
   * and usually wrong for an automatic one.
   * @default false
   */
  applyImmediately?: boolean;

  /**
   * Reset the engine's observation history on seek. The model's 8-step
   * throughput window is meaningless across a discontinuity. @default true
   */
  resetOnSeek?: boolean;

  /**
   * How long `initialize()` waits for a VHS handler to appear before resolving
   * anyway, milliseconds. The adapter keeps trying on later player events, so a
   * timeout here is not a failure — it just stops `initialize()` blocking.
   * @default 15000
   */
  attachTimeoutMs?: number;

  /** Log adapter activity to the console. @default false */
  debug?: boolean;
}

export interface AdapterAttachEvent {
  type: 'attach';
  /** What VHS surface the adapter found. */
  capabilities: Record<string, unknown>;
}

export interface AdapterApplyEvent {
  type: 'apply';
  decision: AbrDecision;
  /** `true` if VHS was handed the rendition; `false` if we deferred to the default. */
  applied: boolean;
  /** Rendition VHS was on when the decision was applied. */
  fromRepresentationId: string | null;
}

export interface AdapterDetachEvent {
  type: 'detach';
  reason: 'destroy' | 'dispose';
}

export interface AdapterEventMap {
  attach: AdapterAttachEvent;
  apply: AdapterApplyEvent;
  detach: AdapterDetachEvent;
}

interface CachedDecision {
  decision: AbrDecision;
  playlist: VhsPlaylist | undefined;
  atMs: number;
}

export class VideoJSAbrAdapter implements PlayerAbrAdapter {
  readonly name = 'videojs';

  private readonly player: any;
  private readonly engine: AbrEngine;
  private readonly opts: Required<Omit<VideoJSAbrAdapterOptions, 'player' | 'abr'>>;
  private readonly emitter = new Emitter<AdapterEventMap>();

  private bridge: VhsBridge | null = null;
  private tracker: SegmentTracker = createTracker();

  /** The `selectPlaylist` VHS had before we touched it. */
  private originalSelectPlaylist: ((this: unknown) => VhsPlaylist | undefined) | null = null;
  private installedOn: any = null;

  private cached: CachedDecision | null = null;
  private inFlight = false;
  private activeFlag = false;
  private destroyed = false;
  private pollTimer: ReturnType<typeof setInterval> | null = null;
  private readonly listeners: Array<() => void> = [];
  /** Listeners bound to the tech / segment loader; dropped and rebound on re-attach. */
  private readonly techListeners: Array<() => void> = [];

  /** Counters, exposed for tests and telemetry. */
  readonly counters = {
    /** `selectPlaylist()` calls VHS made. */
    selectPlaylistCalls: 0,
    /** Of those, ones served from the model/fallback cache. */
    servedFromCache: 0,
    /** Of those, ones delegated to the player's original selector. */
    delegatedToDefault: 0,
    /** Observations built. */
    observations: 0,
    /** Engine decisions received. */
    decisions: 0,
    /** Decisions dropped because one was already in flight. */
    skippedInFlight: 0,
  };

  constructor(options: VideoJSAbrAdapterOptions) {
    if (!options?.player) throw new Error('VideoJSAbrAdapter: `player` is required');
    if (!options?.abr) throw new Error('VideoJSAbrAdapter: `abr` (an AbrEngine) is required');

    this.player = options.player;
    this.engine = options.abr;
    this.opts = {
      defaultSegmentDurationSec: options.defaultSegmentDurationSec ?? SEGMENT_DURATION_SEC,
      decisionTtlMs: options.decisionTtlMs ?? 30_000,
      pollIntervalMs: options.pollIntervalMs ?? 0,
      applyImmediately: options.applyImmediately ?? false,
      resetOnSeek: options.resetOnSeek ?? true,
      attachTimeoutMs: options.attachTimeoutMs ?? 15_000,
      debug: options.debug ?? false,
    };
  }

  get active(): boolean {
    return this.activeFlag;
  }

  /** The underlying engine. */
  get abr(): AbrEngine {
    return this.engine;
  }

  on<K extends keyof AdapterEventMap>(
    event: K,
    listener: (payload: AdapterEventMap[K]) => void,
  ): () => void {
    return this.emitter.on(event, listener);
  }

  // ── Lifecycle ────────────────────────────────────────────────────────────

  /**
   * Start the engine loading and attach to the player.
   *
   * Resolves as soon as the **engine** is ready. It deliberately does *not*
   * wait for VHS: a VhsHandler only exists once the player has a source, and
   * plenty of perfectly normal code sets the source after wiring up plugins:
   *
   * ```js
   * const adapter = new VideoJSAbrAdapter({ player, abr });
   * await adapter.initialize();
   * player.src({ src: '…m3u8', type: 'application/x-mpegURL' });   // ← after
   * ```
   *
   * Blocking here would deadlock that ordering. Instead the adapter attaches
   * opportunistically — immediately if VHS already exists, otherwise on the
   * next `loadstart` / `loadedmetadata` / `canplay`, backed by a bounded
   * background poll (`attachTimeoutMs`). `active` tells you whether it has
   * attached, and the `attach` event fires when it does.
   */
  async initialize(): Promise<void> {
    if (this.destroyed) throw new Error('VideoJSAbrAdapter: already destroyed');

    this.hookPlayerEvents();
    if (!this.tryAttach()) {
      // Background best-effort; not awaited.
      void this.waitForVhs(this.opts.attachTimeoutMs);
    }

    await this.engine.initialize();
  }

  /**
   * Resolve once the adapter has attached to a VhsHandler, or the timeout
   * elapses. Only needed by tests and by code that must observe the attached
   * state; normal integrations do not call this.
   */
  waitUntilAttached(timeoutMs = this.opts.attachTimeoutMs): Promise<boolean> {
    if (this.activeFlag) return Promise.resolve(true);
    return this.waitForVhs(timeoutMs).then(() => this.activeFlag);
  }

  /** Restore the player's ABR and drop every listener. Idempotent. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.detachSelector();
    for (const off of this.techListeners.splice(0)) {
      try {
        off();
      } catch {
        /* teardown must not throw */
      }
    }
    for (const off of this.listeners.splice(0)) {
      try {
        off();
      } catch {
        /* teardown must not throw */
      }
    }
    if (this.pollTimer !== null) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    this.activeFlag = false;
    this.cached = null;
    this.emitter.emit('detach', { type: 'detach', reason: 'destroy' });
    this.emitter.removeAll();
  }

  // ── PlayerAbrAdapter ─────────────────────────────────────────────────────

  getObservation(): AbrObservation | null {
    if (!this.bridge) return null;
    try {
      const obs = buildObservation(this.player, this.bridge, this.tracker, {
        defaultSegmentDurationSec: this.opts.defaultSegmentDurationSec,
        liveRemainingSegments: null,
      });
      if (obs) this.counters.observations++;
      return obs;
    } catch (err) {
      this.log('buildObservation threw', err);
      return null;
    }
  }

  /**
   * Cache a decision so the next synchronous `selectPlaylist()` can serve it.
   *
   * Returns `false` when the decision declines (`source: 'player-default'`) or
   * names a rendition VHS does not know — in both cases the player's own
   * selector stays in charge.
   */
  applyDecision(decision: AbrDecision): boolean {
    if (!this.bridge || this.destroyed) return false;

    if (decision.source === 'player-default' || !decision.representationId) {
      this.cached = { decision, playlist: undefined, atMs: now() };
      this.emitter.emit('apply', {
        type: 'apply',
        decision,
        applied: false,
        fromRepresentationId: this.bridge.currentPlaylistId() ?? null,
      });
      return false;
    }

    const playlist = this.bridge.playlistById(decision.representationId);
    if (!playlist) {
      this.log(`decision named unknown rendition '${decision.representationId}'`);
      this.cached = { decision, playlist: undefined, atMs: now() };
      this.emitter.emit('apply', {
        type: 'apply',
        decision,
        applied: false,
        fromRepresentationId: this.bridge.currentPlaylistId() ?? null,
      });
      return false;
    }

    const from = this.bridge.currentPlaylistId() ?? null;
    this.cached = { decision, playlist, atMs: now() };

    if (this.opts.applyImmediately && from !== decision.representationId) {
      this.bridge.fastQualityChange(playlist);
    }

    this.emitter.emit('apply', {
      type: 'apply',
      decision,
      applied: true,
      fromRepresentationId: from,
    });
    return true;
  }

  // ── Attach / detach ──────────────────────────────────────────────────────

  private vhsHandle(): any {
    const tech = this.techHandle();
    return tech?.vhs ?? tech?.hls ?? null;
  }

  private tryAttach(): boolean {
    if (this.destroyed || this.activeFlag) return this.activeFlag;

    const vhs = this.vhsHandle();
    if (!vhs) return false;

    const bridge = new VhsBridge(vhs);
    if (!bridge.usable()) return false;

    this.bridge = bridge;
    this.installSelector(vhs);
    this.hookSegmentTick(vhs);
    this.tracker = createTracker();
    this.engine.reset();
    this.activeFlag = true;

    if (this.opts.pollIntervalMs > 0 && this.pollTimer === null) {
      this.pollTimer = setInterval(() => void this.tick('poll'), this.opts.pollIntervalMs);
      (this.pollTimer as unknown as { unref?: () => void }).unref?.();
    }

    this.log('attached', bridge.describe());
    this.emitter.emit('attach', { type: 'attach', capabilities: bridge.describe() });

    // Prime the counter tracker so the first real segment produces a usable diff.
    void this.tick('attach');
    return true;
  }

  /**
   * Subscribe to the once-per-segment tick.
   *
   * VHS raises `bandwidthupdate` on the **tech**, not the player:
   *
   * ```js
   * this.mainSegmentLoader_.on('bandwidthupdate', () => {
   *   this.checkABR_('bandwidthupdate');
   *   this.tech_.trigger('bandwidthupdate');   // <- tech, not player
   * });
   * ```
   *
   * Video.js does not forward that event to the player, so listening on the
   * player alone yields no ticks at all. We subscribe to the tech, and — when
   * reachable — directly to the main segment loader as well, which is the
   *original source and fires even if tech forwarding ever changes. Duplicate ticks
   * are harmless: `tick()` drops one that arrives while another is in flight.
   */
  private hookSegmentTick(vhs: any): void {
    for (const off of this.techListeners.splice(0)) {
      try {
        off();
      } catch {
        /* the previous tech may already be disposed */
      }
    }

    const bind = (target: any, event: string): void => {
      if (!target?.on) return;
      const handler = (): void => void this.tick(event);
      target.on(event, handler);
      this.techListeners.push(() => target.off?.(event, handler));
    };

    const tech = this.techHandle();
    bind(tech, 'bandwidthupdate');

    const loader =
      vhs?.playlistController_?.mainSegmentLoader_ ??
      vhs?.masterPlaylistController_?.mainSegmentLoader_;
    bind(loader, 'bandwidthupdate');
  }

  private techHandle(): any {
    try {
      return this.player?.tech?.(true) ?? null;
    } catch {
      return null;
    }
  }

  private installSelector(vhs: any): void {
    // Capture the incumbent. This is the genuine player default and is what
    // `fallback: 'player-default'` delegates to.
    const original = vhs.selectPlaylist;
    this.originalSelectPlaylist =
      typeof original === 'function' ? (original as () => VhsPlaylist | undefined) : null;
    this.installedOn = vhs;

    // VHS's setter binds whatever we assign to the VhsHandler, so `this` inside
    // is the handler — we deliberately close over the adapter instead.
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;
    vhs.selectPlaylist = function selectPlaylistWithAbrEngine(this: unknown) {
      return self.selectPlaylist(this);
    };
  }

  private detachSelector(): void {
    if (!this.installedOn) return;
    try {
      if (this.originalSelectPlaylist) {
        this.installedOn.selectPlaylist = this.originalSelectPlaylist;
      }
    } catch {
      /* the handler may already be disposed */
    }
    this.installedOn = null;
    this.originalSelectPlaylist = null;
  }

  /**
   * The synchronous hot path. Must be cheap and must never throw: VHS calls it
   * from a `setInterval` and from event handlers, and an exception here would
   * surface as a playback error.
   */
  private selectPlaylist(vhsThis: unknown): VhsPlaylist | undefined {
    this.counters.selectPlaylistCalls++;
    try {
      const c = this.cached;
      const fresh =
        c !== null && c.playlist !== undefined && now() - c.atMs <= this.opts.decisionTtlMs;

      if (fresh) {
        this.counters.servedFromCache++;
        return c!.playlist;
      }

      this.counters.delegatedToDefault++;
      return this.callOriginalSelector(vhsThis);
    } catch (err) {
      this.log('selectPlaylist threw, delegating to player default', err);
      try {
        return this.callOriginalSelector(vhsThis);
      } catch {
        return undefined;
      }
    }
  }

  private callOriginalSelector(vhsThis: unknown): VhsPlaylist | undefined {
    if (!this.originalSelectPlaylist) return undefined;
    // The captured function was already bound by VHS's setter, but binding again
    // is harmless and keeps a hand-assigned unbound function working too.
    return this.originalSelectPlaylist.call(vhsThis ?? this.installedOn);
  }

  // ── Decision loop ────────────────────────────────────────────────────────

  /** One observe-and-decide cycle. Async, off the `selectPlaylist` critical path. */
  private async tick(reason: string): Promise<void> {
    if (this.destroyed || !this.activeFlag) return;
    if (this.inFlight) {
      this.counters.skippedInFlight++;
      return;
    }

    const obs = this.getObservation();
    if (!obs) return;

    this.inFlight = true;
    try {
      const decision = await this.engine.decide(obs);
      this.counters.decisions++;
      if (this.destroyed) return;
      this.applyDecision(decision);
      this.log(`decision (${reason})`, {
        source: decision.source,
        rep: decision.representationId,
        action: decision.actionIndex,
      });
    } catch (err) {
      // `engine.decide` is documented never to reject, but an adapter must not
      // depend on that: an unhandled rejection here would be a playback bug.
      this.log('engine.decide rejected', err);
    } finally {
      this.inFlight = false;
    }
  }

  // ── Player events ────────────────────────────────────────────────────────

  private hookPlayerEvents(): void {
    const on = (target: any, event: string, handler: (...a: unknown[]) => void): void => {
      if (!target?.on) return;
      target.on(event, handler);
      this.listeners.push(() => target.off?.(event, handler));
    };

    // A new source means a new VHS handler: re-attach and clear history.
    on(this.player, 'loadstart', () => {
      this.activeFlag = false;
      this.bridge = null;
      this.detachSelector();
      this.cached = null;
      this.tracker = createTracker();
      this.engine.reset();
      this.tryAttach();
    });

    on(this.player, 'loadedmetadata', () => this.tryAttach());
    on(this.player, 'canplay', () => this.tryAttach());

    // Some setups re-emit `bandwidthupdate` on the player; harmless to also
    // listen here. The load-bearing subscription is `hookSegmentTick`.
    on(this.player, 'bandwidthupdate', () => void this.tick('player-bandwidthupdate'));

    if (this.opts.resetOnSeek) {
      on(this.player, 'seeking', () => {
        this.engine.reset();
        this.cached = null;
        this.tracker = createTracker();
      });
    }

    on(this.player, 'dispose', () => this.destroy());
  }

  /** Poll for a VHS handler until one appears or the player goes away. */
  private waitForVhs(timeoutMs: number): Promise<void> {
    return new Promise<void>((resolve) => {
      const started = now();
      const iv = setInterval(() => {
        if (this.destroyed || this.tryAttach() || now() - started > timeoutMs) {
          clearInterval(iv);
          resolve();
        }
      }, 100);
      (iv as unknown as { unref?: () => void }).unref?.();
    });
  }

  private log(message: string, extra?: unknown): void {
    if (!this.opts.debug) return;
    // eslint-disable-next-line no-console
    console.debug(`[abrengine/videojs] ${message}`, extra ?? '');
  }
}

/**
 * Convenience: build an engine, attach an adapter, and wait for both.
 *
 * ```js
 * const { abr, adapter } = await attachAbrEngine(player, { model: 'ac3' });
 * ```
 */
export async function attachAbrEngine(
  player: any,
  engineOrConfig: AbrEngine | AbrConfig,
  adapterOptions?: Omit<VideoJSAbrAdapterOptions, 'player' | 'abr'>,
): Promise<{ abr: AbrEngine; adapter: VideoJSAbrAdapter }> {
  const { AbrEngine: Engine } = await import('../../core/engine.js');
  const abr =
    engineOrConfig instanceof Engine
      ? (engineOrConfig as AbrEngine)
      : new Engine(engineOrConfig as AbrConfig);

  const adapter = new VideoJSAbrAdapter({ player, abr, ...(adapterOptions ?? {}) });
  await adapter.initialize();
  return { abr, adapter };
}
