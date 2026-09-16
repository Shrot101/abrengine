/**
 * A guarded view onto videojs-http-streaming's internals.
 * =======================================================
 *
 * Verified against **video.js 8.24.0 / @videojs/http-streaming 3.17.5** by
 * reading `dist/videojs-http-streaming.cjs.js`. Every access below is either a
 * documented public API or a private field that is probed defensively and has a
 * public fallback. Nothing here assumes a property exists.
 *
 * What VHS actually gives us, and where it comes from in the source:
 *
 * | need                       | surface                                                | kind    |
 * |----------------------------|--------------------------------------------------------|---------|
 * | override bitrate selection | `vhs.selectPlaylist` (defineProperty setter, binds `this`) | public  |
 * | the ladder                 | `vhs.representations()` (renditionSelectionMixin)      | public  |
 * | current rendition          | `vhs.playlists.media()`                                | public  |
 * | ladder objects to return   | `vhs.playlists.main.playlists`                         | public  |
 * | throughput estimate        | `vhs.bandwidth` (bits/s)                               | public  |
 * | per-segment stats          | `vhs.stats.mediaBytesTransferred` / `mediaTransferDuration` | public  |
 * | per-segment stats (better) | `pc.mainSegmentLoader_.{bandwidth,roundTrip,mediaBytesTransferred,mediaTransferDuration}` | private |
 * | segment-completed tick     | `mainSegmentLoader_.on('bandwidthupdate')` → `tech.trigger('bandwidthupdate')` | public event |
 * | force a switch now         | `pc.fastQualityChange_(playlist)`                      | private |
 *
 * Two facts drive the whole adapter design:
 *
 * 1. **`selectPlaylist()` is synchronous.** VHS calls it and immediately uses
 *    the returned playlist object. ONNX inference is asynchronous. So the
 *    adapter computes decisions off the critical path and `selectPlaylist`
 *    returns a *cached* answer.
 * 2. **`selectPlaylist()` is called far more often than segments complete.**
 *    `PlaylistController.startABRTimer_` installs
 *    `setInterval(() => this.checkABR_(), 250)` when `bufferBasedABR` is on, and
 *    `checkABR_` also fires on every `bandwidthupdate` and on fullscreen change.
 *    Running the trained policy at 4 Hz would be nothing like the once-per-chunk
 *    cadence it was trained at, so the adapter drives inference from
 *    segment-completion events only.
 *
 * `vhs.stats.mediaBytesTransferred` sums *all* segment loaders (main + audio +
 * subtitles) via `sumLoaderStat`, so for demuxed audio it over-counts video
 * bytes. We therefore prefer `mainSegmentLoader_` when it is reachable and fall
 * back to `vhs.stats` otherwise, reporting which one is in use via
 * `statsSource`.
 */

export interface VhsPlaylistAttributes {
  BANDWIDTH?: number;
  RESOLUTION?: { width?: number; height?: number };
  CODECS?: string;
  'FRAME-RATE'?: number;
}

export interface VhsPlaylist {
  id?: string;
  uri?: string;
  attributes?: VhsPlaylistAttributes;
  disabled?: boolean;
  endList?: boolean;
  targetDuration?: number;
  segments?: Array<{ duration?: number }>;
}

export interface VhsRepresentation {
  id: string;
  bandwidth?: number;
  width?: number;
  height?: number;
  frameRate?: number;
  codecs?: string;
  playlist: VhsPlaylist;
  enabled: (value?: boolean) => boolean;
}

export type StatsSource = 'main-segment-loader' | 'vhs-stats' | 'none';

/** Cumulative transfer counters, read at one instant. */
export interface TransferCounters {
  /** Cumulative bytes transferred. */
  bytes: number;
  /** Cumulative transfer duration, milliseconds. */
  durationMs: number;
  /** Number of media requests issued so far. */
  requests: number;
  source: StatsSource;
}

/**
 * Wraps a VhsHandler with null-safe accessors.
 *
 * Every method returns `undefined`/`null`/a neutral value instead of throwing,
 * because these are called from player event handlers where an exception would
 * surface as a playback error.
 */
export class VhsBridge {
  constructor(private readonly vhs: any) {}

  /** The raw VhsHandler, for escape hatches. */
  get handle(): any {
    return this.vhs;
  }

  /** `PlaylistController` (VHS ≥3) or `MasterPlaylistController` (VHS 2). */
  private get pc(): any {
    return this.vhs?.playlistController_ ?? this.vhs?.masterPlaylistController_;
  }

  private get mainLoader(): any {
    return this.pc?.mainSegmentLoader_;
  }

  /** `true` when the handler exposes enough surface to drive ABR. */
  usable(): boolean {
    return (
      !!this.vhs &&
      typeof this.vhs.representations === 'function' &&
      !!this.vhs.playlists &&
      typeof this.vhs.playlists.media === 'function'
    );
  }

  /** The ladder as VHS reports it. Empty array when unavailable. */
  representations(): VhsRepresentation[] {
    try {
      const reps = this.vhs?.representations?.();
      return Array.isArray(reps) ? reps : [];
    } catch {
      return [];
    }
  }

  /**
   * Every playlist VHS considers switchable, i.e. the objects `selectPlaylist`
   * is allowed to return. `representations()` filters out incompatible ones,
   * which is exactly the set we want.
   */
  playlistById(id: string): VhsPlaylist | undefined {
    for (const r of this.representations()) {
      if (r.id === id) return r.playlist;
    }
    // Fall back to the main manifest list in case the mixin is not installed yet.
    const list: VhsPlaylist[] | undefined = this.vhs?.playlists?.main?.playlists;
    return list?.find((p) => (p.id ?? p.uri) === id);
  }

  /** The playlist currently being loaded. */
  currentPlaylist(): VhsPlaylist | undefined {
    try {
      return this.vhs?.playlists?.media?.();
    } catch {
      return undefined;
    }
  }

  currentPlaylistId(): string | undefined {
    const m = this.currentPlaylist();
    return m ? (m.id ?? m.uri) : undefined;
  }

  /** Player-wide throughput estimate, bits/s. `null` when unknown. */
  bandwidthBps(): number | null {
    const b = this.vhs?.bandwidth;
    return typeof b === 'number' && Number.isFinite(b) && b > 0 ? b : null;
  }

  /**
   * Cumulative transfer counters.
   *
   * Prefers the main segment loader (video only). Falls back to `vhs.stats`,
   * which aggregates audio and subtitle loaders too.
   */
  counters(): TransferCounters {
    const loader = this.mainLoader;
    if (
      loader &&
      typeof loader.mediaBytesTransferred === 'number' &&
      typeof loader.mediaTransferDuration === 'number'
    ) {
      return {
        bytes: loader.mediaBytesTransferred,
        durationMs: loader.mediaTransferDuration,
        requests: typeof loader.mediaRequests === 'number' ? loader.mediaRequests : 0,
        source: 'main-segment-loader',
      };
    }
    const stats = this.vhs?.stats;
    if (stats && typeof stats.mediaBytesTransferred === 'number') {
      return {
        bytes: stats.mediaBytesTransferred,
        durationMs: stats.mediaTransferDuration ?? 0,
        requests: stats.mediaRequests ?? 0,
        source: 'vhs-stats',
      };
    }
    return { bytes: 0, durationMs: 0, requests: 0, source: 'none' };
  }

  /** Which counter source `counters()` is using. */
  statsSource(): StatsSource {
    return this.counters().source;
  }

  /** Target segment duration of the current rendition, seconds. */
  targetDurationSec(): number | null {
    const m = this.currentPlaylist();
    if (!m) return null;
    if (typeof m.targetDuration === 'number' && m.targetDuration > 0) return m.targetDuration;
    const first = m.segments?.[0]?.duration;
    return typeof first === 'number' && first > 0 ? first : null;
  }

  /** Number of segments in the current rendition, or `null` for live. */
  segmentCount(): number | null {
    const m = this.currentPlaylist();
    if (!m || m.endList !== true) return null;
    return Array.isArray(m.segments) ? m.segments.length : null;
  }

  /** `true` when the current rendition is a live/event playlist. */
  isLive(): boolean {
    const m = this.currentPlaylist();
    return !!m && m.endList !== true;
  }

  /**
   * Force VHS to act on a rendition change immediately rather than waiting for
   * its next ABR tick. Used when a decision arrives between ticks.
   *
   * Returns `false` when the private hook is unavailable — the decision is then
   * simply picked up by the next `selectPlaylist()` call, which costs at most
   * one segment of latency.
   */
  fastQualityChange(playlist: VhsPlaylist): boolean {
    const pc = this.pc;
    if (!pc || typeof pc.fastQualityChange_ !== 'function') return false;
    try {
      pc.fastQualityChange_(playlist);
      return true;
    } catch {
      return false;
    }
  }

  /** Ask VHS whether it would accept this switch. Advisory only. */
  wouldSwitchTo(playlist: VhsPlaylist): boolean | null {
    const pc = this.pc;
    if (!pc || typeof pc.shouldSwitchToMedia_ !== 'function') return null;
    try {
      return !!pc.shouldSwitchToMedia_(playlist);
    } catch {
      return null;
    }
  }

  /** Version strings, for telemetry. */
  describe(): Record<string, unknown> {
    return {
      hasPlaylistController: !!this.vhs?.playlistController_,
      hasMasterPlaylistController: !!this.vhs?.masterPlaylistController_,
      hasRepresentations: typeof this.vhs?.representations === 'function',
      statsSource: this.statsSource(),
      bufferBasedABR: !!this.pc?.bufferBasedABR,
    };
  }
}
