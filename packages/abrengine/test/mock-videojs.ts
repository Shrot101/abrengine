/**
 * A mock Video.js player + VHS handler.
 *
 * Shapes are copied from the real thing (video.js 8.24.0 / VHS 3.17.5):
 *
 *  - `player.tech(true).vhs` is the VhsHandler
 *  - `vhs.selectPlaylist` is an accessor whose setter binds to the handler,
 *    exactly as VHS's `Object.defineProperties(this, { selectPlaylist: {...} })`
 *    does — so the adapter's install/restore path is exercised for real
 *  - `vhs.representations()` returns `Representation`-shaped objects
 *  - `vhs.playlists.media()` / `.main.playlists`
 *  - `vhs.stats.mediaBytesTransferred` / `mediaTransferDuration` (cumulative)
 *  - `vhs.playlistController_.mainSegmentLoader_` with the same counters
 *  - `pc.fastQualityChange_` and `pc.shouldSwitchToMedia_`
 */

export interface MockPlaylist {
  id: string;
  uri: string;
  attributes: { BANDWIDTH: number; RESOLUTION?: { width: number; height: number } };
  disabled?: boolean;
  endList: boolean;
  targetDuration: number;
  segments: Array<{ duration: number }>;
}

export function makePlaylists(
  bitratesBps: number[],
  opts: { live?: boolean; targetDuration?: number; segmentCount?: number } = {},
): MockPlaylist[] {
  const target = opts.targetDuration ?? 4;
  const count = opts.segmentCount ?? 48;
  return bitratesBps.map((bw, i) => ({
    id: `pl${i}`,
    uri: `variant${i}.m3u8`,
    attributes: { BANDWIDTH: bw, RESOLUTION: { width: 320 + i * 320, height: 180 + i * 180 } },
    endList: opts.live !== true,
    targetDuration: target,
    segments: Array.from({ length: count }, () => ({ duration: target })),
  }));
}

/** A tiny synchronous event emitter matching video.js's on/off/trigger surface. */
class MockEvents {
  private readonly handlers = new Map<string, Set<(...a: any[]) => void>>();
  on(event: string, fn: (...a: any[]) => void): void {
    if (!this.handlers.has(event)) this.handlers.set(event, new Set());
    this.handlers.get(event)!.add(fn);
  }
  off(event: string, fn: (...a: any[]) => void): void {
    this.handlers.get(event)?.delete(fn);
  }
  trigger(event: string, ...args: any[]): void {
    for (const fn of [...(this.handlers.get(event) ?? [])]) fn(...args);
  }
  listenerCount(event: string): number {
    return this.handlers.get(event)?.size ?? 0;
  }
}

export interface MockPlayerOptions {
  playlists?: MockPlaylist[];
  /** Attach VHS immediately. When false, call `attachVhs()` later. */
  attachVhs?: boolean;
  live?: boolean;
  /** Which `selectPlaylist` VHS starts with. Defaults to a lowest-bitrate selector. */
  defaultSelector?: () => MockPlaylist | undefined;
}

export class MockPlayer extends MockEvents {
  readonly vhs: MockVhs | null = null;
  private techObj: any = null;

  private currentTimeSec = 0;
  private durationSec: number;
  private pausedFlag = false;
  private bufferedRanges: Array<[number, number]> = [[0, 0]];

  readonly playlists: MockPlaylist[];
  /** Every playlist the adapter's `selectPlaylist` override returned. */
  readonly selections: Array<MockPlaylist | undefined> = [];

  constructor(private readonly opts: MockPlayerOptions = {}) {
    super();
    this.playlists =
      opts.playlists ?? makePlaylists([300e3, 750e3, 1200e3, 1850e3, 2850e3, 4300e3]);
    this.durationSec = opts.live ? Infinity : 192;
    if (opts.attachVhs !== false) this.attachVhs();
  }

  attachVhs(): MockVhs {
    const vhs = new MockVhs(this.playlists, this.opts.defaultSelector, this.opts.live === true);
    this.techObj = { vhs };
    (this as { vhs: MockVhs | null }).vhs = vhs;
    return vhs;
  }

  detachVhs(): void {
    this.techObj = null;
    (this as { vhs: MockVhs | null }).vhs = null;
  }

  tech(_safety?: unknown): any {
    return this.techObj;
  }

  currentTime(): number {
    return this.currentTimeSec;
  }
  duration(): number {
    return this.durationSec;
  }
  paused(): boolean {
    return this.pausedFlag;
  }
  buffered(): { length: number; start(i: number): number; end(i: number): number } {
    const r = this.bufferedRanges;
    return {
      length: r.length,
      start: (i: number) => r[i]![0],
      end: (i: number) => r[i]![1],
    };
  }

  // ── Test controls ────────────────────────────────────────────────────────

  setTime(t: number): this {
    this.currentTimeSec = t;
    return this;
  }
  setDuration(d: number): this {
    this.durationSec = d;
    return this;
  }
  setPaused(p: boolean): this {
    this.pausedFlag = p;
    return this;
  }
  /** Set the forward buffer, in seconds ahead of the current playhead. */
  setBuffer(secondsAhead: number): this {
    this.bufferedRanges = [
      [Math.max(0, this.currentTimeSec - 1), this.currentTimeSec + secondsAhead],
    ];
    return this;
  }
  setBufferedRanges(ranges: Array<[number, number]>): this {
    this.bufferedRanges = ranges;
    return this;
  }

  /**
   * Simulate one completed segment download: advance the cumulative counters
   * and fire `bandwidthupdate`, which is exactly what VHS's main segment loader
   * does on each appended segment.
   */
  async completeSegment(bytes: number, downloadMs: number): Promise<void> {
    this.vhs?.addTransfer(bytes, downloadMs);
    this.trigger('bandwidthupdate');
    // Let the adapter's async decide() settle before the test asserts.
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));
  }

  /** Invoke the currently installed `selectPlaylist`, as VHS's `checkABR_` does. */
  checkABR(): MockPlaylist | undefined {
    const chosen = this.vhs?.callSelectPlaylist();
    this.selections.push(chosen);
    if (chosen) this.vhs?.setMedia(chosen);
    return chosen;
  }

  dispose(): void {
    this.trigger('dispose');
  }
}

export class MockVhs {
  /**
   * Declared for the type checker; the real implementation is the accessor
   * installed in the constructor, which mirrors VHS's own
   * `Object.defineProperties(this, { selectPlaylist: { get, set } })`.
   */
  declare selectPlaylist: (() => MockPlaylist | undefined) | null;

  private selectPlaylistFn: (() => MockPlaylist | undefined) | null;
  private media_: MockPlaylist;

  readonly stats = { mediaBytesTransferred: 0, mediaTransferDuration: 0, mediaRequests: 0 };
  readonly playlistController_: any;
  bandwidth = 3_000_000;

  /** Number of times the installed selector has been invoked. */
  selectPlaylistCalls = 0;
  /** Number of times the *original* (default) selector ran. */
  defaultSelectorCalls = 0;
  readonly fastQualityChanges: MockPlaylist[] = [];

  constructor(
    private readonly variants: MockPlaylist[],
    defaultSelector: (() => MockPlaylist | undefined) | undefined,
    private readonly live: boolean,
  ) {
    this.media_ = variants[0]!;

    const fallback = () => {
      this.defaultSelectorCalls++;
      return (defaultSelector ?? (() => this.variants[0]))();
    };
    this.selectPlaylistFn = fallback;

    // Mirrors VHS's own `const self = this` inside its defineProperties block.
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;
    const loader = {
      mediaBytesTransferred: 0,
      mediaTransferDuration: 0,
      mediaRequests: 0,
      bandwidth: 3_000_000,
      roundTrip: 500,
    };
    this.playlistController_ = {
      mainSegmentLoader_: loader,
      bufferBasedABR: false,
      fastQualityChange_(playlist: MockPlaylist) {
        self.fastQualityChanges.push(playlist);
        self.media_ = playlist;
      },
      shouldSwitchToMedia_(playlist: MockPlaylist) {
        return playlist.id !== self.media_.id;
      },
    };

    // Mirror VHS's accessor: the setter binds the assigned function to the handler.
    Object.defineProperty(this, 'selectPlaylist', {
      configurable: true,
      get() {
        return self.selectPlaylistFn;
      },
      set(fn: () => MockPlaylist | undefined) {
        self.selectPlaylistFn = fn.bind(self);
      },
    });
  }

  /** As `PlaylistController.checkABR_` would. */
  callSelectPlaylist(): MockPlaylist | undefined {
    this.selectPlaylistCalls++;
    return this.selectPlaylistFn?.();
  }

  representations(): Array<{
    id: string;
    bandwidth: number;
    width?: number;
    height?: number;
    playlist: MockPlaylist;
    enabled: (v?: boolean) => boolean;
  }> {
    return this.variants
      .filter((p) => p.disabled !== true || true)
      .map((p) => ({
        id: p.id,
        bandwidth: p.attributes.BANDWIDTH,
        width: p.attributes.RESOLUTION?.width,
        height: p.attributes.RESOLUTION?.height,
        playlist: p,
        enabled: (v?: boolean) => {
          if (v !== undefined) p.disabled = !v;
          return p.disabled !== true;
        },
      }));
  }

  get playlists(): { media: () => MockPlaylist; main: { playlists: MockPlaylist[] } } {
    return {
      media: () => this.media_,
      main: { playlists: this.variants },
    };
  }

  setMedia(p: MockPlaylist): void {
    this.media_ = p;
  }

  currentMediaId(): string {
    return this.media_.id;
  }

  addTransfer(bytes: number, durationMs: number): void {
    this.stats.mediaBytesTransferred += bytes;
    this.stats.mediaTransferDuration += durationMs;
    this.stats.mediaRequests += 1;
    const l = this.playlistController_?.mainSegmentLoader_;
    if (l) {
      l.mediaBytesTransferred += bytes;
      l.mediaTransferDuration += durationMs;
      l.mediaRequests += 1;
    }
  }

  isLive(): boolean {
    return this.live;
  }
}

/** A VHS handler missing the pieces the adapter needs. */
export function brokenVhs(): unknown {
  return { stats: {} };
}
