/**
 * The player-independent observation the ABR engine consumes.
 *
 * This is the *only* thing an adapter has to produce. Nothing in here mentions
 * Video.js, HLS, DASH or MediaSource.
 *
 * Mapping to the research code (`src/train.py::StateBuilder`) is spelled out
 * per field so nothing is ambiguous.
 */

import type { Bytes, BitsPerSecond, Kilobits, Pixels, Seconds } from './units.js';

/**
 * One selectable quality level, i.e. one entry of the bitrate ladder.
 *
 * In Video.js/VHS terms this is one media playlist (a `Representation`);
 * in DASH terms one `Representation` inside an `AdaptationSet`.
 */
export interface AbrRepresentation {
  /**
   * Stable identifier, opaque to the engine. The adapter round-trips it to
   * resolve a decision back to a concrete player object.
   *
   * For VHS this is `Representation.id` (VHS's own playlist id).
   */
  readonly id: string;

  /**
   * Declared **peak** bitrate of this rendition, bits per second.
   *
   * For HLS this is the `BANDWIDTH` attribute of the variant; for DASH the
   * `@bandwidth` of the Representation. This is the field the ladder is sorted
   * and matched on.
   */
  readonly bitrateBps: BitsPerSecond;

  /** Video width in pixels, if the manifest declares a resolution. */
  readonly widthPx?: Pixels;
  /** Video height in pixels, if the manifest declares a resolution. */
  readonly heightPx?: Pixels;
  /** Frame rate in frames per second, if declared. */
  readonly frameRate?: number;
  /** RFC 6381 codec string, if declared. */
  readonly codecs?: string;

  /**
   * `false` when the application (not the ABR algorithm) has disabled this
   * rendition — e.g. a manual quality picker, or a codec the browser cannot
   * play. The engine will never select a disabled representation.
   */
  readonly enabled: boolean;
}

/** What happened during the most recently completed segment download. */
export interface AbrSegmentReport {
  /**
   * Transferred size of the segment, **bytes** (payload as counted by the
   * player, i.e. what `chunk_size` is in the research env).
   */
  readonly sizeBytes: Bytes;

  /**
   * Wall-clock time the segment took to download, **seconds**.
   *
   * Maps to `obs["delay"]` in `src/env.py` (which is ms internally, converted
   * to seconds before reaching `StateBuilder`).
   */
  readonly downloadSec: Seconds;

  /**
   * Media duration the segment represents, **seconds**.
   * Maps to `VIDEO_CHUNK_LEN` (4.0 s in the research env, but real streams vary,
   * so it is measured rather than assumed).
   */
  readonly durationSec: Seconds;

  /**
   * Throughput of this specific download, bits per second.
   *
   * Optional: when omitted the engine derives it as
   * `sizeBytes * 8 / downloadSec`, which is exactly what
   * `StateBuilder.update` does. Supply it only if the player measures it more
   * accurately (e.g. excluding request setup time).
   */
  readonly throughputBps?: BitsPerSecond;

  /** Representation id this segment was downloaded at. */
  readonly representationId: string;
}

/** Current playback and buffer state at decision time. */
export interface AbrPlaybackState {
  /**
   * Forward buffer: media seconds buffered **ahead of the playhead**.
   * Maps to `obs["buffer_size"]` / the `buffer_size` field of the research env.
   */
  readonly bufferSec: Seconds;

  /**
   * Rebuffering (stall) time accumulated since the previous decision, seconds.
   * Maps to `obs["rebuf"]`. The engine does not feed this to the network — the
   * trained policy never saw it as an input — but it is carried for telemetry
   * and for the exported `computeReward()` QoE helper.
   */
  readonly rebufferSec: Seconds;

  /** Current playhead position, seconds. */
  readonly currentTimeSec: Seconds;

  /**
   * Total media duration, seconds. `Infinity` for a live stream — see
   * `remainingChunks` handling in `core/state-builder.ts`.
   */
  readonly durationSec: Seconds;

  /** `true` while the element is paused or has not started. */
  readonly paused: boolean;
}

/** Everything the engine needs for one decision. */
export interface AbrObservation {
  /**
   * Monotonic timestamp of this observation, milliseconds.
   * Use `performance.now()` in the browser / `performance.now()` in Node.
   * Only differences are meaningful.
   */
  readonly timestampMs: number;

  /** The ladder as the player currently sees it, in manifest order. */
  readonly representations: readonly AbrRepresentation[];

  /** The representation currently being downloaded/played. */
  readonly currentRepresentationId: string;

  /** Playback + buffer state. */
  readonly playback: AbrPlaybackState;

  /**
   * The segment that just finished downloading, if any.
   *
   * `null` before the first segment completes (cold start). The engine handles
   * that case by feeding the all-zero state the research code starts from.
   */
  readonly lastSegment: AbrSegmentReport | null;

  /**
   * Player-wide throughput estimate at this instant, bits per second.
   * Used only by throughput-based fallback strategies, never by the model.
   */
  readonly estimatedThroughputBps?: BitsPerSecond;

  /**
   * Size in bytes of the *next* segment at each representation, keyed by
   * representation id.
   *
   * The trained policy consumes this (row 2 of the state tensor). When the
   * player cannot know exact sizes — which is normal for HLS/DASH without
   * byte-range indices — the adapter supplies the standard estimate
   * `bitrateBps * segmentDurationSec / 8`, which is exactly how the research
   * environment generates chunk sizes (`src/env.py::_make_chunk_sizes`).
   *
   * If omitted entirely, the engine derives it from the ladder and
   * {@link AbrObservation.nextSegmentDurationSec}.
   */
  readonly nextSegmentSizesBytes?: Readonly<Record<string, Bytes>>;

  /**
   * Media duration of the next segment, seconds. Defaults to the last
   * segment's duration, then to the configured `segmentDurationSec`.
   */
  readonly nextSegmentDurationSec?: Seconds;

  /**
   * Number of segments left in the presentation.
   *
   * `null` for live/unbounded streams. See `core/state-builder.ts` for how the
   * "remaining chunks" model input is filled in that case.
   */
  readonly remainingSegments: number | null;

  /**
   * Total number of segments in the presentation, if known. Used to normalise
   * `remainingSegments`.
   */
  readonly totalSegments?: number | null;

  /** Free-form adapter diagnostics, surfaced in telemetry. Never read by the model. */
  readonly diagnostics?: Readonly<Record<string, unknown>>;
}

/** Convenience: the ladder entry matching an id. */
export function findRepresentation(
  reps: readonly AbrRepresentation[],
  id: string,
): AbrRepresentation | undefined {
  return reps.find((r) => r.id === id);
}

/** The ladder sorted ascending by declared bitrate, filtered to enabled entries. */
export function enabledLadder(
  reps: readonly AbrRepresentation[],
): readonly AbrRepresentation[] {
  return reps
    .filter((r) => r.enabled)
    .slice()
    .sort((a, b) => (a.bitrateBps as number) - (b.bitrateBps as number));
}

/** Kilobit view of a representation's bitrate, for reward / logging. */
export const representationKbps = (r: AbrRepresentation): Kilobits =>
  ((r.bitrateBps as number) / 1000) as Kilobits;
