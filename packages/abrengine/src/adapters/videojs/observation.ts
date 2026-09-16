/**
 * Video.js/VHS player state -> generic `AbrObservation`.
 *
 * This module contains *no* ABR logic. It reads, converts units, and hands the
 * result to the engine.
 *
 * Unit conversions performed here (each one is a place a bug would hide, so
 * they are all in one file):
 *
 * | VHS value                            | unit VHS uses | unit we emit |
 * |--------------------------------------|---------------|--------------|
 * | `playlist.attributes.BANDWIDTH`      | bits/s        | bits/s       |
 * | `vhs.bandwidth`                      | bits/s        | bits/s       |
 * | `mediaTransferDuration` delta        | milliseconds  | seconds      |
 * | `mediaBytesTransferred` delta        | bytes         | bytes        |
 * | `player.buffered()` / `currentTime()`| seconds       | seconds      |
 */

import type {
  AbrObservation,
  AbrRepresentation,
  AbrSegmentReport,
} from '../../types/observation.js';
import { bitsPerSecond, bytes, pixels, seconds } from '../../types/units.js';
import { now } from '../../utils/clock.js';
import type { VhsBridge } from './vhs-bridge.js';

/** Rolling per-segment measurement derived from VHS's cumulative counters. */
export interface SegmentTracker {
  lastBytes: number;
  lastDurationMs: number;
  lastRequests: number;
  /** Rendition id in force when the previous sample was taken. */
  lastRepresentationId: string | null;
  /** Playhead position at the previous sample, used to estimate stall time. */
  lastCurrentTimeSec: number;
  /** Wall-clock at the previous sample, milliseconds. */
  lastSampleMs: number;
  primed: boolean;
}

export function createTracker(): SegmentTracker {
  return {
    lastBytes: 0,
    lastDurationMs: 0,
    lastRequests: 0,
    lastRepresentationId: null,
    lastCurrentTimeSec: 0,
    lastSampleMs: 0,
    primed: false,
  };
}

/** Forward buffer in seconds: buffered time ahead of the playhead. */
export function forwardBufferSec(player: any): number {
  try {
    const ranges = player?.buffered?.();
    const t = player?.currentTime?.() ?? 0;
    if (!ranges || typeof ranges.length !== 'number') return 0;
    for (let i = 0; i < ranges.length; i++) {
      const start = ranges.start(i);
      const end = ranges.end(i);
      // 0.1 s of slack so a playhead sitting exactly on a range edge still counts.
      if (t >= start - 0.1 && t <= end) return Math.max(0, end - t);
    }
    return 0;
  } catch {
    return 0;
  }
}

function toRepresentations(bridge: VhsBridge): AbrRepresentation[] {
  const out: AbrRepresentation[] = [];
  for (const r of bridge.representations()) {
    const bw = r.bandwidth ?? r.playlist?.attributes?.BANDWIDTH;
    if (typeof bw !== 'number' || !Number.isFinite(bw) || bw <= 0) continue;

    let enabled = true;
    try {
      // VHS's `enabled()` getter reports the app-level enable flag. `disabled`
      // on the playlist is the same thing seen from the other side.
      enabled = r.enabled() !== false && r.playlist?.disabled !== true;
    } catch {
      enabled = r.playlist?.disabled !== true;
    }

    out.push({
      id: r.id,
      bitrateBps: bitsPerSecond(bw),
      ...(typeof r.width === 'number' ? { widthPx: pixels(r.width) } : {}),
      ...(typeof r.height === 'number' ? { heightPx: pixels(r.height) } : {}),
      ...(typeof r.frameRate === 'number' ? { frameRate: r.frameRate } : {}),
      ...(typeof r.codecs === 'string' ? { codecs: r.codecs } : {}),
      enabled,
    });
  }
  return out;
}

export interface BuildObservationOptions {
  /** Fallback segment duration when the manifest does not declare one, seconds. */
  defaultSegmentDurationSec: number;
  /**
   * Treat live streams as having unbounded remaining segments (`null`), which
   * makes the engine pin the "remaining chunks" model input to 1.0.
   */
  liveRemainingSegments: number | null;
}

/**
 * Sample the player once.
 *
 * `tracker` is mutated: cumulative counters are diffed against the previous
 * sample to recover *this* segment's size and download time.
 *
 * Returns `null` when the player is not ready enough to observe.
 */
export function buildObservation(
  player: any,
  bridge: VhsBridge,
  tracker: SegmentTracker,
  opts: BuildObservationOptions,
): AbrObservation | null {
  if (!bridge.usable()) return null;

  const representations = toRepresentations(bridge);
  if (representations.length === 0) return null;

  const currentId = bridge.currentPlaylistId();
  if (!currentId) return null;

  const counters = bridge.counters();
  const nowMs = now();
  const currentTimeSec = Number(player?.currentTime?.() ?? 0) || 0;
  const bufferSec = forwardBufferSec(player);
  const durationRaw = Number(player?.duration?.() ?? 0);
  const durationSec = Number.isFinite(durationRaw) && durationRaw > 0 ? durationRaw : Infinity;

  const targetDur = bridge.targetDurationSec() ?? opts.defaultSegmentDurationSec;

  // ── Diff the cumulative counters into a per-segment report ──────────────
  let lastSegment: AbrSegmentReport | null = null;
  if (tracker.primed) {
    const dBytes = counters.bytes - tracker.lastBytes;
    const dMs = counters.durationMs - tracker.lastDurationMs;
    if (dBytes > 0 && dMs > 0) {
      lastSegment = {
        sizeBytes: bytes(dBytes),
        downloadSec: seconds(dMs / 1000),
        durationSec: seconds(targetDur),
        representationId: tracker.lastRepresentationId ?? currentId,
      };
    }
  }

  // ── Rebuffer estimate ──────────────────────────────────────────────────
  // Wall-clock elapsed minus media-time advanced, while unpaused. This is an
  // estimate: it also catches slow seeks. It never feeds the model (the trained
  // policy has no rebuffer input) — only telemetry and QoE scoring.
  let rebufferSec = 0;
  if (tracker.primed && tracker.lastSampleMs > 0) {
    const paused = player?.paused?.() === true;
    if (!paused) {
      const wallSec = (nowMs - tracker.lastSampleMs) / 1000;
      const mediaSec = currentTimeSec - tracker.lastCurrentTimeSec;
      const stalled = wallSec - Math.max(mediaSec, 0);
      // Ignore sub-frame noise and anything implausibly large (a seek).
      if (stalled > 0.05 && stalled < 60) rebufferSec = stalled;
    }
  }

  // ── Remaining segments ─────────────────────────────────────────────────
  let remainingSegments: number | null;
  let totalSegments: number | null = null;
  if (bridge.isLive()) {
    remainingSegments = opts.liveRemainingSegments;
  } else {
    totalSegments = bridge.segmentCount();
    remainingSegments =
      Number.isFinite(durationSec) && targetDur > 0
        ? Math.max(0, (durationSec - currentTimeSec) / targetDur)
        : null;
  }

  const bandwidth = bridge.bandwidthBps();

  // ── Advance the tracker ────────────────────────────────────────────────
  tracker.lastBytes = counters.bytes;
  tracker.lastDurationMs = counters.durationMs;
  tracker.lastRequests = counters.requests;
  tracker.lastRepresentationId = currentId;
  tracker.lastCurrentTimeSec = currentTimeSec;
  tracker.lastSampleMs = nowMs;
  tracker.primed = true;

  return {
    timestampMs: nowMs,
    representations,
    currentRepresentationId: currentId,
    playback: {
      bufferSec: seconds(bufferSec),
      rebufferSec: seconds(rebufferSec),
      currentTimeSec: seconds(currentTimeSec),
      durationSec: seconds(durationSec),
      paused: player?.paused?.() === true,
    },
    lastSegment,
    ...(bandwidth !== null ? { estimatedThroughputBps: bitsPerSecond(bandwidth) } : {}),
    nextSegmentDurationSec: seconds(targetDur),
    remainingSegments,
    totalSegments,
    diagnostics: {
      statsSource: counters.source,
      mediaRequests: counters.requests,
      live: bridge.isLive(),
    },
  };
}
