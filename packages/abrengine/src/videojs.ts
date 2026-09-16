/**
 * `abrengine/videojs` — the Video.js / VHS integration.
 *
 * ```js
 * import videojs from 'video.js';
 * import { AbrEngine } from 'abrengine';
 * import { VideoJSAbrAdapter } from 'abrengine/videojs';
 *
 * const player = videojs('video');
 * const abr = new AbrEngine({ model: 'ac3' });
 * const adapter = new VideoJSAbrAdapter({ player, abr });
 * await adapter.initialize();
 * ```
 *
 * `video.js` is an optional peer dependency and is never imported here — the
 * adapter only ever touches the player object you hand it, so this entry point
 * adds nothing to your bundle beyond the adapter itself.
 */

export {
  VideoJSAbrAdapter,
  attachAbrEngine,
  type VideoJSAbrAdapterOptions,
  type AdapterEventMap,
  type AdapterAttachEvent,
  type AdapterApplyEvent,
  type AdapterDetachEvent,
} from './adapters/videojs/index.js';

export {
  VhsBridge,
  type VhsPlaylist,
  type VhsRepresentation,
  type TransferCounters,
  type StatsSource,
} from './adapters/videojs/vhs-bridge.js';

export {
  buildObservation,
  createTracker,
  forwardBufferSec,
  type SegmentTracker,
  type BuildObservationOptions,
} from './adapters/videojs/observation.js';
