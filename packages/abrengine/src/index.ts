/**
 * abrengine — public API (framework-independent core).
 *
 * ```ts
 * import { AbrEngine } from 'abrengine';
 *
 * const abr = new AbrEngine({ model: 'ac3', telemetry: true });
 * await abr.initialize();
 * const decision = await abr.decide(observation);
 * ```
 *
 * Nothing exported from this entry point imports Video.js, the DOM, or an ONNX
 * runtime at module load time. The runtime is imported lazily inside
 * `initialize()`.
 */

export { AbrEngine, type AbrEngineStatus } from './core/engine.js';
export { StateBuilder, type StateStep, type StateNormalisation } from './core/state-builder.js';
export {
  actionToRepresentation,
  representationToAction,
  nextSegmentBytesByAction,
  makeLadderContext,
  type LadderContext,
  type LadderMapping,
} from './core/ladder.js';
export {
  validateObservation,
  validateModelOutput,
  type ValidationResult,
  type CleanObservation,
} from './core/validate.js';
export { computeReward, type RewardTerms, type RewardOptions } from './core/reward.js';
export {
  applySafety,
  resolveSafety,
  type AbrSafetyConfig,
  type SafetyClamp,
  type ResolvedSafety,
} from './core/safety.js';
export { compileFallback } from './fallback/strategies.js';
export { AbrError, AbrRuntimeError, AbrModelLoadError, AbrConfigError } from './core/errors.js';
export { Emitter } from './utils/emitter.js';
export { bundledModelUrl } from './model/resolve-source.js';

export {
  A_DIM,
  S_INFO,
  S_LEN,
  STATE_SIZE,
  ROW,
  TRAINING_LADDER_KBPS,
  BUFFER_NORM_SEC,
  CHUNK_NORM_BYTES,
  THROUGHPUT_NORM,
  TOTAL_CHUNKS_NORM,
  SEGMENT_DURATION_SEC,
  REWARD,
  GRAPH_IO,
} from './model/manifest.js';

export * from './types/index.js';

/** Package version. Kept in sync with package.json by the build. */
export const VERSION = '0.1.0';
