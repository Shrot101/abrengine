/**
 * Constants mirrored from the research code.
 *
 * These are NOT independently chosen. They are generated into
 * `models/ac3-controller.json` by `export/export_onnx.py`, which reads them
 * directly from `src/model.py`, `src/env.py` and `src/train.py`. The copy here
 * exists so the package has zero-cost synchronous access to them; the test
 * `test/manifest.test.ts` asserts this file and the generated JSON agree, so
 * they cannot drift.
 */

/** Number of input feature rows in the state tensor. `S_INFO` in `src/model.py`. */
export const S_INFO = 6;
/** History window length. `S_LEN` in `src/model.py`. */
export const S_LEN = 8;
/** Size of the model's action space. `A_DIM` in `src/model.py`. */
export const A_DIM = 6;
/** Flattened state tensor length. */
export const STATE_SIZE = S_INFO * S_LEN;

/** Row indices of the state tensor. See `src/model.py` module docstring. */
export const ROW = {
  /** Throughput history, Mbps, oldest at index 0, newest at index S_LEN-1. */
  THROUGHPUT: 0,
  /** Download-time history, seconds, oldest at 0, newest at S_LEN-1. */
  DOWNLOAD: 1,
  /** Next-chunk sizes for each action slot, bytes / CHUNK_NORM, at indices 0..A_DIM-1. */
  CHUNK_SIZES: 2,
  /** Buffer level, seconds / BUFFER_NORM. Only index S_LEN-1 is read by the model. */
  BUFFER: 3,
  /** Remaining chunks / NUM_CHUNKS. Only index S_LEN-1 is read. */
  REMAINING: 4,
  /** Last action index / (A_DIM-1). Only index S_LEN-1 is read. */
  LAST_BITRATE: 5,
} as const;

/** `BITRATES` from `src/env.py`, kbps. */
export const TRAINING_LADDER_KBPS: readonly number[] = [300, 750, 1200, 1850, 2850, 4300];

/** `BUFFER_NORM` from `src/train.py`. */
export const BUFFER_NORM_SEC = 10.0;
/** `CHUNK_NORM` from `src/train.py`. */
export const CHUNK_NORM_BYTES = 1e6;
/** `THROUGHPUT_NORM` from `src/train.py`. Applied to Mbps. */
export const THROUGHPUT_NORM = 1.0;
/** `NUM_CHUNKS` from `src/env.py`. */
export const TOTAL_CHUNKS_NORM = 48;
/** `VIDEO_CHUNK_LEN` from `src/env.py`, seconds. */
export const SEGMENT_DURATION_SEC = 4.0;

/** Reward weights from `src/env.py::step`. Exposed for offline QoE scoring. */
export const REWARD = {
  /** λ — rebuffer penalty weight. */
  rebufferPenalty: 4.3,
  /** μ — smoothness penalty weight. */
  smoothnessPenalty: 1.0,
} as const;

/** Name of the bundled model asset. */
export const BUNDLED_MODEL_FILE = 'ac3-controller.onnx';

/** Graph IO names produced by `export/export_onnx.py`. */
export const GRAPH_IO = {
  input: 'state',
  probs: 'action_probs',
  value: 'state_value',
} as const;
