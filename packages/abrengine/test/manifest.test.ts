/**
 * Guards against the TypeScript constants drifting from the research code.
 *
 * `src/model/manifest.ts` hard-codes the dimensions, normalisation divisors and
 * bitrate ladder for zero-cost synchronous access. Those numbers originate in
 * `src/model.py`, `src/env.py` and `src/train.py`, and are written into
 * `models/ac3-controller.json` by `export/export_onnx.py` at export time.
 *
 * If someone retrains with a different ladder or a different BUFFER_NORM and
 * re-exports, this test fails — which is exactly what should happen, because
 * every state tensor the JS package builds would otherwise be silently wrong.
 */

import { describe, expect, it } from 'vitest';

import {
  A_DIM,
  BUFFER_NORM_SEC,
  CHUNK_NORM_BYTES,
  GRAPH_IO,
  REWARD,
  S_INFO,
  S_LEN,
  SEGMENT_DURATION_SEC,
  THROUGHPUT_NORM,
  TOTAL_CHUNKS_NORM,
  TRAINING_LADDER_KBPS,
} from '../src/model/manifest.js';
import { loadManifest, loadModelBytes } from './helpers.js';

const manifest = loadManifest();

describe('model manifest ↔ TypeScript constants', () => {
  it('dimensions match', () => {
    expect(manifest.dims.S_INFO).toBe(S_INFO);
    expect(manifest.dims.S_LEN).toBe(S_LEN);
    expect(manifest.dims.A_DIM).toBe(A_DIM);
  });

  it('normalisation constants match src/train.py', () => {
    expect(manifest.normalisation.BUFFER_NORM).toBe(BUFFER_NORM_SEC);
    expect(manifest.normalisation.CHUNK_NORM).toBe(CHUNK_NORM_BYTES);
    expect(manifest.normalisation.THROUGHPUT_NORM).toBe(THROUGHPUT_NORM);
    expect(manifest.normalisation.NUM_CHUNKS).toBe(TOTAL_CHUNKS_NORM);
  });

  it('the bitrate ladder matches src/env.py', () => {
    expect(manifest.ladder.bitratesKbps).toEqual([...TRAINING_LADDER_KBPS]);
    expect(manifest.ladder.numBitrates).toBe(A_DIM);
    expect(manifest.ladder.chunkDurationSeconds).toBe(SEGMENT_DURATION_SEC);
  });

  it('graph IO names match what the exporter produced', () => {
    expect(manifest.graph.inputs[0].name).toBe(GRAPH_IO.input);
    expect(manifest.graph.outputs.map((o: { name: string }) => o.name)).toEqual([
      GRAPH_IO.probs,
      GRAPH_IO.value,
    ]);
  });

  it('the graph input has the shape the StateBuilder produces', () => {
    expect(manifest.graph.inputs[0].shape).toEqual(['batch', S_INFO, S_LEN]);
    expect(manifest.graph.outputs[0].shape).toEqual(['batch', A_DIM]);
  });

  it('uses only ONNX ops that ORT Web supports on the wasm backend', () => {
    const supported = new Set([
      'Concat',
      'Constant',
      'Conv',
      'Flatten',
      'Gather',
      'Gemm',
      'Relu',
      'Slice',
      'Softmax',
      'Unsqueeze',
      'Reshape',
      'Transpose',
      'Squeeze',
      'MatMul',
      'Add',
    ]);
    const unexpected = (manifest.graph.opTypes as string[]).filter((op) => !supported.has(op));
    expect(
      unexpected,
      `unvalidated ops in the exported graph: ${unexpected.join(', ')}`,
    ).toHaveLength(0);
  });

  it('the shipped model file matches the manifest hash and size', async () => {
    const bytes = loadModelBytes();
    expect(bytes.byteLength).toBe(manifest.modelBytes);

    const { createHash } = await import('node:crypto');
    const sha = createHash('sha256').update(bytes).digest('hex');
    expect(sha).toBe(manifest.modelSha256);
  });

  it('records which checkpoint it came from', () => {
    expect(manifest.source.checkpoint).toMatch(/\.pt$/);
    expect(manifest.source.checkpointSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(manifest.source.opset).toBeGreaterThanOrEqual(11);
  });

  it('documents the action semantics', () => {
    expect(manifest.actionSemantics).toMatch(/argmax/);
    expect(manifest.actionSemantics).toMatch(/lowest quality/);
  });

  it('reward weights match src/env.py', () => {
    expect(REWARD.rebufferPenalty).toBe(4.3);
    expect(REWARD.smoothnessPenalty).toBe(1.0);
  });

  it('the model is small enough to ship in an npm package', () => {
    // 265 863 float32 parameters ≈ 1.06 MB, plus graph overhead.
    expect(manifest.modelBytes).toBeLessThan(2 * 1024 * 1024);
  });
});
