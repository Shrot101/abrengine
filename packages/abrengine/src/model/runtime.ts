/**
 * Inference runtime resolution.
 * =============================
 *
 * Why `onnxruntime-web` is the default in **both** browser and Node:
 *
 * - `onnxruntime-node` ships native binaries that its postinstall script
 *   downloads from `api.nuget.org`. That is a real install-time failure mode on
 *   locked-down CI and corporate networks, and it makes the package unusable in
 *   sandboxes that restrict egress.
 * - `onnxruntime-web`'s WASM backend runs unmodified under Node ≥18 and is
 *   validated here against PyTorch to 7.7e-7 max deviation.
 * - This model is 265k parameters. Measured p50 inference is ~0.26 ms on the
 *   WASM backend; the native runtime's advantage is irrelevant at that size.
 *
 * `onnxruntime-node` is still supported for anyone who wants it — set
 * `inference.runtime: 'onnxruntime-node'`.
 *
 * Both runtimes are **optional peer dependencies** and are imported lazily via
 * a dynamic `import()`, so a consumer who supplies their own
 * `{ type: 'session' }` never pays for either.
 */

import type { AbrInferenceConfig, AbrInferenceSession } from '../types/config.js';
import { A_DIM, GRAPH_IO, S_INFO, S_LEN } from './manifest.js';
import { AbrRuntimeError } from '../core/errors.js';

/** Structural type for the bits of the ORT API we touch. Avoids a type-only dep. */
interface OrtLike {
  env: { wasm: { numThreads?: number; wasmPaths?: string; simd?: boolean } };
  Tensor: new (type: 'float32', data: Float32Array, dims: readonly number[]) => unknown;
  InferenceSession: {
    create(
      model: Uint8Array | string,
      options?: { executionProviders?: readonly string[] },
    ): Promise<OrtSessionLike>;
  };
}

interface OrtSessionLike {
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array }>>;
  release?(): Promise<void>;
  inputNames?: readonly string[];
  outputNames?: readonly string[];
}

export interface ResolvedRuntime {
  /** `'onnxruntime-web'` or `'onnxruntime-node'`. */
  readonly name: string;
  readonly ort: OrtLike;
}

// Module specifiers are indirected through a variable so bundlers do not try to
// statically resolve (and warn about) an optional peer that may not be installed.
async function importOrt(name: string): Promise<OrtLike> {
  const spec = name;
  return (await import(
    /* @vite-ignore */ /* webpackIgnore: true */ spec
  )) as unknown as OrtLike;
}

export async function resolveRuntime(
  cfg: AbrInferenceConfig | undefined,
): Promise<ResolvedRuntime> {
  const want = cfg?.runtime ?? 'auto';
  const order = want === 'auto' ? ['onnxruntime-web', 'onnxruntime-node'] : [want];

  const failures: string[] = [];
  for (const name of order) {
    try {
      const ort = await importOrt(name);
      if (!ort?.InferenceSession?.create) {
        failures.push(`${name}: module loaded but has no InferenceSession.create`);
        continue;
      }
      return { name, ort };
    } catch (err) {
      failures.push(`${name}: ${(err as Error)?.message ?? String(err)}`);
    }
  }

  throw new AbrRuntimeError(
    want === 'auto'
      ? `No ONNX runtime available. Install one of: npm i onnxruntime-web (recommended, ` +
          `works in browser and Node) or npm i onnxruntime-node. Details: ${failures.join('; ')}`
      : `Requested runtime '${want}' could not be loaded: ${failures.join('; ')}`,
  );
}

/** Apply the WASM knobs that must be set before the first session is created. */
export function configureRuntime(
  rt: ResolvedRuntime,
  cfg: AbrInferenceConfig | undefined,
): void {
  const wasm = rt.ort.env?.wasm;
  if (!wasm) return;
  // Single-threaded by default: multi-threaded WASM needs cross-origin isolation
  // (COOP/COEP headers), which most sites do not have. A 265k-parameter model
  // gains nothing from threads anyway.
  wasm.numThreads = cfg?.wasmThreads ?? 1;
  if (cfg?.wasmPaths) wasm.wasmPaths = cfg.wasmPaths;
}

/**
 * Wrap an ORT session in the narrow {@link AbrInferenceSession} contract.
 *
 * The input tensor is allocated once and reused across calls: one decision per
 * segment means this is not hot, but an ABR path should not be generating
 * garbage during playback either.
 */
export function wrapSession(rt: ResolvedRuntime, session: OrtSessionLike): AbrInferenceSession {
  const dims = [1, S_INFO, S_LEN] as const;
  const scratch = new Float32Array(S_INFO * S_LEN);

  const inputName = session.inputNames?.[0] ?? GRAPH_IO.input;
  const probsName =
    session.outputNames?.find((n) => n === GRAPH_IO.probs) ??
    session.outputNames?.[0] ??
    GRAPH_IO.probs;
  const valueName =
    session.outputNames?.find((n) => n === GRAPH_IO.value) ??
    session.outputNames?.[1] ??
    GRAPH_IO.value;

  return {
    async run(state: Float32Array) {
      if (state.length !== scratch.length) {
        throw new AbrRuntimeError(
          `state tensor has ${state.length} elements, expected ${scratch.length}`,
        );
      }
      scratch.set(state);
      const tensor = new rt.ort.Tensor('float32', scratch, dims);
      const out = await session.run({ [inputName]: tensor });

      const probs = out[probsName]?.data;
      if (!probs) {
        throw new AbrRuntimeError(
          `model output '${probsName}' missing; got [${Object.keys(out).join(', ')}]`,
        );
      }
      const value = out[valueName]?.data?.[0];

      return {
        actionProbs: probs.length === A_DIM ? probs : probs.subarray(0, A_DIM),
        stateValue: typeof value === 'number' ? value : Number.NaN,
      };
    },
    async release() {
      await session.release?.();
    },
  };
}

/** Create and wrap a session from raw model bytes. */
export async function createSession(
  rt: ResolvedRuntime,
  model: Uint8Array,
  cfg: AbrInferenceConfig | undefined,
): Promise<AbrInferenceSession> {
  const executionProviders = cfg?.executionProviders ?? ['wasm'];
  const session = await rt.ort.InferenceSession.create(model, { executionProviders });
  return wrapSession(rt, session);
}
