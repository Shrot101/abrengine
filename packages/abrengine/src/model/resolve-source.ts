/**
 * Turning an {@link AbrModelSource} into bytes.
 *
 * Distribution strategy (see README "Model distribution"): the ONNX file is
 * **bundled in the npm package** at `models/ac3-controller.onnx` (1.02 MB) and
 * resolved lazily by URL, never inlined into the JS bundle.
 *
 * `'ac3'` resolves through `new URL('../models/…', import.meta.url)`. Vite,
 * webpack 5, Rollup and Node all understand that form and emit/resolve the file
 * as an asset. Bundlers that do not (and any CDN-hosted deployment) should pass
 * `{ type: 'url' }` explicitly — which is also the right choice when you want
 * the model served from your own CDN with your own cache headers.
 */

import type { AbrInferenceSession, AbrModelSource } from '../types/config.js';
import { AbrModelLoadError } from '../core/errors.js';
import { BUNDLED_MODEL_FILE } from './manifest.js';

const BUNDLED_ALIASES = new Set(['ac3', 'ac3-controller']);

/** URL of the model bundled in this package. Exported so apps can pre-fetch it. */
export function bundledModelUrl(): string {
  // `import.meta.url` is rewritten by tsup for the CJS build, and the CJS output
  // gets a `__filename`-based shim, so this works in both module systems.
  return new URL(`../models/${BUNDLED_MODEL_FILE}`, import.meta.url).href;
}

async function fetchBytes(url: string, init?: RequestInit): Promise<Uint8Array> {
  // Node ≥18 has global fetch; file: URLs are not fetchable there, so read them.
  if (url.startsWith('file:')) {
    try {
      const { readFile } = await import('node:fs/promises');
      const { fileURLToPath } = await import('node:url');
      return new Uint8Array(await readFile(fileURLToPath(url)));
    } catch (err) {
      throw new AbrModelLoadError(`could not read model from ${url}`, { cause: err });
    }
  }

  if (typeof fetch !== 'function') {
    throw new AbrModelLoadError(
      `no global fetch available to load the model from ${url}; pass ` +
        `{ type: 'buffer' } with bytes you have loaded yourself`,
    );
  }

  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (err) {
    throw new AbrModelLoadError(`network error fetching model from ${url}`, { cause: err });
  }
  if (!res.ok) {
    throw new AbrModelLoadError(`model fetch failed: ${res.status} ${res.statusText} (${url})`);
  }
  const buf = await res.arrayBuffer();
  if (buf.byteLength === 0) {
    throw new AbrModelLoadError(`model fetched from ${url} is empty`);
  }
  return new Uint8Array(buf);
}

export interface ResolvedModelSource {
  /** `null` when the source was a pre-built session. */
  bytes: Uint8Array | null;
  /** Set only for `{ type: 'session' }`. */
  session: AbrInferenceSession | null;
  /** Where it came from, for telemetry. */
  origin: string;
}

export async function resolveModelSource(
  source: AbrModelSource = 'ac3',
): Promise<ResolvedModelSource> {
  if (typeof source === 'string') {
    if (!BUNDLED_ALIASES.has(source)) {
      throw new AbrModelLoadError(
        `unknown bundled model '${source}'. Known: ${[...BUNDLED_ALIASES].join(', ')}. ` +
          `To load your own, pass { type: 'url' } or { type: 'buffer' }.`,
      );
    }
    const url = bundledModelUrl();
    return { bytes: await fetchBytes(url), session: null, origin: `bundled:${url}` };
  }

  switch (source.type) {
    case 'url':
      return {
        bytes: await fetchBytes(source.url, source.init),
        session: null,
        origin: `url:${source.url}`,
      };

    case 'buffer': {
      const b = source.buffer;
      const bytes = b instanceof Uint8Array ? b : new Uint8Array(b);
      if (bytes.byteLength === 0) throw new AbrModelLoadError('model buffer is empty');
      return { bytes, session: null, origin: 'buffer' };
    }

    case 'session':
      if (!source.session || typeof source.session.run !== 'function') {
        throw new AbrModelLoadError('model source { type: "session" } has no runnable session');
      }
      return { bytes: null, session: source.session, origin: 'session' };

    default:
      throw new AbrModelLoadError(`unsupported model source: ${JSON.stringify(source)}`);
  }
}
