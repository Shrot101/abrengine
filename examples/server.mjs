/**
 * Example dev server.
 *
 * Serves the four examples plus everything they need, with no bundler:
 *
 *   /                     an index of the examples
 *   /vanilla-videojs/     the minimal integration
 *   /telemetry/           a live decision inspector
 *   /fallback/            failure injection and recovery
 *   /custom-model/        loading your own .onnx
 *
 *   /pkg/                 packages/abrengine/dist   (the built package)
 *   /models/              packages/abrengine/models (the ONNX model)
 *   /vendor/              video.js + onnxruntime-web from node_modules
 *   /stream/              the local HLS stream, if you generated one
 *
 * Run: npm run example    (from the repo root)
 */

import { createServer } from 'node:http';
import { createReadStream, existsSync } from 'node:fs';
import { readdir, stat } from 'node:fs/promises';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..');
const PKG = join(REPO, 'packages/abrengine');

const NM_CANDIDATES = [join(REPO, 'node_modules'), join(PKG, 'node_modules')];
const vendor = (rel) => NM_CANDIDATES.map((d) => join(d, rel)).find(existsSync) ?? '';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.cjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.map': 'application/json',
  '.onnx': 'application/octet-stream',
  '.wasm': 'application/wasm',
  '.m3u8': 'application/vnd.apple.mpegurl',
  '.ts': 'video/mp2t',
  '.m4s': 'video/mp4',
  '.mp4': 'video/mp4',
};

const VENDOR = {
  '/vendor/video.js': vendor('video.js/dist/video.js'),
  '/vendor/video-js.css': vendor('video.js/dist/video-js.min.css'),
  '/vendor/ort.mjs': vendor('onnxruntime-web/dist/ort.bundle.min.mjs'),
};

const EXAMPLES = [
  ['vanilla-videojs', 'The minimal integration — 20 lines to swap in the trained controller.'],
  ['telemetry', 'A live inspector: model input, softmax output, chosen rendition, latency.'],
  ['fallback', 'Break the model on purpose and watch the player keep going.'],
  ['custom-model', 'Point the engine at your own exported .onnx file.'],
];

function indexPage(hasStream) {
  const warn = hasStream
    ? ''
    : `<p class="warn">No local stream found. Run <code>npm run make-stream</code> first,
       or edit an example to point at your own HLS URL.</p>`;
  return `<!doctype html><meta charset="utf-8"><title>abrengine examples</title>
<style>
 body{font:16px/1.6 system-ui,sans-serif;max-width:46rem;margin:3rem auto;padding:0 1.5rem;background:#0f1115;color:#e6e8ee}
 a{color:#7cc4ff} h1{font-size:1.5rem} li{margin:.9rem 0}
 code{background:#1b1f27;padding:.1rem .35rem;border-radius:4px;font-size:.9em}
 .warn{background:#3a2a12;border-left:3px solid #e0a44a;padding:.75rem 1rem;border-radius:4px}
 .d{color:#98a2b3;font-size:.92rem}
</style>
<h1>abrengine examples</h1>
${warn}
<ul>${EXAMPLES.map(
    ([slug, desc]) => `<li><a href="/${slug}/">${slug}</a><br><span class="d">${desc}</span></li>`,
  ).join('')}</ul>`;
}

async function resolveRequest(urlPath) {
  if (urlPath === '/') return { index: true };
  for (const [prefix, dir] of [
    ['/pkg/', join(PKG, 'dist')],
    ['/models/', join(PKG, 'models')],
    ['/stream/', join(REPO, 'test-e2e/stream')],
  ]) {
    if (urlPath.startsWith(prefix)) {
      const rel = urlPath.slice(prefix.length);
      return rel.includes('..') ? null : { file: join(dir, rel) };
    }
  }
  if (urlPath === '/_shared.js') return { file: join(HERE, '_shared.js') };
  if (urlPath === '/codec-shim.js') return { file: join(HERE, 'codec-shim.js') };
  if (VENDOR[urlPath]) return { file: VENDOR[urlPath] };
  if (urlPath.startsWith('/vendor/')) {
    const rel = urlPath.slice('/vendor/'.length);
    return rel.includes('..') ? null : { file: vendor(join('onnxruntime-web/dist', rel)) };
  }
  for (const [slug] of EXAMPLES) {
    if (urlPath === `/${slug}` || urlPath === `/${slug}/`) {
      return { file: join(HERE, slug, 'index.html') };
    }
    if (urlPath.startsWith(`/${slug}/`)) {
      const rel = urlPath.slice(slug.length + 2);
      return rel.includes('..') ? null : { file: join(HERE, slug, rel) };
    }
  }
  return null;
}

const hasStream = existsSync(join(REPO, 'test-e2e/stream/master.m3u8'));

const server = createServer(async (req, res) => {
  const urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const hit = await resolveRequest(urlPath);

  if (hit?.index) {
    const body = indexPage(hasStream);
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }).end(body);
    return;
  }
  if (!hit?.file) {
    res.writeHead(404).end(`not found: ${urlPath}`);
    return;
  }
  try {
    const st = await stat(hit.file);
    res.writeHead(200, {
      'content-type': MIME[extname(hit.file)] ?? 'application/octet-stream',
      'content-length': st.size,
      'cache-control': 'no-store',
    });
    createReadStream(hit.file).pipe(res);
  } catch {
    res.writeHead(404).end(`not found: ${urlPath}`);
  }
});

const PORT = Number(process.env.PORT ?? 8080);
server.listen(PORT, () => {
  console.log(`\n  abrengine examples → http://localhost:${PORT}\n`);
  for (const [slug, desc] of EXAMPLES) {
    console.log(`    http://localhost:${PORT}/${slug}/`);
    console.log(`      ${desc}`);
  }
  if (!hasStream) {
    console.log(
      `\n  ⚠ No local HLS stream. Run "npm run make-stream" to generate one,\n` +
        `    or edit the examples to use your own stream URL.`,
    );
  }
  console.log('');
  void readdir;
});
