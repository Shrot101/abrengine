/**
 * End-to-end browser test.
 * ========================
 *
 * This is the test that backs the claim "it works with Video.js in a browser".
 * Nothing here is mocked:
 *
 *   - a real 6-rendition HLS stream, produced by ffmpeg (`make-stream.sh`)
 *   - real video.js + videojs-http-streaming from node_modules
 *   - the real built package from `packages/abrengine/dist`
 *   - the real exported ONNX model, executed by onnxruntime-web's WASM backend
 *   - real Chromium, playing real video with MediaSource
 *
 * It asserts:
 *   1. the model loads and runs in the browser
 *   2. the adapter attaches to the real VhsHandler and finds the surface it needs
 *   3. decisions come from the model, not a fallback
 *   4. VHS actually switches renditions in response
 *   5. browser inference matches Node inference on an identical input
 *   6. playback produces no player errors
 *   7. the fallback path takes over when the model is removed, without breaking playback
 *
 * Run: node test-e2e/run.mjs
 */

import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { createReadStream, existsSync } from 'node:fs';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..');
const PKG = join(REPO, 'packages/abrengine');
// npm workspaces hoist most dependencies to the repo root; a few stay local.
const NM_CANDIDATES = [join(REPO, 'node_modules'), join(PKG, 'node_modules')];
const NM = NM_CANDIDATES[0];

/** First candidate directory that actually contains `rel`. */
function vendor(rel) {
  for (const dir of NM_CANDIDATES) {
    const p = join(dir, rel);
    if (existsSync(p)) return p;
  }
  return join(NM, rel);
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.onnx': 'application/octet-stream',
  '.wasm': 'application/wasm',
  '.m3u8': 'application/vnd.apple.mpegurl',
  '.ts': 'video/mp2t',
  '.m4s': 'video/mp4',
  '.mp4': 'video/mp4',
  '.map': 'application/json',
};

/** Route prefix → directory on disk. */
const ROUTES = [
  ['/pkg/', join(PKG, 'dist')],
  ['/models/', join(PKG, 'models')],
  ['/stream/', join(HERE, 'stream')],
  ['/vendor/ort/', vendor('onnxruntime-web/dist')],
];

const VENDOR_FILES = {
  '/vendor/video.js': vendor('video.js/dist/video.js'),
  '/vendor/video-js.css': vendor('video.js/dist/video-js.min.css'),
  // The self-contained wasm bundle: no separate .wasm fetch, no CDN.
  '/vendor/ort.mjs': vendor('onnxruntime-web/dist/ort.bundle.min.mjs'),
};

async function resolveRequest(urlPath) {
  if (urlPath === '/' || urlPath === '/index.html') return join(HERE, 'page.html');
  if (VENDOR_FILES[urlPath]) return VENDOR_FILES[urlPath];
  for (const [prefix, dir] of ROUTES) {
    if (urlPath.startsWith(prefix)) {
      const rel = urlPath.slice(prefix.length);
      if (rel.includes('..')) return null;
      return join(dir, rel);
    }
  }
  // onnxruntime-web fetches its own .wasm/.mjs assets relative to the script URL.
  if (urlPath.startsWith('/vendor/')) {
    const rel = urlPath.slice('/vendor/'.length);
    if (!rel.includes('..')) return vendor(join('onnxruntime-web/dist', rel));
  }
  return null;
}

function startServer() {
  const server = createServer(async (req, res) => {
    const urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    const file = await resolveRequest(urlPath);
    if (!file) {
      res.writeHead(404).end('not found');
      return;
    }
    try {
      const st = await stat(file);
      res.writeHead(200, {
        'content-type': MIME[extname(file)] ?? 'application/octet-stream',
        'content-length': st.size,
        'access-control-allow-origin': '*',
        'cache-control': 'no-store',
      });
      createReadStream(file).pipe(res);
    } catch {
      res.writeHead(404).end(`not found: ${urlPath}`);
    }
  });
  return new Promise((r) => server.listen(0, '127.0.0.1', () => r(server)));
}

// ── assertions ─────────────────────────────────────────────────────────────

const results = [];
function check(name, condition, detail = '') {
  results.push({ name, ok: !!condition, detail });
  const mark = condition ? '  ✓' : '  ✗';
  console.log(`${mark} ${name}${detail ? ` — ${detail}` : ''}`);
}

/** Run one state through onnxruntime-web under Node — the parity reference. */
async function nodeInference(state) {
  const ort = await import(vendor('onnxruntime-web/dist/ort.bundle.min.mjs'));
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(
    await readFile(join(PKG, 'models/ac3-controller.onnx')),
    { executionProviders: ['wasm'] },
  );
  const out = await session.run({
    state: new ort.Tensor('float32', Float32Array.from(state), [1, 6, 8]),
  });
  return { probs: Array.from(out.action_probs.data), value: out.state_value.data[0] };
}

async function main() {
  const server = await startServer();
  const port = server.address().port;
  const base = `http://127.0.0.1:${port}`;
  console.log(`\nserving ${base}\n`);

  const { chromium } = await import('playwright');
  const browser = await chromium.launch({
    // CHROMIUM_PATH lets CI point at a Chromium that Playwright did not install
    // (this repo's container ships one at /opt/pw-browsers).
    executablePath: process.env.CHROMIUM_PATH || undefined,
    args: [
      '--autoplay-policy=no-user-gesture-required',
      '--mute-audio',
      ...(process.env.CHROMIUM_NO_SANDBOX ? ['--no-sandbox'] : []),
    ],
  });
  const page = await browser.newPage();

  const consoleErrors = [];
  page.on('console', (m) => {
    if (m.type() === 'error') consoleErrors.push(m.text());
  });
  page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`));

  try {
    console.log('opening page…');
    await page.goto(base, { waitUntil: 'load', timeout: 60_000 });
    console.log('page loaded');

    console.log('── model + adapter bring-up ──');
    await page.waitForFunction(() => window.__e2e?.ready === true, null, { timeout: 60_000 });
    const boot = await page.evaluate(() => ({
      runtime: window.__e2e.runtime,
      loadMs: window.__e2e.loadMs,
      status: window.__e2e.engineStatus,
      caps: window.__e2e.vhsCapabilities,
    }));
    check('ONNX model loads in the browser', boot.runtime === 'onnxruntime-web', boot.runtime);
    check('engine reaches ready', boot.status === 'ready', boot.status);
    check('model load time is reasonable', boot.loadMs < 10_000, `${boot.loadMs?.toFixed(0)}ms`);
    check(
      'adapter attached to the real VhsHandler',
      boot.caps?.hasPlaylistController === true && boot.caps?.hasRepresentations === true,
      JSON.stringify(boot.caps),
    );
    check(
      'adapter reads per-segment stats from the main segment loader',
      boot.caps?.statsSource === 'main-segment-loader',
      String(boot.caps?.statsSource),
    );

    console.log('\n── playback ──');
    // Let it play far enough for several segment downloads and ABR ticks.
    await page.waitForFunction(
      () => (window.__player?.currentTime() ?? 0) > 6 || window.__e2e.playbackErrors.length > 0,
      null,
      { timeout: 90_000 },
    );
    await page.waitForTimeout(8000);

    const mid = await page.evaluate(() => ({
      currentTime: window.__player.currentTime(),
      decisions: window.__e2e.decisions,
      applies: window.__e2e.applies,
      errors: window.__e2e.errors,
      selected: window.__e2e.selectedPlaylists,
      playbackErrors: window.__e2e.playbackErrors,
      probe: window.__e2e.modelParityProbe,
      counters: {
        selectPlaylistCalls: window.__adapter.counters.selectPlaylistCalls,
        servedFromCache: window.__adapter.counters.servedFromCache,
        delegatedToDefault: window.__adapter.counters.delegatedToDefault,
        decisions: window.__adapter.counters.decisions,
        observations: window.__adapter.counters.observations,
      },
    }));

    check('video actually played', mid.currentTime > 5, `t=${mid.currentTime.toFixed(1)}s`);
    check('no player errors', mid.playbackErrors.length === 0, mid.playbackErrors.join('; '));

    const modelDecisions = mid.decisions.filter((d) => d.source === 'model');
    check(
      'the trained model produced decisions',
      modelDecisions.length >= 2,
      `${modelDecisions.length}/${mid.decisions.length} decisions from the model`,
    );
    check(
      'decisions carry a valid action index',
      modelDecisions.every((d) => d.actionIndex >= 0 && d.actionIndex < 6),
      `actions: ${[...new Set(modelDecisions.map((d) => d.actionIndex))].join(',')}`,
    );
    check(
      'observations carry real measured segment stats',
      modelDecisions.some((d) => d.segmentBytes > 0 && d.downloadSec > 0),
      (() => {
        const d = modelDecisions.find((x) => x.segmentBytes > 0);
        return d ? `${d.segmentBytes}B in ${d.downloadSec.toFixed(3)}s` : 'none';
      })(),
    );

    const inf = modelDecisions.map((d) => d.inferenceMs).sort((a, b) => a - b);
    check(
      'inference latency is playback-safe',
      inf.length > 0 && inf[Math.floor(inf.length * 0.95)] < 50,
      inf.length ? `p50=${inf[Math.floor(inf.length / 2)].toFixed(2)}ms p95=${inf[Math.floor(inf.length * 0.95)].toFixed(2)}ms` : 'n/a',
    );

    check(
      'selectPlaylist is served from the decision cache',
      mid.counters.servedFromCache > 0,
      `${mid.counters.servedFromCache} of ${mid.counters.selectPlaylistCalls} calls served from cache`,
    );

    // The load-bearing property of the cached-decision design: VHS can hammer
    // selectPlaylist (it polls at 4 Hz with bufferBasedABR on) without that
    // triggering a single extra inference.
    const hammer = await page.evaluate(async () => {
      const vhs = window.__player.tech(true).vhs;
      const before = window.__adapter.counters.decisions;
      const calls = window.__adapter.counters.selectPlaylistCalls;
      for (let i = 0; i < 500; i++) vhs.selectPlaylist();
      await new Promise((r) => setTimeout(r, 250));
      return {
        newDecisions: window.__adapter.counters.decisions - before,
        newCalls: window.__adapter.counters.selectPlaylistCalls - calls,
      };
    });
    check(
      '500 synchronous selectPlaylist calls trigger zero inferences',
      hammer.newCalls >= 500 && hammer.newDecisions === 0,
      `${hammer.newCalls} calls -> ${hammer.newDecisions} decisions`,
    );

    check(
      'VHS applied the engine decisions',
      mid.applies.some((a) => a.applied),
      `${mid.applies.filter((a) => a.applied).length} applied`,
    );

    // The player started with bandwidth=500000, so the default ABR would sit on
    // the 364 kbps rendition. If VHS ends up anywhere else, the model moved it.
    const finalRendition = mid.selected.at(-1);
    check(
      'VHS switched rendition under the model (not the default ABR)',
      mid.selected.length > 1 || (finalRendition && finalRendition.bandwidth > 364_000),
      `renditions played: ${mid.selected.map((s) => `${s.id}@${s.bandwidth}`).join(' -> ')}`,
    );

    check(
      'no engine errors during normal operation',
      mid.errors.length === 0,
      mid.errors.map((e) => `${e.reason}: ${e.message}`).join('; '),
    );

    // ── cross-runtime parity: browser vs Node ────────────────────────────
    console.log('\n── browser vs Node numerical parity ──');
    if (mid.probe) {
      const nodeOut = await nodeInference(mid.probe.input);
      let maxD = 0;
      for (let i = 0; i < 6; i++) {
        maxD = Math.max(maxD, Math.abs(nodeOut.probs[i] - mid.probe.probs[i]));
      }
      check(
        'browser inference matches Node inference bit-for-bit',
        maxD === 0,
        `max |Δp| = ${maxD.toExponential(3)}`,
      );
    } else {
      check('browser vs Node parity probe available', false, 'probe missing');
    }

    // ── fallback: break the model mid-playback ───────────────────────────
    console.log('\n── fallback ──');
    const fb = await page.evaluate(async () => {
      const player = window.__player;
      const engine = window.__abr;
      const tech = player.tech(true);

      // Sabotage the live inference session so every forward pass throws. This
      // stands in for the real failure modes: a corrupted model file, an ORT
      // wasm load failure, an OOM inside the runtime.
      Object.defineProperty(engine, 'session', {
        configurable: true,
        writable: true,
        value: {
          async run() {
            throw new Error('injected inference failure');
          },
        },
      });

      const errsBefore = window.__e2e.errors.length;
      const seenBefore = window.__e2e.decisions.length;
      const delegatedBefore = window.__adapter.counters.delegatedToDefault;

      // Seek back and replay so VHS downloads segments again. The test stream is
      // only 24 s and VHS has already buffered all of it, so without this there
      // are no further segment completions to tick on.
      player.currentTime(0);
      await player.play().catch(() => {});
      const before = player.currentTime();

      // Drive the same tick VHS raises on each appended segment, so the failure
      // path is exercised deterministically rather than depending on how much
      // the player happens to re-download.
      for (let i = 0; i < 5; i++) {
        tech.trigger('bandwidthupdate');
        await new Promise((r) => setTimeout(r, 400));
      }
      await new Promise((r) => setTimeout(r, 3000));

      // And confirm the synchronous hot path still answers safely.
      let selectPlaylistThrew = false;
      let selected = null;
      try {
        const p = tech.vhs.selectPlaylist();
        selected = p ? (p.id ?? p.uri) : null;
      } catch {
        selectPlaylistThrew = true;
      }

      return {
        before,
        after: player.currentTime(),
        newErrors: window.__e2e.errors.slice(errsBefore),
        newDecisions: window.__e2e.decisions.slice(seenBefore),
        playbackErrors: window.__e2e.playbackErrors,
        delegated: window.__adapter.counters.delegatedToDefault - delegatedBefore,
        selectPlaylistThrew,
        selected,
      };
    });

    check(
      'playback continues after the model starts failing',
      fb.after > fb.before,
      `t ${fb.before.toFixed(1)}s → ${fb.after.toFixed(1)}s`,
    );
    check(
      'the failure is reported, not swallowed',
      fb.newErrors.some((e) => e.reason === 'inference-error'),
      fb.newErrors.map((e) => e.reason).join(', ') || 'no errors emitted',
    );
    check(
      'decisions fall back to the player default',
      fb.newDecisions.some((d) => d.source === 'player-default'),
      [...new Set(fb.newDecisions.map((d) => d.source))].join(', ') || 'no decisions',
    );
    check(
      'selectPlaylist keeps answering and delegates to the player default',
      !fb.selectPlaylistThrew && fb.delegated > 0,
      `delegated ${fb.delegated} times, returned ${fb.selected}`,
    );
    check('still no player errors', fb.playbackErrors.length === 0, fb.playbackErrors.join('; '));

    const fatalConsole = consoleErrors.filter(
      (t) => !/favicon|Failed to load resource/i.test(t),
    );
    check('no uncaught console errors', fatalConsole.length === 0, fatalConsole.slice(0, 3).join(' | '));
  } finally {
    await browser.close();
    server.close();
  }

  // ── report ────────────────────────────────────────────────────────────
  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  if (failed.length) {
    console.log('\nFAILED:');
    for (const f of failed) console.log(`  - ${f.name}${f.detail ? ` (${f.detail})` : ''}`);
    process.exitCode = 1;
  } else {
    console.log('END-TO-END PASS');
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
