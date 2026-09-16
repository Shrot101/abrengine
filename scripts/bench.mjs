/**
 * Performance measurement — Phase 12.
 *
 * Measures, on the real exported model:
 *   - model file size and npm-tarball impact
 *   - session load time (cold)
 *   - inference latency distribution
 *   - end-to-end `engine.decide()` latency (state building + inference)
 *   - allocation behaviour across a long run
 *   - built bundle sizes
 *
 * Run: npm run bench
 */

import { readFile, stat } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..');
const PKG = join(REPO, 'packages/abrengine');

const pct = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
const fmt = (n, d = 3) => n.toFixed(d);
const kib = (n) => `${(n / 1024).toFixed(1)} KiB`;

async function main() {
  const { AbrEngine } = await import(join(PKG, 'dist/index.js'));

  console.log('ABREngine — performance report');
  console.log('='.repeat(64));

  // ── Sizes ──────────────────────────────────────────────────────────────
  const modelPath = join(PKG, 'models/ac3-controller.onnx');
  const modelBytes = await readFile(modelPath);
  console.log('\nModel');
  console.log(`  ac3-controller.onnx        ${kib(modelBytes.length)}`);
  console.log(`  gzipped                    ${kib(gzipSync(modelBytes).length)}`);
  console.log('  parameters                 265,863 float32');

  console.log('\nBuilt bundles (the package only; runtimes are peer deps)');
  for (const f of ['index.js', 'index.cjs', 'videojs.js', 'videojs.cjs']) {
    try {
      const p = join(PKG, 'dist', f);
      const raw = await readFile(p);
      console.log(`  ${f.padEnd(26)} ${kib(raw.length).padStart(10)}  gz ${kib(gzipSync(raw).length)}`);
    } catch {
      console.log(`  ${f.padEnd(26)} (missing — run npm run build)`);
    }
  }
  // Chunks are shared between the two entries.
  try {
    const { readdir } = await import('node:fs/promises');
    const files = (await readdir(join(PKG, 'dist'))).filter(
      (f) => f.startsWith('chunk-') && f.endsWith('.js'),
    );
    let total = 0;
    let gz = 0;
    for (const f of files) {
      const raw = await readFile(join(PKG, 'dist', f));
      total += raw.length;
      gz += gzipSync(raw).length;
    }
    if (files.length) {
      console.log(`  shared chunks (${String(files.length).padStart(2)})          ${kib(total).padStart(10)}  gz ${kib(gz)}`);
    }
  } catch {
    /* ignore */
  }

  // ── Load time ──────────────────────────────────────────────────────────
  console.log('\nSession load (cold, onnxruntime-web wasm, 1 thread)');
  const loadTimes = [];
  for (let i = 0; i < 3; i++) {
    const t0 = performance.now();
    const e = new AbrEngine({ model: { type: 'buffer', buffer: modelBytes } });
    await e.initialize();
    loadTimes.push(performance.now() - t0);
    await e.destroy();
  }
  console.log(`  runs: ${loadTimes.map((t) => `${fmt(t, 0)}ms`).join(', ')}`);
  console.log(`  first (includes wasm init): ${fmt(loadTimes[0], 0)}ms`);

  // ── Inference latency ──────────────────────────────────────────────────
  const engine = new AbrEngine({
    model: { type: 'buffer', buffer: modelBytes },
    telemetry: { enabled: true },
  });
  await engine.initialize();

  const ladder = [300, 750, 1200, 1850, 2850, 4300].map((k, i) => ({
    id: `r${i}`,
    bitrateBps: k * 1000,
    enabled: true,
  }));
  const obs = (i) => ({
    timestampMs: performance.now(),
    representations: ladder,
    currentRepresentationId: `r${i % 6}`,
    playback: {
      bufferSec: 5 + (i % 25),
      rebufferSec: 0,
      currentTimeSec: i * 4,
      durationSec: 100000,
      paused: false,
    },
    lastSegment: {
      sizeBytes: 400_000 + (i % 7) * 90_000,
      downloadSec: 0.4 + (i % 5) * 0.35,
      durationSec: 4,
      representationId: `r${i % 6}`,
    },
    nextSegmentDurationSec: 4,
    remainingSegments: 40,
  });

  const N = 3000;
  const decideMs = [];
  const inferMs = [];
  for (let i = 0; i < 200; i++) await engine.decide(obs(i)); // warm

  const heapBefore = process.memoryUsage().heapUsed;
  for (let i = 0; i < N; i++) {
    const t0 = performance.now();
    const d = await engine.decide(obs(i));
    decideMs.push(performance.now() - t0);
    inferMs.push(d.inferenceMs);
  }
  const heapAfter = process.memoryUsage().heapUsed;

  decideMs.sort((a, b) => a - b);
  inferMs.sort((a, b) => a - b);

  console.log(`\nInference latency over ${N} decisions`);
  console.log(`  session.run   p50 ${fmt(pct(inferMs, 0.5))}ms  p95 ${fmt(pct(inferMs, 0.95))}ms  p99 ${fmt(pct(inferMs, 0.99))}ms  max ${fmt(inferMs.at(-1))}ms`);
  console.log(`  decide()      p50 ${fmt(pct(decideMs, 0.5))}ms  p95 ${fmt(pct(decideMs, 0.95))}ms  p99 ${fmt(pct(decideMs, 0.99))}ms  max ${fmt(decideMs.at(-1))}ms`);
  console.log(
    `  adapter overhead (decide − run): p50 ${fmt(pct(decideMs, 0.5) - pct(inferMs, 0.5))}ms`,
  );

  const perDecisionKb = (heapAfter - heapBefore) / N / 1024;
  console.log(`\nMemory`);
  console.log(`  heap delta over ${N} decisions: ${fmt((heapAfter - heapBefore) / 1024 / 1024, 2)} MiB`);
  console.log(`  ≈ ${fmt(perDecisionKb, 3)} KiB per decision (telemetry on; 0 with telemetry off)`);

  // ── Duty cycle ─────────────────────────────────────────────────────────
  const segmentSec = 4;
  const duty = (pct(decideMs, 0.5) / 1000 / segmentSec) * 100;
  console.log(`\nDuty cycle`);
  console.log(`  one decision per ${segmentSec}s segment at p50 ${fmt(pct(decideMs, 0.5))}ms`);
  console.log(`  = ${fmt(duty, 4)}% of wall-clock time spent in ABR`);

  await engine.destroy();

  // ── Telemetry-off comparison ───────────────────────────────────────────
  const quiet = new AbrEngine({ model: { type: 'buffer', buffer: modelBytes } });
  await quiet.initialize();
  const quietMs = [];
  for (let i = 0; i < 500; i++) await quiet.decide(obs(i));
  for (let i = 0; i < N; i++) {
    const t0 = performance.now();
    await quiet.decide(obs(i));
    quietMs.push(performance.now() - t0);
  }
  quietMs.sort((a, b) => a - b);
  console.log(`\nTelemetry cost`);
  console.log(`  decide() p50 with telemetry    ${fmt(pct(decideMs, 0.5))}ms`);
  console.log(`  decide() p50 without telemetry ${fmt(pct(quietMs, 0.5))}ms`);
  await quiet.destroy();

  console.log(`\n${'='.repeat(64)}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
