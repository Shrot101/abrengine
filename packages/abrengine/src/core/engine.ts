/**
 * AbrEngine — the framework-independent controller.
 * =================================================
 *
 * Knows nothing about Video.js, HLS, DASH or the DOM. Its whole job:
 *
 *     AbrObservation  ->  state tensor  ->  ONNX policy  ->  AbrDecision
 *
 * Design notes that matter:
 *
 * - **`decide()` is async and never rejects.** Any failure — model not loaded,
 *   runtime missing, inference threw, model returned nonsense — produces a
 *   decision with `source: 'fallback'` and a `reason`, plus an `error` event.
 *   A player integration can therefore call it without a try/catch and without
 *   ever risking playback.
 * - **State is stateful.** The model consumes an 8-step history, so the engine
 *   accumulates observations across calls. `reset()` clears it, and adapters
 *   call that on seek/source-change.
 * - **One decision per segment.** `decide()` should be driven by
 *   segment-completion events. `minDecisionIntervalMs` guards against players
 *   that poll far faster than they download.
 */

import type {
  AbrConfig,
  AbrFallbackFn,
  AbrInferenceSession,
  AbrModelSemantics,
} from '../types/config.js';
import type { AbrDecision, AbrFallbackReason, AbrModelOutput } from '../types/decision.js';
import type { AbrObservation } from '../types/observation.js';
import type { AbrDecisionEvent, AbrEventMap } from '../types/telemetry.js';
import { enabledLadder } from '../types/observation.js';
import { bitsPerSecond, bytes, milliseconds, seconds } from '../types/units.js';

import {
  A_DIM,
  BUFFER_NORM_SEC,
  CHUNK_NORM_BYTES,
  SEGMENT_DURATION_SEC,
  THROUGHPUT_NORM,
  TOTAL_CHUNKS_NORM,
  TRAINING_LADDER_KBPS,
} from '../model/manifest.js';
import { createSession, configureRuntime, resolveRuntime } from '../model/runtime.js';
import { resolveModelSource } from '../model/resolve-source.js';

import { StateBuilder } from './state-builder.js';
import {
  actionToRepresentation,
  makeLadderContext,
  nextSegmentBytesByAction,
  representationToAction,
  type LadderContext,
} from './ladder.js';
import { validateModelOutput, validateObservation, type CleanObservation } from './validate.js';
import { compileFallback } from '../fallback/strategies.js';
import { applySafety, resolveSafety, type ResolvedSafety } from './safety.js';
import { Emitter } from '../utils/emitter.js';
import { now, withTimeout } from '../utils/clock.js';
import { AbrConfigError, AbrError } from './errors.js';

interface ResolvedSemantics extends Required<Omit<AbrModelSemantics, 'trainingLadderKbps'>> {
  trainingLadderKbps: readonly number[];
}

function resolveSemantics(s: AbrModelSemantics | undefined): ResolvedSemantics {
  return {
    trainingLadderKbps: s?.trainingLadderKbps ?? TRAINING_LADDER_KBPS,
    ladderMapping: s?.ladderMapping ?? 'nearest-bitrate',
    bufferNormSec: s?.bufferNormSec ?? BUFFER_NORM_SEC,
    chunkNormBytes: s?.chunkNormBytes ?? CHUNK_NORM_BYTES,
    throughputNorm: s?.throughputNorm ?? THROUGHPUT_NORM,
    totalChunksNorm: s?.totalChunksNorm ?? TOTAL_CHUNKS_NORM,
    segmentDurationSec: (s?.segmentDurationSec as number | undefined) ?? SEGMENT_DURATION_SEC,
  } as ResolvedSemantics;
}

export type AbrEngineStatus =
  | 'idle'
  | 'loading'
  | 'ready'
  /** Model loading failed permanently; every decision falls back. */
  | 'failed'
  | 'destroyed';

export class AbrEngine {
  private readonly config: AbrConfig;
  private readonly semantics: ResolvedSemantics;
  private readonly emitter = new Emitter<AbrEventMap>();
  private readonly builder: StateBuilder;
  private readonly fallbackFn: AbrFallbackFn;
  private readonly telemetryEnabled: boolean;
  private readonly includeTensors: boolean;
  private readonly historySize: number;
  private readonly debug: boolean;
  private readonly safety: ResolvedSafety;

  private session: AbrInferenceSession | null = null;
  private runtimeName = 'none';
  private initPromise: Promise<void> | null = null;
  private statusValue: AbrEngineStatus = 'idle';
  private enabledFlag: boolean;
  private modelByteLength = 0;

  private lastDecision: AbrDecision | null = null;
  private lastModelEvalMs = -Infinity;
  /** Action index of the segment most recently downloaded. Feeds row 5. */
  private lastActionIndex = 0;
  private readonly ring: AbrDecisionEvent[] = [];

  constructor(config: AbrConfig = {}) {
    this.config = config;
    this.semantics = resolveSemantics(config.semantics);

    if (this.semantics.trainingLadderKbps.length !== A_DIM) {
      throw new AbrConfigError(
        `semantics.trainingLadderKbps must have exactly ${A_DIM} entries to match the ` +
          `model's action space, got ${this.semantics.trainingLadderKbps.length}`,
      );
    }
    for (const k of [
      'bufferNormSec',
      'chunkNormBytes',
      'throughputNorm',
      'totalChunksNorm',
    ] as const) {
      const v = this.semantics[k];
      if (!Number.isFinite(v) || v === 0) {
        throw new AbrConfigError(`semantics.${k} must be a non-zero finite number, got ${v}`);
      }
    }

    this.builder = new StateBuilder({
      bufferNormSec: this.semantics.bufferNormSec,
      chunkNormBytes: this.semantics.chunkNormBytes,
      throughputNorm: this.semantics.throughputNorm,
      totalChunksNorm: this.semantics.totalChunksNorm,
    });

    this.fallbackFn = compileFallback(config.fallback, {
      throughputSafetyFactor: config.throughputSafetyFactor ?? 0.9,
    });

    const t = config.telemetry;
    const tObj = typeof t === 'object' && t !== null ? t : undefined;
    this.telemetryEnabled = t === true || tObj?.enabled === true;
    this.includeTensors = tObj?.includeTensors === true;
    this.historySize = Math.max(0, tObj?.historySize ?? 0);
    this.debug = config.debug === true;
    this.safety = resolveSafety(config.safety);
    this.enabledFlag = config.startDisabled !== true;
  }

  // ── Lifecycle ────────────────────────────────────────────────────────────

  get status(): AbrEngineStatus {
    return this.statusValue;
  }

  /** `false` while `disable()`d; every decision then falls back with reason `'disabled'`. */
  get enabled(): boolean {
    return this.enabledFlag;
  }

  /** Which runtime loaded the model. `'none'` before initialisation. */
  get runtime(): string {
    return this.runtimeName;
  }

  /**
   * Load the runtime and the model. Idempotent and concurrency-safe: repeated
   * or parallel calls share one load.
   *
   * Does **not** reject on failure — it records `status: 'failed'`, emits an
   * `error` event, and resolves. The engine then serves fallbacks forever.
   * Callers who want the failure can read `status` or listen for `error`.
   */
  async initialize(): Promise<void> {
    if (this.statusValue === 'destroyed') return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  private async doInitialize(): Promise<void> {
    this.statusValue = 'loading';
    const t0 = now();
    try {
      const resolved = await resolveModelSource(this.config.model ?? 'ac3');

      if (resolved.session) {
        this.session = resolved.session;
        this.runtimeName = 'custom';
      } else {
        const rt = await resolveRuntime(this.config.inference);
        configureRuntime(rt, this.config.inference);
        this.runtimeName = rt.name;
        this.modelByteLength = resolved.bytes!.byteLength;
        this.session = await createSession(rt, resolved.bytes!, this.config.inference);
      }

      if (this.config.inference?.warmup !== false) {
        // One dummy pass so kernel compilation does not land on the first real
        // decision, which happens right as playback starts.
        //
        // A warm-up failure is deliberately NOT fatal. The session was created
        // successfully; if it then misbehaves, the per-decision path already
        // handles that with a fallback and an `inference-error`. Treating warm-up
        // as fatal would take the model out of play for a transient fault.
        try {
          await this.session.run(new Float32Array(this.builder.get().length));
        } catch (err) {
          this.fail(
            'inference-error',
            `warm-up inference failed (continuing anyway): ${(err as Error)?.message ?? String(err)}`,
            err,
            true,
          );
        }
      }

      this.statusValue = 'ready';
      this.log('ready', { runtime: this.runtimeName, loadMs: now() - t0 });
      this.emitter.emit('ready', {
        type: 'ready',
        timestamp: Date.now(),
        runtime: this.runtimeName,
        executionProviders: this.config.inference?.executionProviders ?? ['wasm'],
        loadMs: milliseconds(now() - t0),
        modelBytes: bytes(this.modelByteLength),
      });
    } catch (err) {
      this.statusValue = 'failed';
      this.session = null;
      this.fail(
        err instanceof AbrError && err.name === 'AbrRuntimeError'
          ? 'runtime-unavailable'
          : 'model-load-failed',
        `model initialisation failed: ${(err as Error)?.message ?? String(err)}`,
        err,
        false,
      );
    }
  }

  /**
   * Clear the observation history.
   *
   * Call on seek, source change, or any discontinuity: the model's 8-step
   * throughput/download history is meaningless across a seek, and carrying it
   * over is exactly the kind of silent semantic drift that makes an ABR
   * controller behave differently from how it was trained.
   */
  reset(): void {
    this.builder.reset();
    this.lastDecision = null;
    this.lastActionIndex = 0;
    this.lastModelEvalMs = -Infinity;
  }

  /** Stop using the model; every decision falls back. Model stays loaded. */
  disable(): void {
    if (!this.enabledFlag) return;
    this.enabledFlag = false;
    this.emitter.emit('state', { type: 'state', timestamp: Date.now(), enabled: false });
  }

  /** Resume using the model. */
  enable(): void {
    if (this.enabledFlag) return;
    this.enabledFlag = true;
    this.emitter.emit('state', { type: 'state', timestamp: Date.now(), enabled: true });
  }

  /** Release the session and all listeners. The engine is unusable afterwards. */
  async destroy(): Promise<void> {
    this.statusValue = 'destroyed';
    try {
      await this.session?.release?.();
    } catch {
      /* releasing a session must never throw into a teardown path */
    }
    this.session = null;
    this.emitter.removeAll();
    this.ring.length = 0;
  }

  // ── Events ───────────────────────────────────────────────────────────────

  on<K extends keyof AbrEventMap>(
    event: K,
    listener: (payload: AbrEventMap[K]) => void,
  ): () => void {
    return this.emitter.on(event, listener);
  }

  off<K extends keyof AbrEventMap>(
    event: K,
    listener: (payload: AbrEventMap[K]) => void,
  ): void {
    this.emitter.off(event, listener);
  }

  /** Retained decision events, oldest first. Empty unless `telemetry.historySize > 0`. */
  history(): readonly AbrDecisionEvent[] {
    return this.ring;
  }

  /** The most recent decision, or `null`. */
  get currentDecision(): AbrDecision | null {
    return this.lastDecision;
  }

  // ── The decision path ────────────────────────────────────────────────────

  /**
   * Feed an observation into the model's rolling history **without** asking for
   * a decision.
   *
   * `decide()` already does this, so most integrations never need `observe()`.
   * It exists for the case where the player downloads a segment on a path that
   * does not lead to a bitrate decision (a seek-triggered refill, an init
   * segment, a rendition the app forced manually) and you still want the
   * throughput/download history to reflect reality.
   *
   * Returns `true` if the observation was ingested, `false` if it was rejected
   * or was a cold-start observation with no completed segment.
   */
  observe(observation: AbrObservation): boolean {
    const v = validateObservation(observation);
    if (!v.ok) return false;
    const ladder = enabledLadder(v.clean.ladder);
    if (ladder.length === 0) return false;
    if (v.clean.segmentBytes === null || v.clean.downloadSec === null) return false;
    this.ingest(
      v.clean,
      makeLadderContext(
        ladder,
        this.semantics.ladderMapping,
        this.semantics.trainingLadderKbps,
      ),
    );
    return true;
  }

  /**
   * The current model input tensor, `[S_INFO * S_LEN]` row-major.
   * A copy — safe to retain. For debugging and tests.
   */
  snapshotState(): Float32Array {
    return this.builder.get(true);
  }

  /**
   * Produce a decision for one observation.
   *
   * Never rejects. Never throws. Call once per completed segment download.
   */
  async decide(observation: AbrObservation): Promise<AbrDecision> {
    const t0 = now();

    const v = validateObservation(observation);
    if (this.telemetryEnabled && this.emitter.has('observation')) {
      this.emitter.emit('observation', {
        type: 'observation',
        timestamp: Date.now(),
        observation,
        repairs: v.repairs,
      });
    }
    if (!v.ok) {
      return this.fallbackDecision(
        observation,
        v.clean.ladder.length === 0 ? 'empty-ladder' : 'invalid-observation',
        v.problem ?? 'observation failed validation',
        t0,
      );
    }

    const clean = v.clean;
    const ladder = enabledLadder(clean.ladder);
    if (ladder.length === 0) {
      return this.fallbackDecision(observation, 'empty-ladder', 'no enabled renditions', t0);
    }
    const ctx = makeLadderContext(
      ladder,
      this.semantics.ladderMapping,
      this.semantics.trainingLadderKbps,
    );

    // Feed the observation into the history *before* deciding whether to run
    // the model: the history must stay complete even on ticks we skip, or the
    // 8-step window silently becomes a window over a subsample.
    this.ingest(clean, ctx);

    if (!this.enabledFlag) {
      return this.fallbackDecision(observation, 'disabled', 'engine is disabled', t0);
    }
    if (this.statusValue === 'idle') {
      // Kick off loading so the *next* decision can use the model, but do not
      // block this one behind a multi-hundred-millisecond model fetch.
      void this.initialize();
      return this.fallbackDecision(observation, 'not-initialised', 'model still loading', t0);
    }
    if (this.statusValue === 'loading') {
      return this.fallbackDecision(observation, 'not-initialised', 'model still loading', t0);
    }
    if (this.statusValue !== 'ready' || !this.session) {
      return this.fallbackDecision(
        observation,
        'model-load-failed',
        `engine status is '${this.statusValue}'`,
        t0,
      );
    }

    const minGap = this.config.minDecisionIntervalMs ?? 0;
    if (minGap > 0 && t0 - this.lastModelEvalMs < minGap && this.lastDecision) {
      // Rate-limited: reuse the previous decision rather than falling back, so
      // the player is not yanked between the model and a heuristic.
      return this.lastDecision;
    }

    // ── Inference ──────────────────────────────────────────────────────────
    const stateForModel = this.builder.get(this.includeTensors);
    const tInfer = now();
    let probs: Float32Array;
    let stateValue: number;
    try {
      const out = await withTimeout(
        this.session.run(stateForModel),
        this.config.inference?.timeoutMs ?? 250,
        () => new AbrError('inference timed out'),
      );
      probs = out.actionProbs;
      stateValue = out.stateValue;
    } catch (err) {
      const timedOut = (err as Error)?.message === 'inference timed out';
      return this.fallbackDecision(
        observation,
        timedOut ? 'inference-timeout' : 'inference-error',
        `inference failed: ${(err as Error)?.message ?? String(err)}`,
        t0,
        err,
      );
    }
    const inferenceMs = now() - tInfer;
    this.lastModelEvalMs = t0;

    const problem = validateModelOutput(probs, A_DIM);
    if (problem) {
      return this.fallbackDecision(
        observation,
        'invalid-model-output',
        `model output rejected: ${problem}`,
        t0,
      );
    }

    // argmax — matching `src/test.py::policy_abrengine`, which takes the argmax
    // rather than sampling. Deterministic playback is what production wants.
    let actionIndex = 0;
    for (let i = 1; i < A_DIM; i++) {
      if ((probs[i] as number) > (probs[actionIndex] as number)) actionIndex = i;
    }

    const modelRep = actionToRepresentation(ctx, actionIndex);
    if (!modelRep) {
      return this.fallbackDecision(
        observation,
        'invalid-model-output',
        `action ${actionIndex} did not map to any rendition`,
        t0,
      );
    }

    // Optional guardrail. Off by default; see src/core/safety.ts for why it
    // exists and what it costs.
    const measuredTp =
      clean.throughputBpsMeasured ??
      (clean.segmentBytes !== null && clean.downloadSec !== null && clean.downloadSec > 0
        ? (clean.segmentBytes * 8) / clean.downloadSec
        : null);
    const guarded = applySafety(this.safety, {
      ladder,
      chosen: modelRep,
      bufferSec: clean.bufferSec,
      measuredThroughputBps: measuredTp,
    });
    const rep = guarded.representation;
    if (guarded.clamp) {
      this.log('safety guard lowered the model selection', guarded.clamp);
    }

    const model: AbrModelOutput | null = this.telemetryEnabled
      ? { actionProbs: Array.from(probs), stateValue, actionIndex }
      : null;

    const decision: AbrDecision = {
      representationId: rep.id,
      bitrateBps: rep.bitrateBps,
      source: 'model',
      reason: null,
      actionIndex,
      model,
      safetyClamp: guarded.clamp,
      inferenceMs: milliseconds(inferenceMs),
      observedAtMs: clean.timestampMs,
      decidedAtMs: now(),
    };

    this.lastDecision = decision;
    this.emitDecision(decision, observation, clean, t0, stateForModel, probs);
    return decision;
  }

  // ── Internals ────────────────────────────────────────────────────────────

  /** Push one observation into the rolling state tensor. */
  private ingest(clean: CleanObservation, ctx: LadderContext): void {
    // Cold start: no completed segment yet. The research code's episode also
    // begins from the all-zero state, so we simply do not update.
    if (clean.segmentBytes === null || clean.downloadSec === null) return;

    // Which action slot did the completed segment correspond to?
    const repId = clean.lastSegmentRepresentationId ?? clean.currentRepresentationId;
    this.lastActionIndex = representationToAction(ctx, repId);

    const nextDur =
      clean.nextSegmentDurationSec ??
      (clean.segmentDurationSec && clean.segmentDurationSec > 0
        ? clean.segmentDurationSec
        : this.semantics.segmentDurationSec);

    const remaining = this.resolveRemainingSegments(clean, nextDur);

    this.builder.update({
      segmentBytes: clean.segmentBytes,
      downloadSec: clean.downloadSec,
      throughputMbps:
        clean.throughputBpsMeasured !== null ? clean.throughputBpsMeasured / 1e6 : undefined,
      bufferSec: clean.bufferSec,
      remainingSegments: remaining,
      lastActionIndex: this.lastActionIndex,
      nextSegmentBytesByAction: nextSegmentBytesByAction(
        ctx,
        nextDur,
        clean.nextSegmentSizesBytes ?? undefined,
      ),
    });
  }

  /**
   * "Remaining chunks" for row 4.
   *
   * VOD: the adapter's count, or `(duration - currentTime) / segmentDuration`.
   *
   * Live: the concept does not exist. The model was trained on 48-chunk VOD
   * episodes where this input decays from 1 to 0 and effectively encodes
   * "how close to the end am I". Feeding 0 would tell the policy the video is
   * about to end for the whole stream; feeding a huge number is out of
   * distribution. We pin it to `totalChunksNorm`, i.e. a normalised value of
   * 1.0 — "plenty left", which is true for a live edge and is inside the
   * training distribution. This is documented in the README as a known
   * approximation.
   */
  private resolveRemainingSegments(
    clean: CleanObservation,
    segmentDurationSec: number,
  ): number {
    if (clean.remainingSegments !== null) return clean.remainingSegments;
    const d = clean.durationSec;
    if (Number.isFinite(d) && d > 0 && segmentDurationSec > 0) {
      return Math.max(0, (d - clean.currentTimeSec) / segmentDurationSec);
    }
    return this.semantics.totalChunksNorm;
  }

  private fallbackDecision(
    observation: AbrObservation,
    reason: AbrFallbackReason,
    message: string,
    t0: number,
    cause?: unknown,
  ): AbrDecision {
    // Reasons that reflect a genuine malfunction get an error event. `disabled`
    // and `not-initialised` are normal operating states, not failures.
    if (reason !== 'disabled' && reason !== 'not-initialised') {
      this.fail(reason, message, cause, true);
    } else {
      this.log(message, { reason });
    }

    let id: string | null = null;
    try {
      id = this.fallbackFn(observation, reason);
    } catch (err) {
      this.fail(reason, `fallback strategy threw: ${(err as Error)?.message}`, err, true);
      id = null;
    }

    const ladder = enabledLadder(observation?.representations ?? []);
    const rep = id ? ladder.find((r) => r.id === id) : undefined;

    const decision: AbrDecision =
      rep === undefined
        ? {
            // The fallback declined (or named an unknown rendition): hand control
            // back to the player. `representationId` is empty and adapters must
            // treat that as "run your own selector".
            representationId: '',
            bitrateBps: bitsPerSecond(0),
            source: 'player-default',
            reason,
            actionIndex: null,
            model: null,
            safetyClamp: null,
            inferenceMs: milliseconds(0),
            observedAtMs: observation?.timestampMs ?? 0,
            decidedAtMs: now(),
          }
        : {
            representationId: rep.id,
            bitrateBps: rep.bitrateBps,
            source: 'fallback',
            reason,
            actionIndex: null,
            model: null,
            safetyClamp: null,
            inferenceMs: milliseconds(0),
            observedAtMs: observation?.timestampMs ?? 0,
            decidedAtMs: now(),
          };

    this.lastDecision = decision;
    if (observation) {
      const v = validateObservation(observation);
      this.emitDecision(decision, observation, v.clean, t0, null, null);
    }
    return decision;
  }

  private emitDecision(
    decision: AbrDecision,
    observation: AbrObservation,
    clean: CleanObservation,
    t0: number,
    input: Float32Array | null,
    output: Float32Array | null,
  ): void {
    const wantEvent = this.telemetryEnabled && this.emitter.has('decision');
    const wantHistory = this.historySize > 0;
    if (!wantEvent && !wantHistory) return;

    const prev = clean.ladder.find((r) => r.id === clean.currentRepresentationId);
    const event: AbrDecisionEvent = {
      type: 'decision',
      timestamp: Date.now(),
      monotonicMs: now(),
      decision,
      bufferSec: seconds(clean.bufferSec),
      throughputBps:
        clean.segmentBytes !== null && clean.downloadSec !== null && clean.downloadSec > 0
          ? bitsPerSecond((clean.segmentBytes * 8) / clean.downloadSec)
          : null,
      downloadSec: clean.downloadSec !== null ? seconds(clean.downloadSec) : null,
      segmentBytes: clean.segmentBytes !== null ? bytes(clean.segmentBytes) : null,
      previousBitrateBps: prev ? prev.bitrateBps : null,
      selectedBitrateBps: decision.bitrateBps,
      availableBitratesBps: enabledLadder(clean.ladder).map((r) => r.bitrateBps),
      inferenceMs: decision.inferenceMs,
      totalMs: milliseconds(now() - t0),
      ...(this.includeTensors && input ? { modelInput: input } : {}),
      ...(this.includeTensors && output ? { modelOutput: output.slice() } : {}),
    };

    if (wantHistory) {
      this.ring.push(event);
      while (this.ring.length > this.historySize) this.ring.shift();
    }
    if (wantEvent) this.emitter.emit('decision', event);
    void observation;
  }

  private fail(
    reason: AbrFallbackReason,
    message: string,
    error: unknown,
    recoverable: boolean,
  ): void {
    this.log(message, { reason, error });
    this.emitter.emit('error', {
      type: 'error',
      timestamp: Date.now(),
      reason,
      message,
      error,
      recoverable,
    });
  }

  private log(message: string, extra?: unknown): void {
    if (!this.debug) return;
    // eslint-disable-next-line no-console
    console.debug(`[abrengine] ${message}`, extra ?? '');
  }
}
