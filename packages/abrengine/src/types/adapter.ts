/**
 * The contract every player integration implements.
 *
 * Deliberately narrow: an adapter translates *player state -> observation* and
 * *decision -> player action*, and nothing else. No adapter contains ABR logic.
 */

import type { AbrDecision } from './decision.js';
import type { AbrObservation } from './observation.js';

export interface PlayerAbrAdapter {
  /** Human-readable name, e.g. `'videojs'`. Used in telemetry and errors. */
  readonly name: string;

  /**
   * Attach to the player and take over bitrate selection.
   *
   * Resolves once the engine is initialised *and* the player exposes enough
   * state to build observations. Safe to call before the player has a source:
   * the adapter waits for one.
   */
  initialize(): Promise<void>;

  /**
   * Build an observation from the player's current state.
   *
   * Returns `null` when the player is not ready (no source, no ladder, no
   * timing information yet). Never throws.
   */
  getObservation(): AbrObservation | null;

  /**
   * Apply a decision to the player.
   *
   * Returns `true` if the player accepted the change (or was already on that
   * rendition), `false` if it refused — players commonly refuse switches based
   * on their own buffer guards, and that is not an error.
   */
  applyDecision(decision: AbrDecision): boolean;

  /**
   * Restore the player's original ABR behaviour and release all listeners.
   * Idempotent.
   */
  destroy(): void;

  /** `true` between a successful `initialize()` and `destroy()`. */
  readonly active: boolean;
}
