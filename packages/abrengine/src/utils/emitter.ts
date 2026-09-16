/**
 * A ~40-line typed event emitter.
 *
 * Reasons not to take a dependency: the whole surface needed is on/off/emit
 * with typed payloads, and a listener that throws must not break playback —
 * a behaviour most emitters do not give us for free.
 */

export type Listener<T> = (event: T) => void;

export class Emitter<M> {
  private readonly map = new Map<keyof M, Set<Listener<never>>>();
  private readonly onListenerError: (err: unknown, event: keyof M) => void;

  constructor(onListenerError?: (err: unknown, event: keyof M) => void) {
    this.onListenerError =
      onListenerError ??
      ((err, event) => {
        console.error(`[abrengine] listener for '${String(event)}' threw:`, err);
      });
  }

  on<K extends keyof M>(event: K, listener: Listener<M[K]>): () => void {
    let set = this.map.get(event);
    if (!set) {
      set = new Set();
      this.map.set(event, set);
    }
    set.add(listener as Listener<never>);
    return () => this.off(event, listener);
  }

  once<K extends keyof M>(event: K, listener: Listener<M[K]>): () => void {
    const off = this.on(event, (payload) => {
      off();
      listener(payload);
    });
    return off;
  }

  off<K extends keyof M>(event: K, listener: Listener<M[K]>): void {
    this.map.get(event)?.delete(listener as Listener<never>);
  }

  /** `true` if anything is listening. Used to skip building expensive payloads. */
  has<K extends keyof M>(event: K): boolean {
    const s = this.map.get(event);
    return s !== undefined && s.size > 0;
  }

  emit<K extends keyof M>(event: K, payload: M[K]): void {
    const set = this.map.get(event);
    if (!set || set.size === 0) return;
    // Copy so a listener that unsubscribes mid-emit does not skip a sibling.
    for (const listener of [...set]) {
      try {
        (listener as Listener<M[K]>)(payload);
      } catch (err) {
        this.onListenerError(err, event);
      }
    }
  }

  removeAll(): void {
    this.map.clear();
  }
}
