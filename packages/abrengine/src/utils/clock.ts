/** Monotonic clock, milliseconds. Falls back to `Date.now` where `performance` is absent. */
export const now = (): number =>
  typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();

/**
 * Race a promise against a deadline.
 *
 * The loser is *not* cancelled — ONNX Runtime has no cancellation — so a timed-out
 * inference keeps running to completion in the background and its result is
 * discarded. That is the correct trade: the alternative is blocking a decision
 * behind a hung session.
 */
export async function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  onTimeout: () => Error,
): Promise<T> {
  if (!Number.isFinite(ms) || ms <= 0) return promise;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(onTimeout()), ms);
        // Do not hold a Node process open on account of this timer.
        (timer as unknown as { unref?: () => void }).unref?.();
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}
