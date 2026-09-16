/**
 * Unit vocabulary.
 * ================
 *
 * Every numeric field in this package carries its unit in the *type name* and in
 * the *property name suffix*. There is no field anywhere in the public API whose
 * unit has to be guessed.
 *
 * The suffix convention is mechanical:
 *
 * | suffix        | meaning                                    |
 * |---------------|--------------------------------------------|
 * | `Bps`         | **bits** per second                        |
 * | `Kbps`        | kilobits per second (1 Kbps = 1000 bps)    |
 * | `BytesPerSec` | **bytes** per second                       |
 * | `Bytes`       | bytes                                      |
 * | `Sec`         | seconds (floating point)                   |
 * | `Ms`          | milliseconds (floating point)              |
 * | `Norm`        | normalised, dimensionless                  |
 * | `Index`       | zero-based integer index                   |
 * | `Px`          | pixels                                     |
 *
 * These are *branded* types: they compile away to `number`, but the brand makes
 * `bitsPerSecond(x)` and `bytesPerSecond(x)` mutually unassignable, so a
 * bits/bytes mix-up is a compile error rather than a 8x throughput bug.
 *
 * Branding is deliberately opt-in at the boundary: constructors below are the
 * only way to mint a branded value, and `unwrap` is the only way out. Inside the
 * package we do arithmetic on plain numbers.
 */

declare const brand: unique symbol;

/** A `number` tagged with a unit. Erased at runtime. */
export type Unit<TName extends string> = number & { readonly [brand]: TName };

/** Bits per second. E.g. a 1.5 Mbps rendition is `1_500_000`. */
export type BitsPerSecond = Unit<'bits/s'>;
/** Kilobits per second. E.g. a 1.5 Mbps rendition is `1500`. */
export type Kilobits = Unit<'kbit/s'>;
/** Bytes per second. */
export type BytesPerSecond = Unit<'bytes/s'>;
/** A count of bytes. */
export type Bytes = Unit<'bytes'>;
/** Seconds, floating point. */
export type Seconds = Unit<'s'>;
/** Milliseconds, floating point. */
export type Milliseconds = Unit<'ms'>;
/** Dimensionless normalised value (typically but not necessarily in [0, 1]). */
export type Normalised = Unit<'norm'>;
/** Pixels. */
export type Pixels = Unit<'px'>;

export const bitsPerSecond = (n: number): BitsPerSecond => n as BitsPerSecond;
export const kilobits = (n: number): Kilobits => n as Kilobits;
export const bytesPerSecond = (n: number): BytesPerSecond => n as BytesPerSecond;
export const bytes = (n: number): Bytes => n as Bytes;
export const seconds = (n: number): Seconds => n as Seconds;
export const milliseconds = (n: number): Milliseconds => n as Milliseconds;
export const normalised = (n: number): Normalised => n as Normalised;
export const pixels = (n: number): Pixels => n as Pixels;

/** Strip the brand. Identity at runtime. */
export const unwrap = (n: Unit<string>): number => n as number;

// ── Conversions ────────────────────────────────────────────────────────────
// Named so the direction is unambiguous at every call site.

export const kbpsToBps = (k: Kilobits): BitsPerSecond => bitsPerSecond((k as number) * 1000);
export const bpsToKbps = (b: BitsPerSecond): Kilobits => kilobits((b as number) / 1000);
export const bpsToBytesPerSec = (b: BitsPerSecond): BytesPerSecond =>
  bytesPerSecond((b as number) / 8);
export const bytesPerSecToBps = (b: BytesPerSecond): BitsPerSecond =>
  bitsPerSecond((b as number) * 8);
export const msToSec = (m: Milliseconds): Seconds => seconds((m as number) / 1000);
export const secToMs = (s: Seconds): Milliseconds => milliseconds((s as number) * 1000);

/**
 * Throughput of a completed segment download, in bits per second.
 *
 * @param sizeBytes  transferred payload size, bytes
 * @param durationSec wall-clock download duration, seconds (must be > 0)
 */
export const throughputBps = (sizeBytes: Bytes, durationSec: Seconds): BitsPerSecond =>
  bitsPerSecond(((sizeBytes as number) * 8) / Math.max(durationSec as number, 1e-9));
