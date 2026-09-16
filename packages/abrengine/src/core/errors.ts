/** Error types. All extend `Error`, so `instanceof Error` checks keep working. */

export class AbrError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message);
    this.name = new.target.name;
    if (options?.cause !== undefined) {
      (this as { cause?: unknown }).cause = options.cause;
    }
    // Preserve prototype chain when compiled to ES5-ish targets.
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** The inference runtime could not be loaded or misbehaved. */
export class AbrRuntimeError extends AbrError {}

/** The model bytes could not be fetched or parsed. */
export class AbrModelLoadError extends AbrError {}

/** Configuration is internally inconsistent. Thrown from `initialize()` only. */
export class AbrConfigError extends AbrError {}
