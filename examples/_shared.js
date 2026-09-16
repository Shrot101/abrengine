/**
 * Shared helpers for the examples.
 *
 * The codec-probe shim lives in `codec-shim.js` instead of here: it must be a
 * classic script that runs before video.js, and a module would be too late.
 */

/** Append-only log pane. */
export function makeLogger(el) {
  return (...parts) => {
    const line = parts
      .map((p) => (typeof p === 'string' ? p : JSON.stringify(p)))
      .join(' ');
    el.textContent += line + '\n';
    el.scrollTop = el.scrollHeight;
  };
}

/** Shared page chrome so the examples look like one family. */
export const STYLE = `
  :root { color-scheme: dark; }
  body { font: 15px/1.55 system-ui, sans-serif; margin: 0; padding: 2rem 1.5rem;
         background: #0f1115; color: #e6e8ee; }
  main { max-width: 60rem; margin: 0 auto; }
  h1 { font-size: 1.35rem; margin: 0 0 .25rem; }
  .sub { color: #98a2b3; margin: 0 0 1.5rem; }
  .video-js { width: 100%; max-width: 720px; aspect-ratio: 16/9; height: auto; }
  pre { background: #14171e; border: 1px solid #232833; border-radius: 6px;
        padding: .75rem; font-size: 12.5px; max-height: 22rem; overflow: auto; }
  code { background: #1b1f27; padding: .1rem .35rem; border-radius: 4px; }
  button { font: inherit; background: #223; color: #e6e8ee; border: 1px solid #39415a;
           border-radius: 6px; padding: .4rem .8rem; cursor: pointer; }
  button:hover { background: #2b3550; }
  .row { display: flex; gap: .5rem; flex-wrap: wrap; margin: 1rem 0; align-items: center; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: .35rem .6rem; border-bottom: 1px solid #232833; }
  th { color: #98a2b3; font-weight: 600; }
  .pill { display: inline-block; padding: .05rem .5rem; border-radius: 999px; font-size: 12px; }
  .pill.model { background: #14361f; color: #86efac; }
  .pill.fallback { background: #3a2a12; color: #fbbf24; }
  .pill.default { background: #2a2f3a; color: #cbd5e1; }
  .note { background: #14171e; border-left: 3px solid #39415a; padding: .7rem 1rem;
          border-radius: 4px; color: #b6becd; font-size: 14px; }
`;
