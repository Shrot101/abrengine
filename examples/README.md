# Examples

Four runnable examples. No bundler, no build step beyond building the package
itself.

```bash
# from the repository root
npm install
npm run build          # build the package into packages/abrengine/dist
npm run make-stream    # generate a local HLS stream (needs ffmpeg, ~30 s)
npm run example        # serve everything at http://localhost:8080
```

| Example | What it shows |
|---|---|
| [`vanilla-videojs/`](./vanilla-videojs/) | The minimal integration — six lines. Plus a button to switch between the trained controller and Video.js's default at runtime. |
| [`telemetry/`](./telemetry/) | A live inspector: the full 6×8 state tensor with labelled rows and units, the softmax over all six actions as a bar chart, the critic's V(s), and running latency percentiles. |
| [`fallback/`](./fallback/) | Break the model on purpose and watch the player keep going. Failure injection, invalid observations, forced ticks. |
| [`custom-model/`](./custom-model/) | The four ways to supply a controller, including a file picker that swaps in your own `.onnx` live. |

## Using your own stream

Change one line in any example:

```js
player.src({ src: 'https://your-cdn.example/master.m3u8', type: 'application/x-mpegURL' });
```

## About `codec-shim.js`

The examples load a small classic script before video.js. It exists for CI, and
it is a **no-op in Chrome, Edge, Firefox and Safari** — you do not need it in your
own application.

videojs-http-streaming refuses to register its source handler unless the browser
can decode H.264 + AAC in MediaSource:

```js
const supportsNativeMediaSources = () =>
  browserSupportsCodec('avc1.4d400d,mp4a.40.2', true);
if (supportsNativeMediaSources()) {
  videojs.getTech('Html5').registerSourceHandler(VhsSourceHandler, 0);
}
```

That is a statement about the browser, not about your stream. Chromium builds
without proprietary codecs — Playwright's bundled Chromium, most Linux distro
`chromium` packages — fail it, so VHS never loads there even for a VP9/Opus stream
they decode perfectly well.

When (and only when) H.264 is genuinely unavailable, the shim answers `true` for
that one probe string so VHS registers. Nothing else is faked: MediaSource really
appends the segments, the browser really decodes them, and every query about the
codecs actually in your stream is answered by the browser.

To generate a VP9/Opus stream for such a browser:

```bash
npm run make-stream -- ./test-e2e/stream vp9
```

## Import maps

These pages run without a bundler, so a bare `onnxruntime-web` specifier needs an
import map:

```html
<script type="importmap">
  { "imports": { "onnxruntime-web": "/vendor/ort.mjs" } }
</script>
```

A Vite / webpack / Rollup application resolves it from `node_modules` and needs
none of this.
