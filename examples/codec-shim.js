/**
 * Codec-probe shim — a classic script, loaded BEFORE video.js.
 *
 * It has to be a classic script and it has to come first: videojs-http-streaming
 * decides whether to register its source handler at *module evaluation time*,
 * so a deferred `<script type="module">` runs too late to influence it.
 *
 * Why it exists
 * -------------
 * VHS refuses to register at all unless the browser can decode H.264 + AAC in
 * MediaSource:
 *
 *     const supportsNativeMediaSources = () =>
 *       browserSupportsCodec('avc1.4d400d,mp4a.40.2', true);
 *     if (supportsNativeMediaSources()) {
 *       videojs.getTech('Html5').registerSourceHandler(VhsSourceHandler, 0);
 *     }
 *
 * That is a claim about the browser, not about your stream. Chromium builds
 * without proprietary codecs — Playwright's bundled Chromium, most Linux distro
 * `chromium` packages — fail it, so VHS never loads there, even for a VP9/Opus
 * stream they decode perfectly well.
 *
 * When (and only when) H.264 is genuinely unavailable, this answers `true` for
 * that one probe string. NOTHING ELSE IS FAKED: MediaSource really appends the
 * segments, the browser really decodes them, and every query about the codecs
 * actually present in your stream is answered by the browser itself.
 *
 * In Chrome, Edge, Firefox or Safari this is a no-op, and you do not need it in
 * your own application at all — it is here so these examples run in a
 * codec-stripped CI browser.
 */
(function () {
  var PROBE = 'video/mp4;codecs="avc1.4d400d,mp4a.40.2"';
  var normalised = PROBE.replace(/\s/g, '');

  window.__codecShim = { shimmed: false, reason: 'browser has H.264/AAC MediaSource support' };

  if (typeof MediaSource === 'undefined' || typeof MediaSource.isTypeSupported !== 'function') {
    window.__codecShim = { shimmed: false, reason: 'MediaSource unavailable' };
    return;
  }
  if (MediaSource.isTypeSupported(PROBE)) return;

  var real = MediaSource.isTypeSupported.bind(MediaSource);
  MediaSource.isTypeSupported = function (type) {
    if (typeof type === 'string' && type.replace(/\s/g, '') === normalised) return true;
    return real(type);
  };

  window.__codecShim = {
    shimmed: true,
    reason:
      'no H.264/AAC MediaSource support — answered VHS’s registration probe so it loads; ' +
      'stream codecs are still checked by the browser',
  };
})();
