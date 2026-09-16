import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    videojs: 'src/videojs.ts',
  },
  format: ['esm', 'cjs'],
  dts: true,
  sourcemap: true,
  clean: true,
  splitting: true,
  treeshake: true,
  target: 'es2020',
  outExtension: ({ format }) => ({ js: format === 'cjs' ? '.cjs' : '.js' }),
  // Never bundle the inference runtimes or the player: they are peer deps and
  // pulling them in would multiply the consumer's bundle size (ORT-Web alone is
  // several MB of glue + wasm).
  external: ['onnxruntime-web', 'onnxruntime-node', 'video.js'],
});
