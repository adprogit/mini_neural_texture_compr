// MLP helpers: repack train.py weights as vec4 rows and generate a fully specialized WGSL decoder.
// Shared by the viewer (main.js) and offline tests; no DOM / WebGPU access here.

const ceil4 = (n) => Math.ceil(n / 4);

// mlp.bin layout (train.py): per layer, W (out x in, row-major) then bias, float32.
// Packed layout: per layer, out4*4 rows of in4 vec4 (zero-padded), then out4 bias vec4.
export function packWeights(src, layers) {
  const offsets = [];
  let n = 0;
  for (const l of layers) {
    const in4 = ceil4(l.in), out4 = ceil4(l.out);
    offsets.push({ w: n, b: n + out4 * 4 * in4, in4, out4 });
    n += out4 * 4 * in4 + out4;
  }
  const dst = new Float32Array(n * 4);
  let s = 0;
  layers.forEach((l, k) => {
    const { w, b, in4 } = offsets[k];
    for (let o = 0; o < l.out; o++)
      for (let i = 0; i < l.in; i++) dst[(w + o * in4) * 4 + i] = src[s++];
    for (let o = 0; o < l.out; o++) dst[b * 4 + o] = src[s++];
  });
  return { data: dst, vec4Count: n, offsets };
}

// WGSL for `fn mlp(uv) -> array<vec4f, OUT4>`, loop bounds and offsets baked in as constants.
export function mlpWGSL(layers, offsets, f0Layers, f1Layers) {
  if (offsets[0].in4 !== f0Layers + f1Layers) throw new Error("MLP input size != latent feature count");
  let body = `  var a0 : array<vec4f, ${offsets[0].in4}>;\n`;
  body += `  for (var l = 0u; l < ${f0Layers}u; l++) { a0[l] = textureSampleLevel(grid0, samp, uv, l, 0.0) * 2.0 - 1.0; }\n`;
  body += `  for (var l = 0u; l < ${f1Layers}u; l++) { a0[${f0Layers}u + l] = textureSampleLevel(grid1, samp, uv, l, 0.0) * 2.0 - 1.0; }\n`;
  offsets.forEach(({ w, b, in4, out4 }, k) => {
    const last = k === offsets.length - 1;
    body += `  var a${k + 1} : array<vec4f, ${out4}>;
  for (var o = 0u; o < ${out4}u; o++) {
    var s = W[${b}u + o];
    for (var i = 0u; i < ${in4}u; i++) {
      let x = a${k}[i];
      let r = ${w}u + o * ${4 * in4}u + i;
      s += vec4f(dot(W[r], x), dot(W[r + ${in4}u], x), dot(W[r + ${2 * in4}u], x), dot(W[r + ${3 * in4}u], x));
    }
    a${k + 1}[o] = ${last ? "s" : "max(s, vec4f(0.0))"};
  }\n`;
  });
  return `fn mlp(uv : vec2f) -> array<vec4f, OUT4> {\n${body}  return a${offsets.length};\n}\n`;
}
