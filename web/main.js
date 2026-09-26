// Neural Texture Compression — WebGPU viewer.
// Loads data/manifest.json + grid*.bin + mlp.bin exported by train.py.
// Frame = decode pass (compute, decode.wgsl) only when the view changes + cheap display pass (view.wgsl).

import { packWeights, mlpWGSL } from "./mlp.js";

const DATA = "data/";
const $ = (id) => document.getElementById(id);

async function fetchBin(name) {
  const r = await fetch(DATA + name);
  if (!r.ok) throw new Error(`Missing ${DATA}${name}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function loadImageTexture(device, name, fallback) {
  // Raw values, no color-space conversion: same data the network was trained on.
  let r = await fetch(DATA + name);
  if (!r.ok) r = await fetch(DATA + fallback);
  const bmp = await createImageBitmap(await r.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const tex = device.createTexture({
    size: [bmp.width, bmp.height], format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: bmp }, { texture: tex }, [bmp.width, bmp.height]);
  return tex;
}

function findMap(layout, keywords, minCount) {
  const e = layout.find((m) => keywords.some((k) => m.name.toLowerCase().includes(k)) && m.count >= minCount);
  return e ?? layout.find((m) => m.count >= minCount) ?? layout[0];
}

async function compile(device, code, label) {
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  for (const m of info.messages) console[m.type === "error" ? "error" : "warn"](`${label} ${m.lineNum}:${m.linePos} ${m.message}`);
  const err = info.messages.find((m) => m.type === "error");
  if (err) throw new Error(`${label} ${err.lineNum}:${err.linePos} ${err.message}`);
  return module;
}

async function main() {
  if (!navigator.gpu) throw new Error("WebGPU not supported in this browser.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter (enable hardware acceleration / check chrome://gpu).");
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (e) => { $("stats").textContent = "GPU error: " + e.error.message; console.error(e.error); });
  device.lost.then((i) => { $("stats").textContent = "GPU device lost: " + i.message; });
  const canvas = $("canvas");
  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  const man = await (await fetch(DATA + "manifest.json")).json();
  const albedoMap = findMap(man.layout, ["albedo", "color", "diff", "base"], 3);
  const normalMap = findMap(man.layout, ["normal", "nrm"], 3);
  const roughMap = findMap(man.layout, ["rough"], 1);
  $("stats").textContent =
    `${man.res}² · ${man.channels} channels · ${man.metrics.bpp.toFixed(2)} bpp ` +
    `(x${(man.metrics.uncompressed_bpp / man.metrics.bpp).toFixed(1)}) · PSNR ${man.metrics.psnr_all.toFixed(2)} dB ` +
    `(downsample baseline ${man.metrics.baseline_psnr_all.toFixed(2)} dB)`;

  // ---- Latent grids -> 2D array textures (one RGBA layer = 4 features)
  const gridTex = [];
  for (const g of man.grids) {
    const data = await fetchBin(g.file);
    const tex = device.createTexture({
      size: [g.width, g.height, g.layers], format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: tex }, data,
      { bytesPerRow: g.width * 4, rowsPerImage: g.height }, [g.width, g.height, g.layers]);
    gridTex.push(tex);
  }

  // ---- MLP weights, repacked as vec4 rows. Uniform buffer when it fits (broadcast reads are cheap).
  const raw = await fetchBin("mlp.bin");
  const packed = packWeights(new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4), man.mlp);
  const useUniform = packed.data.byteLength <= device.limits.maxUniformBufferBindingSize;
  const wBuf = device.createBuffer({
    size: packed.data.byteLength,
    usage: (useUniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(wBuf, 0, packed.data);

  // ---- Reference textures for the A/B split
  const oAlb = await loadImageTexture(device, `orig_${albedoMap.name}.png`, `orig_${man.layout[0].name}.png`);
  const oNor = await loadImageTexture(device, `orig_${normalMap.name}.png`, `orig_${man.layout[0].name}.png`);
  const oRou = await loadImageTexture(device, `orig_${roughMap.name}.png`, `orig_${man.layout[0].name}.png`);

  // ---- Shaders (decoder specialized for this MLP)
  const out4 = packed.offsets.at(-1).out4;
  let decodeCode = await (await fetch("decode.wgsl")).text();
  const consts = {
    OUT4: `${out4}u`, W_SPACE: useUniform ? "uniform" : "storage, read", W_LEN: packed.vec4Count,
    MLP: mlpWGSL(man.mlp, packed.offsets, man.grids[0].layers, man.grids[1].layers),
  };
  for (const [k, v] of Object.entries(consts)) decodeCode = decodeCode.replaceAll(`{{${k}}}`, String(v));
  const decodeModule = await compile(device, decodeCode, "decode.wgsl");
  const viewModule = await compile(device, await (await fetch("view.wgsl")).text(), "view.wgsl");

  const decodePipeline = device.createComputePipeline({ layout: "auto", compute: { module: decodeModule, entryPoint: "cs" } });
  const viewPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: viewModule, entryPoint: "vs" },
    fragment: { module: viewModule, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });
  const viewUBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // Decode target (neural channels at screen resolution) + its bind groups; rebuilt on resize.
  function makeDecodeTarget(w, h) {
    const tex = device.createTexture({
      size: [w, h, out4], format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const uBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const view = tex.createView({ dimension: "2d-array" });
    const decodeBG = device.createBindGroup({
      layout: decodePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: gridTex[0].createView({ dimension: "2d-array" }) },
        { binding: 2, resource: gridTex[1].createView({ dimension: "2d-array" }) },
        { binding: 3, resource: { buffer: wBuf } },
        { binding: 4, resource: { buffer: uBuf } },
        { binding: 5, resource: view },
      ],
    });
    const viewBG = device.createBindGroup({
      layout: viewPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: view },
        { binding: 2, resource: { buffer: viewUBuf } },
        { binding: 3, resource: oAlb.createView() },
        { binding: 4, resource: oNor.createView() },
        { binding: 5, resource: oRou.createView() },
      ],
    });
    return { tex, uBuf, decodeBG, viewBG, w, h };
  }

  function encodeDecode(enc, tgt) {
    const a = new ArrayBuffer(32), f = new Float32Array(a), u = new Uint32Array(a);
    f[0] = st.zoom; f[1] = st.cx; f[2] = st.cy; f[3] = tgt.w / tgt.h; u[4] = tgt.w; u[5] = tgt.h;
    device.queue.writeBuffer(tgt.uBuf, 0, a);
    const pass = enc.beginComputePass();
    pass.setPipeline(decodePipeline); pass.setBindGroup(0, tgt.decodeBG);
    pass.dispatchWorkgroups(Math.ceil(tgt.w / 8), Math.ceil(tgt.h / 8)); pass.end();
  }

  // ---- Interaction state. viewDirty -> re-decode; dirty -> redraw only.
  const st = { zoom: 1, cx: 0.5, cy: 0.5, split: 0.5, t: 0 };
  let viewDirty = true, dirty = true;
  const moved = () => { viewDirty = dirty = true; };
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); st.zoom = Math.min(64, Math.max(1, st.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15))); moved(); }, { passive: false });
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, cx: st.cx, cy: st.cy, split: e.shiftKey }; });
  window.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const r = canvas.getBoundingClientRect();
    if (drag.split) { st.split = (e.clientX - r.left) / r.width; dirty = true; return; }
    st.cx = drag.cx - (e.clientX - drag.x) / r.height / st.zoom;
    st.cy = drag.cy - (e.clientY - drag.y) / r.height / st.zoom;
    moved();
  });
  $("split").addEventListener("input", (e) => { st.split = e.target.value / 100; dirty = true; });
  $("mode").addEventListener("change", () => { dirty = true; });

  function writeViewUniforms(aspect) {
    const a = new ArrayBuffer(48), u = new Uint32Array(a), f = new Float32Array(a);
    u[0] = Number($("mode").value); u[1] = albedoMap.start; u[2] = normalMap.start; u[3] = roughMap.start;
    f[4] = st.split; f[5] = st.zoom; f[6] = st.cx; f[7] = st.cy;
    f[8] = Math.cos(st.t) * 0.6; f[9] = Math.sin(st.t) * 0.6; f[10] = 0.7; f[11] = aspect;
    device.queue.writeBuffer(viewUBuf, 0, a);
  }

  let screen = null;
  function frame() {
    requestAnimationFrame(frame);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (!screen || screen.w !== w || screen.h !== h) {
      canvas.width = w; canvas.height = h;
      screen?.tex.destroy();
      screen = makeDecodeTarget(w, h);
      viewDirty = dirty = true;
    }
    if ($("animate").checked) { st.t += 0.01; dirty = true; }
    if (!dirty) return;

    const enc = device.createCommandEncoder();
    if (viewDirty) encodeDecode(enc, screen);
    writeViewUniforms(w / h);
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(viewPipeline); pass.setBindGroup(0, screen.viewBG); pass.draw(3); pass.end();
    device.queue.submit([enc.finish()]);
    viewDirty = dirty = false;
  }
  requestAnimationFrame(frame);

  // ---- Benchmark: N full neural decodes at 1920x1080 (every pixel runs the MLP).
  // Coarse CPU-side timing; TODO(perf): use timestamp-query when available.
  $("bench").addEventListener("click", async () => {
    const tgt = makeDecodeTarget(1920, 1080), N = 100;
    const run = () => { const enc = device.createCommandEncoder(); encodeDecode(enc, tgt); device.queue.submit([enc.finish()]); };
    run(); await device.queue.onSubmittedWorkDone();   // warm-up
    const t0 = performance.now();
    for (let i = 0; i < N; i++) run();
    await device.queue.onSubmittedWorkDone();
    const ms = (performance.now() - t0) / N;
    tgt.tex.destroy();
    $("benchOut").textContent = `${ms.toFixed(2)} ms / frame (1080p, full neural decode per pixel)`;
  });
}

main().catch((e) => { $("stats").textContent = "Error: " + e.message; console.error(e); });
