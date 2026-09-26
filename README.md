# Neural Texture Compression — PyTorch training + real-time WebGPU decoding

> Re-implementation (simplified) of **Vaidyanathan et al., _Random-Access Neural Compression of Material Textures_, ACM TOG / SIGGRAPH 2023**.
> A full PBR material (albedo, normal, roughness, AO…) is compressed jointly into two quantized latent grids + one tiny MLP, and decoded **per pixel in a fragment shader**.

<!-- TODO: GIF / screenshot of the viewer (split view original | neural) -->
<!-- TODO: link to the live demo on the portfolio -->

## Why this matters
<!-- TODO (2–3 lines, your own words): block compression (BC7/ASTC) compresses each texture independently
and at a fixed rate; the neural approach exploits correlations between channels and mip levels,
reaching higher quality at the same bit rate. Mention where the industry is (NVIDIA RTX Neural
Texture Compression SDK — check current status and link it). -->

## Method
- **Input**: one material, `C` channels stacked (e.g. albedo 3 + normal 3 + roughness 1 + AO 1 = 8).
- **Latents**: two feature grids at 1/4 and 1/8 of the texture resolution, 8 features each,
  bilinearly sampled (texel-center convention, identical on CPU and GPU).
- **Decoder**: MLP `16 → 64 → 64 → C` with ReLU, shared by every texel.
- **Quantization-aware training**: float training, then uniform noise of one quantization step
  injected into the grids (last 40 % of steps), final hard quantization to `b` bits (2/4/8).
- **Bit rate**: `(grid values × b + MLP params × 16) / (H × W)` bits per pixel.
- **GPU decode** (`web/decode.wgsl`, compute pass): 2 × `F/4` array-texture lookups → MLP evaluated
  per screen pixel, weights repacked as `vec4` rows in a uniform buffer, loops specialized per MLP
  (`web/mlp.js`). The decode is cached and only re-run when zoom / pan / size change; lighting and
  the A/B split (`web/view.wgsl`) read the cache.
- **Correctness check**: GPU decode reproduces the PyTorch reconstruction exactly
  (max abs difference 0/255 on the test material).

<!-- TODO: small diagram: uv -> [grid 1/4] + [grid 1/8] -> concat -> MLP -> C channels -->

## Results
<!-- TODO: fill with YOUR measurements. Material: name + source (ambientCG / Poly Haven, CC0). -->

| Setting | bpp | Ratio vs 8-bit | PSNR all (dB) | PSNR albedo | PSNR normal | PSNR roughness |
|---|---|---|---|---|---|---|
| Neural, 4-bit latents | | | | | | |
| Neural, 2-bit latents | | | | | | |
| Downsample baseline (same bpp) | | | | | | |
| BC7 (TODO, if encoder available) | | | | | | |

<!-- TODO: rate–distortion plot (PSNR vs bpp) for several --bits / --feats / --scales -->

Decode cost: `__ ms / frame` at 1080p on `<GPU>` (viewer "Benchmark" button, coarse CPU-side timing).

## Differences from the paper (honest scope)
- No mip-mapping: the paper trains one latent pyramid for all mip levels — here level 0 only.
- No positional encoding of the in-cell position (paper uses one).
- MLP weights stored as fp32 in the demo (fp16 assumed in the bit-rate count); no tensor-core / cooperative-vector inference.
- Latents stored in 8-bit textures (values lie exactly on the 2/4-bit grid; real bit-packing not implemented).
- Simple Blinn-Phong shading in the viewer instead of full GGX.

## Future work
<!-- TODO: mip levels, filtering (stochastic / anisotropic), f16 packed weights, compare to BC7/ASTC,
comparison with follow-up work (cite 1–3 recent papers you actually read). -->

## Run it
```bash
pip install -r requirements.txt
# one material folder (e.g. ambientCG 2K PNG): keep ONE normal map (GL convention)
python train.py --material path/to/Material_2K --res 1024 --bits 4 --out web/data
python -m http.server -d web 8000        # then open http://localhost:8000 (Chrome/Edge, WebGPU)
```
Smoke test without data: `python train.py --synthetic --res 256 --steps 300 --out web/data`.

## Reference
K. Vaidyanathan, M. Salvi, B. Wronski, T. Akenine-Möller, P. Ebelin, A. Lefohn.
*Random-Access Neural Compression of Material Textures.* ACM Transactions on Graphics 42(4), SIGGRAPH 2023.
