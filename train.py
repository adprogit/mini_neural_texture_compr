"""
Neural Texture Compression — training, quantization, evaluation and export.

Simplified re-implementation of the core idea of:
  Vaidyanathan et al., "Random-Access Neural Compression of Material Textures",
  ACM TOG (SIGGRAPH 2023).

A PBR material (all channels stacked: albedo, normal, roughness, AO, ...) is
represented by two low-resolution latent feature grids + one small MLP shared
by every texel. Decoding one texel = 2 bilinear lookups + a tiny MLP, so it can
run per-pixel in a fragment shader (see web/decode.wgsl).

Usage:
  python train.py --material path/to/material_folder --res 1024 --out web/data
  python train.py --synthetic --res 256 --steps 300 --out /tmp/ntc_test   # smoke test
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def load_image(path):
    """Load an image as float32 HxWxC in [0,1]. Collapses RGB maps whose three
    channels are identical (e.g. roughness saved as RGB) to a single channel."""
    img = Image.open(path)
    if img.mode.startswith("I"):              # 16-bit grayscale
        a = np.asarray(img).astype(np.float32) / 65535.0
    elif img.mode in ("L", "LA", "P") and img.mode != "P":
        a = np.asarray(img.convert("L")).astype(np.float32) / 255.0
    else:
        a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[-1] == 3 and np.allclose(a[..., 0], a[..., 1]) and np.allclose(a[..., 0], a[..., 2]):
        a = a[..., :1]
    return a


MAP_KEYWORDS = ("color", "albedo", "basecolor", "diffuse", "normal", "rough", "ambientocclusion",
                "_ao", "occlusion", "metal", "displacement", "height", "opacity", "specular")


def load_material(folder, res, exclude):
    """Stack every map of a material folder into one C x res x res tensor.
    Only files whose name looks like a PBR map are kept (skips previews, thumbnails...)."""
    files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                   if os.path.splitext(f)[1].lower() in IMG_EXT)
    kept = []
    for f in files:
        name = os.path.basename(f).lower()
        if any(e.lower() in name for e in exclude) or not any(k in name for k in MAP_KEYWORDS):
            print(f"  (skipped) {os.path.basename(f)}")
        else:
            kept.append(f)
    files = kept
    if not files:
        raise SystemExit(f"No image found in {folder}")
    chans, layout, start = [], [], 0
    for f in files:
        a = torch.from_numpy(load_image(f)).permute(2, 0, 1)[None]      # 1xCxHxW
        a = F.interpolate(a, size=(res, res), mode="bilinear", antialias=True, align_corners=False)[0]
        name = os.path.splitext(os.path.basename(f))[0]
        layout.append({"name": name, "start": start, "count": a.shape[0]})
        start += a.shape[0]
        chans.append(a.clamp(0, 1))
        print(f"  {name:40s} {a.shape[0]} channel(s)")
    return torch.cat(chans, 0), layout


def synthetic_material(res):
    """Procedural 8-channel 'material' to test the pipeline without data."""
    y, x = torch.meshgrid(torch.linspace(0, 1, res), torch.linspace(0, 1, res), indexing="ij")
    g = torch.Generator().manual_seed(0)
    noise = F.interpolate(torch.rand(1, 1, res // 16, res // 16, generator=g), size=(res, res),
                          mode="bicubic", align_corners=False)[0, 0].clamp(0, 1)
    albedo = torch.stack([0.5 + 0.4 * torch.sin(12 * x) * noise,
                          0.4 + 0.3 * torch.cos(9 * y),
                          0.3 + 0.2 * noise])
    nx, ny = 0.3 * torch.cos(12 * x), 0.3 * torch.sin(9 * y)
    normal = torch.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, torch.sqrt(1 - nx**2 - ny**2) * 0.5 + 0.5])
    rough = (0.3 + 0.6 * noise)[None]
    ao = (1 - 0.5 * noise**2)[None]
    tex = torch.cat([albedo, normal, rough, ao]).clamp(0, 1)
    layout = [{"name": "albedo", "start": 0, "count": 3}, {"name": "normal", "start": 3, "count": 3},
              {"name": "roughness", "start": 6, "count": 1}, {"name": "ao", "start": 7, "count": 1}]
    return tex, layout


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def quantize(v, bits):
    """Uniform quantization of values in [-1, 1] to 2^bits levels."""
    levels = 2 ** bits - 1
    return torch.round((v.clamp(-1, 1) + 1) * 0.5 * levels) / levels * 2 - 1


class NeuralTexture(nn.Module):
    def __init__(self, channels, res, feats=(8, 8), scales=(4, 8), hidden=64, n_hidden=2):
        super().__init__()
        assert all(f % 4 == 0 for f in feats), "features per grid must be a multiple of 4 (RGBA packing)"
        self.grids = nn.ParameterList(
            nn.Parameter(0.1 * torch.randn(1, f, res // s, res // s)) for f, s in zip(feats, scales))
        dims = [sum(feats)] + [hidden] * n_hidden + [channels]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        # TODO(experiment): the paper also feeds a positional encoding of the
        # position inside the grid cell to the MLP — try adding it.

    def grid_values(self, mode, bits):
        """mode: 'float' | 'noise' (quantization-aware training) | 'quant'."""
        out = []
        for g in self.grids:
            v = g.clamp(-1, 1)
            if mode == "noise":
                step = 2.0 / (2 ** bits - 1)
                v = v + (torch.rand_like(v) - 0.5) * step
            elif mode == "quant":
                v = quantize(v, bits)
            out.append(v)
        return out

    def forward(self, uv, mode="float", bits=8):
        """uv: N x 2 in [0,1] (x right, y down) -> N x C."""
        coords = (uv * 2 - 1).view(1, -1, 1, 2)
        feats = [F.grid_sample(v, coords, mode="bilinear", padding_mode="border", align_corners=False)
                 for v in self.grid_values(mode, bits)]          # same convention as GPU texel centers + clamp
        f = torch.cat(feats, 1)[0, :, :, 0].t()                   # N x sum(feats)
        return self.mlp(f)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


@torch.no_grad()
def decode_full(model, res, mode, bits, device, rows_per_chunk=64):
    xs = (torch.arange(res, device=device) + 0.5) / res
    out = []
    for y0 in range(0, res, rows_per_chunk):
        ys = xs[y0:y0 + rows_per_chunk]
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        uv = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)
        out.append(model(uv, mode, bits).clamp(0, 1).t().reshape(-1, len(ys), res))
    return torch.cat(out, 1)


def bits_per_pixel(model, res, bits):
    grid_bits = sum(g.numel() for g in model.grids) * bits
    mlp_bits = sum(p.numel() for p in model.mlp.parameters()) * 16   # assumes fp16 weights
    return (grid_bits + mlp_bits) / (res * res)


def downsample_baseline(tex, bpp):
    """Naive baseline at the same bit budget: store every channel in 8 bits at
    a lower resolution, upsample bilinearly. TODO(experiment): replace/extend
    with a real BC7 / ASTC encoder (e.g. astcenc, Compressonator)."""
    C, H, W = tex.shape
    scale = math.sqrt(C * 8 / bpp)
    h, w = max(1, round(H / scale)), max(1, round(W / scale))
    low = F.interpolate(tex[None], size=(h, w), mode="area")
    low = torch.round(low * 255) / 255
    up = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)[0]
    return up.clamp(0, 1), (h, w)


# -----------------------------------------------------------------------------
# Export (read by web/main.js)
# -----------------------------------------------------------------------------
def save_png(t, path):
    a = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    Image.fromarray(a[..., 0] if a.shape[-1] == 1 else a).save(path)


def export(model, tex, recon, layout, args, metrics, out):
    os.makedirs(out, exist_ok=True)
    grids_meta = []
    for i, v in enumerate(model.grid_values("quant", args.bits)):
        v = v[0]                                                   # F x h x w in [-1,1]
        u8 = torch.round((v + 1) * 0.5 * 255).clamp(0, 255).to(torch.uint8)
        f, h, w = u8.shape
        packed = u8.view(f // 4, 4, h, w).permute(0, 2, 3, 1).contiguous()   # layers x h x w x RGBA
        packed.cpu().numpy().tofile(os.path.join(out, f"grid{i}.bin"))
        grids_meta.append({"file": f"grid{i}.bin", "width": w, "height": h, "layers": f // 4})

    # MLP weights, per layer: W (out x in, row-major) then bias. float32.
    flat, layers = [], []
    for m in model.mlp:
        if isinstance(m, nn.Linear):
            flat += [m.weight.detach().cpu().reshape(-1), m.bias.detach().cpu()]
            layers.append({"in": m.in_features, "out": m.out_features})
    torch.cat(flat).numpy().astype(np.float32).tofile(os.path.join(out, "mlp.bin"))

    manifest = {"res": args.res, "channels": tex.shape[0], "layout": layout, "bits": args.bits,
                "grids": grids_meta, "mlp": layers, "metrics": metrics}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    for entry in layout:                                          # reference / reconstruction images
        s, c = entry["start"], entry["count"]
        save_png(tex[s:s + c], os.path.join(out, f"orig_{entry['name']}.png"))
        save_png(recon[s:s + c], os.path.join(out, f"recon_{entry['name']}.png"))
    print(f"Exported to {out}/")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--material", help="folder containing the maps of ONE material")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--exclude", nargs="*", default=["preview", "NormalDX", "displacement"],
                   help="skip files whose name contains one of these strings")
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--feats", type=int, nargs=2, default=[8, 8])
    p.add_argument("--scales", type=int, nargs=2, default=[4, 8])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-hidden", type=int, default=2)
    p.add_argument("--bits", type=int, default=4, choices=[2, 4, 8],
                   help="latent bits (2/4/8 map exactly onto 8-bit GPU textures)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=1 << 16)
    p.add_argument("--qat-start", type=float, default=0.6, help="fraction of steps before noise-QAT")
    p.add_argument("--out", default="web/data")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print("Loading material...")
    tex, layout = synthetic_material(args.res) if args.synthetic else load_material(args.material, args.res, args.exclude)
    tex = tex.to(device)
    C = tex.shape[0]

    model = NeuralTexture(C, args.res, tuple(args.feats), tuple(args.scales), args.hidden, args.n_hidden).to(device)
    opt = torch.optim.Adam([{"params": model.grids.parameters(), "lr": 1e-2},
                            {"params": model.mlp.parameters(), "lr": 5e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    flat_tex = tex.reshape(C, -1)

    print(f"Training on {device}: {C} channels, {args.res}^2, {args.steps} steps")
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, args.res * args.res, (args.batch,), device=device)
        uv = torch.stack([(idx % args.res).float(), (idx // args.res).float()], 1).add_(0.5).div_(args.res)
        mode = "noise" if step >= args.qat_start * args.steps else "float"
        loss = F.mse_loss(model(uv, mode, args.bits), flat_tex[:, idx].t())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  loss {loss.item():.6f}  mode {mode}  ({time.time() - t0:.0f}s)")

    # ---- Evaluation
    recon = decode_full(model, args.res, "quant", args.bits, device)
    bpp = bits_per_pixel(model, args.res, args.bits)
    base, base_res = downsample_baseline(tex, bpp)
    metrics = {"bpp": bpp, "uncompressed_bpp": C * 8, "psnr_all": psnr(recon, tex),
               "baseline_downsample_res": list(base_res), "baseline_psnr_all": psnr(base, tex), "per_map": {}}
    print(f"\nBit rate: {bpp:.2f} bpp (uncompressed 8-bit: {C * 8} bpp, ratio x{C * 8 / bpp:.1f})")
    print(f"{'map':40s} {'neural':>8s} {'downsample':>11s}")
    for e in layout:
        s, c = e["start"], e["count"]
        pn, pb = psnr(recon[s:s + c], tex[s:s + c]), psnr(base[s:s + c], tex[s:s + c])
        metrics["per_map"][e["name"]] = {"neural": pn, "downsample": pb}
        print(f"{e['name']:40s} {pn:8.2f} {pb:11.2f}")
    print(f"{'ALL':40s} {metrics['psnr_all']:8.2f} {metrics['baseline_psnr_all']:11.2f}")

    export(model, tex, recon, layout, args, metrics, args.out)


if __name__ == "__main__":
    main()
