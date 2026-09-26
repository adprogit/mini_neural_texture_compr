// Display pass: A/B split between the original maps and the cached neural decode, + lighting.

struct Params {
  mode      : u32,   // 0 lit, 1 albedo, 2 normal, 3 roughness, 4 |orig - neural| x8
  albedoIdx : u32,
  normalIdx : u32,
  roughIdx  : u32,
  split     : f32,   // screen x in [0,1]: left = original, right = neural
  zoom      : f32,
  centerX   : f32,
  centerY   : f32,
  lightX    : f32,
  lightY    : f32,
  lightZ    : f32,
  aspect    : f32,
};

@group(0) @binding(0) var samp       : sampler;
@group(0) @binding(1) var decoded    : texture_2d_array<f32>;   // written by decode.wgsl
@group(0) @binding(2) var<uniform> P : Params;
@group(0) @binding(3) var origAlbedo : texture_2d<f32>;
@group(0) @binding(4) var origNormal : texture_2d<f32>;
@group(0) @binding(5) var origRough  : texture_2d<f32>;

struct VSOut { @builtin(position) pos : vec4f, @location(0) screen : vec2f };

@vertex
fn vs(@builtin(vertex_index) i : u32) -> VSOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));        // full-screen triangle
  var o : VSOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.screen = vec2f(p.x, 1.0 - p.y);
  return o;
}

fn channel(px : vec2i, c : u32) -> f32 {
  return textureLoad(decoded, px, c / 4u, 0)[c % 4u];
}

fn neural3(px : vec2i, c : u32) -> vec3f {
  return clamp(vec3f(channel(px, c), channel(px, c + 1u), channel(px, c + 2u)), vec3f(0.0), vec3f(1.0));
}

fn shade(albedo : vec3f, nTex : vec3f, rough : f32) -> vec3f {
  // Minimal lighting to show the normal map: Lambert + Blinn-Phong.
  // TODO(experiment): replace with GGX to really showcase the PBR channels.
  let n = normalize(nTex * 2.0 - 1.0);          // OpenGL-convention normal map
  let l = normalize(vec3f(P.lightX, P.lightY, P.lightZ));
  let v = vec3f(0.0, 0.0, 1.0);
  let h = normalize(l + v);
  let shininess = mix(256.0, 4.0, clamp(rough, 0.0, 1.0));
  let spec = pow(max(dot(n, h), 0.0), shininess) * (1.0 - rough) * 0.5;
  return albedo * (0.08 + max(dot(n, l), 0.0)) + vec3f(spec);
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4f {
  let s = in.screen;
  let uv = vec2f(P.centerX, P.centerY) + (s - 0.5) * vec2f(P.aspect, 1.0) / P.zoom;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return vec4f(0.1, 0.1, 0.12, 1.0); }
  let px = vec2i(in.pos.xy);

  var alb = textureSampleLevel(origAlbedo, samp, uv, 0.0).rgb;
  var nor = textureSampleLevel(origNormal, samp, uv, 0.0).rgb;
  var rou = textureSampleLevel(origRough, samp, uv, 0.0).r;
  let oAlb = alb;
  if (s.x >= P.split) {
    alb = neural3(px, P.albedoIdx);
    nor = neural3(px, P.normalIdx);
    rou = clamp(channel(px, P.roughIdx), 0.0, 1.0);
  }

  var c : vec3f;
  switch (P.mode) {
    case 1u: { c = alb; }
    case 2u: { c = nor; }
    case 3u: { c = vec3f(rou); }
    case 4u: { c = abs(oAlb - neural3(px, P.albedoIdx)) * 8.0; }
    default: { c = shade(alb, nor, rou); }
  }
  if (abs(s.x - P.split) < 0.0015) { c = vec3f(1.0); }       // split line
  return vec4f(c, 1.0);
}
