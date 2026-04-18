#!/usr/bin/env python3
"""
Cosmic Wallpaper Generator v2
Each style uses a purpose-built algorithm — not just layered Perlin noise.

Requirements:
    pip install Pillow numpy scipy
"""

import argparse, random, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# ── Palettes ─────────────────────────────────────────────────────────────────

PALETTES = {
    "nebula":  [(5,0,25),(40,0,80),(100,0,160),(180,20,200),(255,60,180),(255,140,60),(255,220,40),(60,220,255),(0,100,200)],
    "lava":    [(10,0,0),(80,0,0),(180,20,0),(255,60,0),(255,140,0),(255,220,40),(255,255,180),(120,0,60)],
    "tiedye":  [(255,0,100),(255,100,0),(255,220,0),(0,200,80),(0,150,255),(100,0,255),(255,0,200),(0,255,200)],
    "aurora":  [(0,5,20),(0,40,60),(0,120,100),(0,220,160),(100,255,200),(180,255,230),(60,100,200),(160,80,255)],
    "ocean":   [(0,5,30),(0,30,80),(0,80,140),(0,150,200),(0,220,240),(100,240,255),(200,250,255),(20,60,120)],
    "candy":   [(255,180,220),(255,120,180),(255,80,140),(200,100,255),(140,160,255),(100,220,255),(160,255,200),(255,220,160)],
    "forest":  [(5,15,5),(20,50,10),(40,100,20),(80,160,40),(140,200,80),(200,230,140),(60,120,60),(20,80,40)],
    "void":    [(0,0,0),(10,0,20),(30,0,60),(60,0,100),(100,0,140),(160,0,200),(200,100,255),(255,200,255)],
}

# ── Shared helpers ────────────────────────────────────────────────────────────

def rgb_to_oklab(rgb):
    """Convert (R,G,B) 0-255 to Oklab (L, a, b)."""
    r, g, b = [x / 255.0 for x in rgb]
    r = r**2.2 if r > 0.04045 else r / 12.92
    g = g**2.2 if g > 0.04045 else g / 12.92
    b = b**2.2 if b > 0.04045 else b / 12.92
    X = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
    Y = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
    Z = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
    l = X**(1/3); m = Y**(1/3); s = Z**(1/3)
    return (
        0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
        1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
        0.0259040371*l + 0.7827717662*m - 0.8086757660*s,
    )

def oklab_to_rgb(lab):
    """Convert Oklab (L, a, b) back to (R,G,B) 0-255, clamped."""
    L, a, b = lab
    l_ = L + 0.3963377774*a + 0.2158037573*b
    m_ = L - 0.1055613458*a - 0.0638541728*b
    s_ = L - 0.0894841775*a - 1.2914855480*b
    l, m, s = l_**3, m_**3, s_**3
    r =  4.0767416621*l - 3.3077115913*m + 0.2309699292*s
    g = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s
    b_ = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s
    def lin2srgb(x):
        x = max(0.0, min(1.0, x))
        return 1.055 * x**(1/2.2) - 0.055 if x > 0.0031308 else 12.92 * x
    return tuple(int(round(lin2srgb(c) * 255)) for c in (r, g, b_))

def rgb_to_oklch(rgb):
    L, a, b = rgb_to_oklab(rgb)
    return (L, np.sqrt(a**2 + b**2), np.arctan2(b, a))

def oklch_to_rgb(lch):
    L, C, H = lch
    return oklab_to_rgb((L, C * np.cos(H), C * np.sin(H)))

def build_lut(colors, n=2048):
    """
    Perceptually-uniform LUT in Oklab space with arc-length stop spacing.
    Each color occupies LUT space proportional to its perceptual distance
    from its neighbours, so no single color dominates the ramp.
    """
    labs = [rgb_to_oklab(c) for c in colors]
    dists = [0.0]
    for i in range(1, len(labs)):
        dL = labs[i][0] - labs[i-1][0]
        da = labs[i][1] - labs[i-1][1]
        db = labs[i][2] - labs[i-1][2]
        dists.append(dists[-1] + np.sqrt(dL**2 + da**2 + db**2))
    total = dists[-1]
    if total < 1e-9:
        return np.tile(oklab_to_rgb(labs[0]), (n, 1)).astype(np.uint8)
    stops = np.array(dists) / total
    t = np.linspace(0, 1, n)
    L_vals = np.interp(t, stops, [lab[0] for lab in labs])
    a_vals = np.interp(t, stops, [lab[1] for lab in labs])
    b_vals = np.interp(t, stops, [lab[2] for lab in labs])
    rgb_lut = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        rgb_lut[i] = oklab_to_rgb((L_vals[i], a_vals[i], b_vals[i]))
    return rgb_lut

def equalize_field(field, strength=1.0, n_colors=None):
    """Histogram-equalize a [0,1] field for even palette coverage."""
    flat = field.ravel()
    sorted_vals = np.sort(flat)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    equalized = np.interp(flat, sorted_vals, cdf).reshape(field.shape)
    if n_colors is not None and n_colors > 1:
        out = equalized.copy()
        band = 1.0 / n_colors
        for k in range(n_colors):
            lo, hi = k * band, (k + 1) * band
            mask = (equalized >= lo) & (equalized < hi if k < n_colors-1 else equalized <= hi)
            if mask.sum() == 0:
                continue
            vals = equalized[mask]
            ranks = np.argsort(np.argsort(vals))
            out[mask] = lo + (ranks / max(len(ranks) - 1, 1)) * band
        equalized = out
    return field * (1.0 - strength) + equalized * strength

def apply_lut(field, lut, equalize=0.82, n_colors=None):
    """Apply LUT with optional histogram equalization."""
    if equalize > 0:
        field = equalize_field(np.clip(field, 0, 1), strength=equalize, n_colors=n_colors)
    idx = (np.clip(field, 0, 1) * (len(lut)-1)).astype(np.int32)
    return lut[idx]

def fbm(shape, octaves=7, persistence=0.5, scale=1.0, seed=None):
    rng = np.random.default_rng(seed)
    h, w = shape
    field = np.zeros((h, w))
    amp, freq, total = 1.0, 1.0, 0.0
    for _ in range(octaves):
        sigma = max(0.5, min(h, w) / (8.0 * freq * scale))
        layer = gaussian_filter(rng.random((h, w)), sigma=sigma)
        field += amp * layer
        total += amp
        amp *= persistence
        freq *= 2.0
    field /= total
    return (field - field.min()) / (field.max() - field.min() + 1e-9)

def vignette(arr, strength=0.5):
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w] / np.array([h, w], dtype=np.float64)[:, None, None] * 2 - 1
    r = np.sqrt(xx**2 + yy**2)
    mask = np.clip(1.0 - (r * strength) ** 2, 0, 1)
    return (arr * mask[:, :, None]).clip(0, 255).astype(np.uint8)

def color_grade(arr, contrast=1.1, saturation=1.2):
    img = arr.astype(np.float64)
    img = (img - 127.5) * contrast + 127.5
    lum = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    img = lum[:,:,None] + saturation * (img - lum[:,:,None])
    return img.clip(0, 255).astype(np.uint8)

# ── NEBULA ────────────────────────────────────────────────────────────────────

def _nebula_palette_weights(colors):
    """
    Compute per-color LUT weights for nebula. Near-white and near-black colors
    are visually dominant in the bloom step — white blows out and pure black is
    invisible — so we down-weight them in the LUT so chromatic colors get more
    image real-estate.

    Returns a list of floats (one per color, summing to len(colors)) used to
    warp the LUT stop positions. Weight < 1 = compressed band, > 1 = expanded.
    Also returns (has_achromatic, mean_lightness) for caller to adapt bloom.
    """
    labs = [rgb_to_oklab(c) for c in colors]
    weights = []
    for L, a, b in labs:
        chroma = np.sqrt(a**2 + b**2)
        # Proximity to white (L~1, low chroma) or black (L~0)
        near_white = max(0.0, (L - 0.80) / 0.20) * max(0.0, 1.0 - chroma / 0.08)
        near_black = max(0.0, (0.25 - L) / 0.25)
        achromatic_penalty = max(near_white, near_black)
        weights.append(max(0.25, 1.0 - 0.75 * achromatic_penalty))
    mean_L = np.mean([lab[0] for lab in labs])
    has_achromatic = any(
        (lab[0] > 0.80 and np.sqrt(lab[1]**2+lab[2]**2) < 0.08) or lab[0] < 0.25
        for lab in labs
    )
    return weights, has_achromatic, mean_L

def build_lut_weighted(colors, weights, n=2048):
    """
    Like build_lut but each color's LUT band is scaled by its weight.
    Low-weight colors (near white/black) occupy less of the LUT so chromatic
    colors dominate the visible output.
    """
    labs = [rgb_to_oklab(c) for c in colors]
    # Raw perceptual arc-length between consecutive stops
    raw_dists = [0.0]
    for i in range(1, len(labs)):
        dL = labs[i][0]-labs[i-1][0]; da = labs[i][1]-labs[i-1][1]; db = labs[i][2]-labs[i-1][2]
        raw_dists.append(np.sqrt(dL**2+da**2+db**2))
    # Scale each segment by the average weight of its two endpoints
    seg_weights = [1.0]  # dummy for index 0
    for i in range(1, len(labs)):
        seg_weights.append((weights[i-1] + weights[i]) / 2.0)
    weighted = [raw_dists[i] * seg_weights[i] for i in range(len(raw_dists))]
    cumulative = np.cumsum(weighted)
    total = cumulative[-1]
    if total < 1e-9:
        return np.tile(oklab_to_rgb(labs[0]), (n, 1)).astype(np.uint8)
    stops = cumulative / total
    stops[0] = 0.0
    t = np.linspace(0, 1, n)
    L_vals = np.interp(t, stops, [lab[0] for lab in labs])
    a_vals = np.interp(t, stops, [lab[1] for lab in labs])
    b_vals = np.interp(t, stops, [lab[2] for lab in labs])
    rgb_lut = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        rgb_lut[i] = oklab_to_rgb((L_vals[i], a_vals[i], b_vals[i]))
    return rgb_lut

def generate_nebula(shape, colors, seed):
    rng = np.random.default_rng(seed)
    h, w = shape

    # 1. Dark void background
    canvas = np.zeros((h, w, 3), dtype=np.float64)

    # 2. Place N glowing nebula clouds (Gaussian blobs, oriented ellipses)
    n_clouds = rng.integers(4, 9)
    cloud_field = np.zeros((h, w), dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    for _ in range(n_clouds):
        cx = rng.uniform(0.1, 0.9) * w
        cy = rng.uniform(0.1, 0.9) * h
        rx = rng.uniform(0.08, 0.35) * w
        ry = rng.uniform(0.08, 0.30) * h
        angle = rng.uniform(0, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx = (xx - cx) * cos_a + (yy - cy) * sin_a
        dy = -(xx - cx) * sin_a + (yy - cy) * cos_a
        blob = np.exp(-0.5 * ((dx/rx)**2 + (dy/ry)**2))
        # Modulate blob with fine noise for wispy edges
        noise = fbm((h, w), octaves=5, scale=2.0, seed=rng.integers(0, 2**31))
        blob = blob * (0.6 + 0.4 * noise)
        cloud_field += blob * rng.uniform(0.5, 1.0)

    cloud_field = np.clip(cloud_field, 0, None)
    cloud_field /= cloud_field.max() + 1e-9

    # 3. Dust lanes: dark fractal noise that punches holes in clouds
    dust = fbm((h, w), octaves=8, scale=0.7, seed=rng.integers(0, 2**31))
    dust = np.where(dust > 0.55, (dust - 0.55) / 0.45, 0.0)  # only dark the dense regions
    cloud_field = cloud_field * (1.0 - 0.7 * dust)
    cloud_field = np.clip(cloud_field, 0, 1)

    # 4. Multi-color nebula: map different cloud density ranges to different colors.
    # Down-weight near-white and near-black colors so chromatic hues dominate.
    weights, has_achromatic, mean_L = _nebula_palette_weights(colors)
    lut_a = build_lut_weighted(colors, weights, 2048)
    lut_b = build_lut_weighted(list(reversed(colors)), list(reversed(weights)), 2048)
    zone = fbm((h, w), octaves=4, scale=0.5, seed=rng.integers(0, 2**31))
    # Equalize cloud_field so dark/bright palette entries both appear
    cloud_eq = equalize_field(cloud_field, strength=0.80)
    img_a = apply_lut(cloud_eq, lut_a, equalize=0, n_colors=len(colors)).astype(np.float64)
    img_b = apply_lut(cloud_eq, lut_b, equalize=0, n_colors=len(colors)).astype(np.float64)
    canvas = img_a * zone[:,:,None] + img_b * (1-zone[:,:,None])

    # 5. Emission glow: blur bright regions and add back (bloom effect).
    # Reduce bloom strength for high-luminance or achromatic palettes to avoid blowout.
    bloom_scale = 0.5 if has_achromatic else (0.7 if mean_L > 0.70 else 1.0)
    brightness = cloud_field ** 1.5
    for sigma, strength in [(8, 0.4), (20, 0.25), (50, 0.15)]:
        glow_r = gaussian_filter(canvas[:,:,0] * brightness, sigma=sigma)
        glow_g = gaussian_filter(canvas[:,:,1] * brightness, sigma=sigma)
        glow_b = gaussian_filter(canvas[:,:,2] * brightness, sigma=sigma)
        glow = np.stack([glow_r, glow_g, glow_b], axis=2)
        canvas = canvas + (strength * bloom_scale) * glow

    # 6. Stars — clustered near bright nebula + random field stars
    # Paint point sources onto sparse arrays, then Gaussian-blur for smooth
    # circular glow. No pixel-neighbour loops = no boxy square artefacts.
    prob = cloud_field / (cloud_field.sum() + 1e-9)
    n_cluster_stars = int(h * w * 0.0003)
    indices = rng.choice(h * w, size=n_cluster_stars, p=prob.ravel())
    sy_c, sx_c = np.unravel_index(indices, (h, w))
    n_field_stars = int(h * w * 0.0005)
    sy_f = rng.integers(0, h, n_field_stars)
    sx_f = rng.integers(0, w, n_field_stars)
    sy = np.concatenate([sy_c, sy_f]).astype(np.int32)
    sx = np.concatenate([sx_c, sx_f]).astype(np.int32)

    star_brightness = rng.uniform(0.5, 1.0, len(sy))
    star_colors_arr = np.array(
        [colors[i] for i in rng.integers(0, len(colors), len(sy))],
        dtype=np.float64) / 255.0
    star_sizes = rng.choice([0, 0, 0, 1, 1, 2, 3], len(sy))

    # Separate buffers: point cores and spike lines
    dots   = np.zeros((h, w, 3), dtype=np.float64)
    spikes = np.zeros((h, w, 3), dtype=np.float64)

    for i in range(len(sy)):
        r, c, b, s = int(sy[i]), int(sx[i]), float(star_brightness[i]), int(star_sizes[i])
        col = star_colors_arr[i] * b
        dots[r, c] += col
        # Spikes only for large stars; drawn as 1-px lines then blurred smooth
        if s >= 2:
            spike_len = 6 + s * 10
            for d in range(1, spike_len + 1):
                fade = np.exp(-d * 0.18) * b * 0.7
                for dr, dc in [(0, d), (0, -d), (d, 0), (-d, 0)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        spikes[rr, cc] += col * fade

    # Blur dots: tight core + wide diffuse halo — produces smooth circular star
    core_sigma  = max(0.6, min(w, h) * 0.0008)
    wide_sigma  = core_sigma * 5
    spike_sigma = max(0.4, min(w, h) * 0.0004)   # softens spike edges only slightly
    star_field = np.zeros((h, w, 3), dtype=np.float64)
    for ch in range(3):
        star_field[:, :, ch] = (
            gaussian_filter(dots[:, :, ch],   sigma=core_sigma)  * 1.0 +
            gaussian_filter(dots[:, :, ch],   sigma=wide_sigma)  * 0.4 +
            gaussian_filter(spikes[:, :, ch], sigma=spike_sigma) * 0.8
        )

    canvas = canvas + star_field * 255
    return np.clip(canvas, 0, 255).astype(np.uint8)

# ── LAVA LAMP ─────────────────────────────────────────────────────────────────

def generate_lava(shape, colors, seed):
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    n_blobs = rng.integers(6, 14)
    potential = np.zeros((h, w), dtype=np.float64)
    for _ in range(n_blobs):
        cx = rng.uniform(0.05, 0.95) * w
        cy = rng.uniform(0.05, 0.95) * h
        r = rng.choice([rng.uniform(0.04, 0.10), rng.uniform(0.10, 0.22), rng.uniform(0.22, 0.38)],
                       p=[0.4, 0.4, 0.2]) * min(w, h)
        strength = rng.uniform(0.5, 1.5)
        dist2 = (xx - cx)**2 + (yy - cy)**2
        potential += strength * r**2 / (dist2 + r**2 * 0.25)
    potential = np.clip(potential, 0, None)
    potential /= potential.max() + 1e-9
    shaped = 1.0 / (1.0 + np.exp(-18.0 * (potential - 0.28)))
    shaped = (shaped - shaped.min()) / (shaped.max() - shaped.min() + 1e-9)
    noise = fbm((h, w), octaves=5, scale=0.5, seed=rng.integers(0, 2**31))
    shaped = np.clip(shaped + 0.08 * (noise - 0.5), 0, 1)
    shaped = (shaped - shaped.min()) / (shaped.max() - shaped.min() + 1e-9)
    lut = build_lut(colors, 2048)
    img = apply_lut(shaped, lut, equalize=0.6, n_colors=len(colors)).astype(np.float64)

    # Hot bright core: where potential is very high, push toward white/yellow
    core_mask = np.clip((shaped - 0.75) * 4, 0, 1)[:,:,None]
    hot_color = np.array(colors[-2], dtype=np.float64)  # near-end palette color
    img = img * (1 - core_mask) + (img * 0.5 + hot_color * 0.5 + 80) * core_mask

    # Bloom: blur bright regions
    bright = shaped ** 2
    for sigma, strength in [(6, 0.5), (18, 0.3), (45, 0.15)]:
        for c in range(3):
            img[:,:,c] += strength * gaussian_filter(img[:,:,c] * bright, sigma=sigma)

    # Background: very dark, slight color tint from darkest palette color
    bg_color = np.array(colors[0], dtype=np.float64)
    bg_mask = np.clip(1 - shaped * 3, 0, 1)[:,:,None]
    img = img * (1 - bg_mask) + bg_color[None,None,:] * bg_mask * 0.5

    return np.clip(img, 0, 255).astype(np.uint8)

# ── COSMIC ────────────────────────────────────────────────────────────────────

def generate_cosmic(shape, colors, seed):
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx = w * rng.uniform(0.35, 0.65)
    cy = h * rng.uniform(0.35, 0.65)
    dx = (xx - cx) / w
    dy = (yy - cy) / h
    r = np.sqrt(dx**2 + dy**2) + 1e-9

    # ── Layer 1: Star field (vectorised — no pixel loops) ──
    star_bg = np.zeros((h, w, 3), dtype=np.float64)
    n_stars = int(h * w * 0.0012)
    sy = rng.integers(0, h, n_stars)
    sx = rng.integers(0, w, n_stars)
    sb = rng.uniform(60, 255, n_stars)
    scol = np.array([colors[i] for i in rng.integers(0, len(colors), n_stars)], dtype=np.float64) / 255.0
    dots_star = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(n_stars):
        dots_star[int(sy[i]), int(sx[i])] += scol[i] * (sb[i] / 255.0)
    star_sigma = max(0.5, min(w, h) * 0.0006)
    for ch in range(3):
        star_bg[:,:,ch] = (gaussian_filter(dots_star[:,:,ch], sigma=star_sigma) * 1.0 +
                           gaussian_filter(dots_star[:,:,ch], sigma=star_sigma*4) * 0.3)

    # ── Layer 2: Nebula gas ──
    # gas_img is 0-255 RGB; attenuate by density so it's dark away from center,
    # then scale to 0-1 for compositing
    gas_noise = fbm((h, w), octaves=6, scale=0.8, seed=rng.integers(0, 2**31))
    gas_density = np.clip(gas_noise * np.exp(-r * 1.5) + gas_noise * 0.2, 0, 1)
    lut_gas = build_lut(colors, 2048)
    gas_img = apply_lut(gas_density, lut_gas, equalize=0.75, n_colors=len(colors)).astype(np.float64)
    # Modulate by density so thin gas is dim; result stays 0-255
    gas_img *= gas_density[:,:,None]

    # ── Layer 3: Spiral arms ──
    angle = np.arctan2(dy, dx)
    n_arms = rng.integers(2, 5)
    arm_tightness = rng.uniform(1.5, 4.0)
    spiral_phase = rng.uniform(0, 2*np.pi)
    arm_field = np.zeros((h, w), dtype=np.float64)
    for arm in range(n_arms):
        arm_angle = angle - arm_tightness * np.log(r + 0.01) - spiral_phase - (2*np.pi*arm/n_arms)
        arm_wave = np.cos(arm_angle) ** 8
        radial_envelope = np.exp(-((r - 0.15)**2) / 0.04) + np.exp(-((r - 0.30)**2) / 0.06)
        arm_field += arm_wave * radial_envelope
    arm_field = np.clip(arm_field, 0, None)
    arm_field /= arm_field.max() + 1e-9
    arm_noise = fbm((h, w), octaves=6, scale=1.5, seed=rng.integers(0, 2**31))
    arm_field = arm_field * (0.7 + 0.3 * arm_noise)
    rot = max(1, len(colors) // 2)
    arm_colors = colors[rot:] + colors[:rot]
    lut_arm = build_lut(arm_colors, 2048)
    # arm_img: color from LUT, masked by arm_field so inter-arm gaps are dark
    arm_img = apply_lut(arm_field, lut_arm, equalize=0.75, n_colors=len(arm_colors)).astype(np.float64)
    arm_img *= arm_field[:,:,None]   # 0-255 attenuated by arm intensity

    # ── Layer 4: Central glowing core ──
    body_radius = rng.uniform(0.05, 0.12)
    core_field = np.exp(-(r**2) / (body_radius**2 * 0.5))
    core_color = np.array([255, 255, 220], dtype=np.float64)
    outer_color = np.array(colors[len(colors)//2], dtype=np.float64)
    # core_img in 0-255 range before bloom
    core_img = (core_color[None,None,:] * core_field[:,:,None]
                + outer_color[None,None,:] * np.clip(core_field * 0.3, 0, 1)[:,:,None])
    # Bloom: accumulate into a separate buffer so we control its weight
    core_bloom = np.zeros_like(core_img)
    for sigma, strength in [(10, 0.8), (30, 0.5), (80, 0.25), (150, 0.1)]:
        for c in range(3):
            core_bloom[:,:,c] += strength * gaussian_filter(core_img[:,:,c], sigma=sigma)
    core_img = core_img + core_bloom * 0.25   # bloom is additive but capped

    # ── Compose all layers in 0-255 space ──
    # Each layer is already 0-255; weights control relative brightness.
    # Sum then clip — no layer alone should reach 255 at full weight.
    canvas  = star_bg * 255 * 0.8          # stars: dim points of light
    canvas += gas_img  * 0.5               # gas: soft background haze
    canvas += arm_img  * 0.9               # arms: main structural feature
    canvas += np.clip(core_img, 0, 255) * 0.6  # core: bright center, not blown out
    return np.clip(canvas, 0, 255).astype(np.uint8)

# ── TIEDYE ────────────────────────────────────────────────────────────────────

def generate_tiedye(shape, colors, seed):
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    n_centers = rng.integers(3, 8)
    cx = rng.uniform(0.1 * w, 0.9 * w, n_centers)
    cy = rng.uniform(0.1 * h, 0.9 * h, n_centers)
    swirl = np.zeros((h, w), dtype=np.float64)
    for i in range(n_centers):
        dx = (xx - cx[i]) / w
        dy = (yy - cy[i]) / h
        r = np.sqrt(dx**2 + dy**2) + 1e-6
        sin_a = dy / r
        cos_a = dx / r
        s = rng.uniform(4, 9)
        rings = rng.uniform(5, 14)
        angular = s * (sin_a + cos_a)
        contrib = np.sin(rings * r * np.pi + angular) * 0.5 + 0.5
        weight = np.exp(-r * rng.uniform(1.5, 4.0))
        swirl += contrib * weight
    noise = fbm((h, w), octaves=5, scale=1.5, seed=rng.integers(0, 2**31))
    field = 0.78 * (swirl / (swirl.max() + 1e-9)) + 0.22 * noise
    field = (field - field.min()) / (field.max() - field.min() + 1e-9)
    shuffled = list(colors)
    rng.shuffle(shuffled)
    lut = build_lut(shuffled, 2048)
    return apply_lut(field, lut, equalize=0.82, n_colors=len(colors))

# ── Post-processing ──────────────────────────────────────────────────────────

def postprocess(arr, do_vignette=True, do_grade=True, contrast=1.15, saturation=1.2):
    if do_grade:
        arr = color_grade(arr, contrast, saturation)
    if do_vignette:
        arr = vignette(arr)
    return arr

STYLES = {
    "nebula":  generate_nebula,
    "lava":    generate_lava,
    "tiedye":  generate_tiedye,
    "cosmic":  generate_cosmic,
}

# ── Color expansion ──────────────────────────────────────────────────────────

def expand_interpolate(colors, steps_between):
    if steps_between <= 0 or len(colors) < 2:
        return colors
    labs = [rgb_to_oklab(c) for c in colors]
    result = []
    for i in range(len(labs) - 1):
        result.append(colors[i])
        for s in range(1, steps_between + 1):
            t = s / (steps_between + 1)
            interp = tuple(labs[i][k] * (1-t) + labs[i+1][k] * t for k in range(3))
            result.append(oklab_to_rgb(interp))
    result.append(colors[-1])
    return result

def expand_variations(colors, variations, spread, rng_seed=None):
    if variations <= 0:
        return colors
    rng = np.random.default_rng(rng_seed)
    result = []
    for col in colors:
        result.append(col)
        L, a, b = rgb_to_oklab(col)
        chroma = np.sqrt(a**2 + b**2) + 1e-9
        hue = np.arctan2(b, a)
        for _ in range(variations):
            new_hue    = hue    + rng.uniform(-np.pi * spread, np.pi * spread)
            new_chroma = chroma * rng.uniform(1 - spread * 0.5, 1 + spread * 0.5)
            new_L      = np.clip(L + rng.uniform(-0.15 * spread, 0.15 * spread), 0.05, 0.95)
            result.append(oklab_to_rgb((new_L, new_chroma * np.cos(new_hue), new_chroma * np.sin(new_hue))))
    return result

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_color(s):
    s = s.strip()
    if s.startswith("#"):
        h = s.lstrip("#")
        if len(h) != 6:
            raise argparse.ArgumentTypeError(f"Invalid hex: {s}")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    parts = s.split(",")
    if len(parts) == 3:
        try:
            return tuple(int(p.strip()) for p in parts)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"Color must be '#RRGGBB' or 'R,G,B', got: {s}")

def main():
    parser = argparse.ArgumentParser(
        description="Cosmic wallpaper generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wallpaper_gen.py --style nebula --width 3840 --height 2160
  python wallpaper_gen.py --style lava --colors "#000000" "#ff4400" "#ffffff"
  python wallpaper_gen.py --style tiedye --expand interpolate --expand-count 3
  python wallpaper_gen.py --style cosmic --seed 42 --no-vignette
""")
    parser.add_argument("-W", "--width",    type=int, default=1920)
    parser.add_argument("-H", "--height",   type=int, default=1080)
    parser.add_argument("--style",          choices=list(STYLES.keys()), default="cosmic")
    parser.add_argument("--palette",        choices=list(PALETTES.keys()), default=None)
    parser.add_argument("--colors",         type=parse_color, nargs="+", metavar="COLOR")
    parser.add_argument("--expand",         choices=["interpolate", "variations"], default=None)
    parser.add_argument("--expand-count",   type=int, default=2, metavar="N")
    parser.add_argument("--expand-spread",  type=float, default=0.3, metavar="S")
    parser.add_argument("--seed",           type=int, default=None)
    parser.add_argument("--output",         type=str, default=None)
    # Use store_false with explicit dest so --no-vignette / --no-grade disable them
    parser.add_argument("--no-vignette",    dest="do_vignette", action="store_false",
                        help="Disable vignette post-processing (off by default)")
    parser.add_argument("--no-grade",       dest="do_grade",    action="store_false",
                        help="Disable contrast/saturation grading (off by default)")
    parser.set_defaults(do_vignette=False, do_grade=False)
    parser.add_argument("--contrast",       type=float, default=1.15)
    parser.add_argument("--saturation",     type=float, default=1.20)
    parser.add_argument("--list-palettes",  action="store_true")
    args = parser.parse_args()

    if args.list_palettes:
        for name, cols in PALETTES.items():
            print(f"  {name:10s}  {len(cols)} colours")
        return

    if args.colors:
        colors = list(args.colors)
    elif args.palette:
        colors = list(PALETTES[args.palette])
    else:
        defaults = {"nebula": "nebula", "lava": "lava", "tiedye": "tiedye", "cosmic": "nebula"}
        colors = list(PALETTES[defaults.get(args.style, "nebula")])

    if args.expand == "interpolate":
        before = len(colors)
        colors = expand_interpolate(colors, args.expand_count)
        print(f"  Palette  : {before} → {len(colors)} colors (interpolated, {args.expand_count} steps between)")
    elif args.expand == "variations":
        before = len(colors)
        colors = expand_variations(colors, args.expand_count, args.expand_spread, rng_seed=args.seed)
        print(f"  Palette  : {before} → {len(colors)} colors (variations ×{args.expand_count}, spread {args.expand_spread})")

    seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    shape = (args.height, args.width)

    print(f"  Style    : {args.style}  |  {args.width}×{args.height}  |  seed {seed}")
    print("  Rendering...", flush=True)
    t0 = time.time()

    img_array = STYLES[args.style](shape, colors, seed)
    img_array = postprocess(img_array, args.do_vignette, args.do_grade, args.contrast, args.saturation)

    print(f"  Done in {time.time()-t0:.1f}s")

    out_path = Path(args.output) if args.output else Path(f"wallpaper_{args.style}_{args.width}x{args.height}_{seed}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_array).save(out_path, format="PNG")
    print(f"  Saved: {out_path}")

if __name__ == "__main__":
    main()
