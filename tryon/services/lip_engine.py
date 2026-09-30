"""
Lip colouring engine: MediaPipe FaceLandmarker (.task) -> lip mask -> finish.

The paint colour is a mix of the lipstick and the user's own lip colour (LIP_MIX), and is
lightened/darkened per pixel by the lip's own texture (see _shade), not by the room's lighting.
Every finish then lays that paint down with the same RGBA composite
out = img*(1-a*m) + paint*(a*m)  (a = settings.LIP_ALPHA) and adds its own look on top.
Nothing replaces pixels outright, so the natural lip texture always remains underneath.
"""
import threading
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from django.conf import settings
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
             291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
             308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


class NoFaceError(Exception):
    pass


# --------------------------------------------------------------------------
# Landmarker (created once, reused; guarded by a lock for threaded servers)
# --------------------------------------------------------------------------
_lock = threading.Lock()
_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        _landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(settings.FACE_LANDMARKER_MODEL)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
            ))
    return _landmarker


def detect_landmarks(img_bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with _lock:
        res = _get_landmarker().detect(mp_img)
    if not res.face_landmarks:
        raise NoFaceError("No face detected in the image.")
    return res.face_landmarks[0]


def build_lip_mask(lms, shape):
    h, w = shape[:2]
    pts = lambda idx: np.array([[lms[i].x * w, lms[i].y * h] for i in idx], np.float32)
    ss = 4
    m = np.zeros((h * ss, w * ss), np.uint8)
    cv2.fillPoly(m, [np.round(pts(OUTER_LIP) * ss).astype(np.int32)], 255, cv2.LINE_AA)
    cv2.fillPoly(m, [np.round(pts(INNER_LIP) * ss).astype(np.int32)], 0, cv2.LINE_AA)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    sigma = max(1.0, np.ptp(pts(OUTER_LIP)[:, 0]) * 0.02)
    k = int(sigma * 4) | 1
    return np.clip(cv2.GaussianBlur(m, (k, k), sigma), 0, 1)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@dataclass
class Ctx:
    img: np.ndarray      # float32 BGR 0..255
    mask: np.ndarray     # float32 0..1
    color: np.ndarray    # float32 (1,1,3) BGR - the paint colour (already mixed, see LIP_MIX)
    alpha: float
    depth: np.ndarray    # 0 at lip edge -> 1 at lip centre
    shade: np.ndarray    # ~0.6..1.4 local texture lightness (see _shade); 1.0 = neutral
    lip_w: float
    rng: np.random.Generator


def _depth(mask):
    inner = (mask > 0.5).astype(np.uint8)
    d = cv2.distanceTransform(inner, cv2.DIST_L2, 5)
    return d / d.max() if d.max() > 0 else d


def _blur(a, sigma):
    k = int(sigma * 4) | 1
    return cv2.GaussianBlur(a, (k, k), sigma)


def _composite(ctx, weight=None):
    w = (ctx.mask * ctx.alpha if weight is None else weight)[..., None]
    return ctx.img * (1 - w) + ctx.color * w


def _highlights(ctx, strength=1.0):
    """Specular map from the lip's own bright spots (keeps real lip geometry)."""
    sel = ctx.mask > 0.5
    mu, sd = ctx.luma[sel].mean(), ctx.luma[sel].std() + 1e-3
    s = np.clip((ctx.luma - (mu + 0.4 * sd)) / (2.0 * sd), 0, 1) ** 1.5
    return _blur(s * ctx.mask, ctx.lip_w * 0.008) * strength


def _band(ctx, centre=0.6, width=0.22):
    """Soft sheen band following the lip's shape."""
    return np.exp(-((ctx.depth - centre) / width) ** 2) * ctx.mask


def _soften(ctx, out, amount):
    """Blend towards a slightly blurred version inside the lips (hides fine cracks)."""
    b = _blur(out, max(1.0, ctx.lip_w * 0.006))
    m = (ctx.mask * amount)[..., None]
    return out * (1 - m) + b * m


def _add_white(out, amount_map):
    return out + 255.0 * amount_map[..., None]


# --------------------------------------------------------------------------
# Finishes
# --------------------------------------------------------------------------
# Paint colour = LIP_MIX * lipstick + (1 - LIP_MIX) * the user's own average lip colour.
# This is a real colour mix (not just the alpha compositing below), so the lipstick's hue is
# already warmed/cooled by the person's own pigment before it ever touches the photo.
LIP_MIX = 0.7

# How strongly the LIP'S OWN TEXTURE (not the room's lighting - see _shade below) lightens a
# naturally-raised/highlighted spot, and darkens a naturally-recessed one, for each finish.
TONE = {
    "matte":  (0.35, 0.35),   # (light_gain, dark_gain)
    "satin":  (0.50, 0.30),
    "glossy": (0.65, 0.25),
    "velvet": (0.25, 0.45),
}
# Glossy only: extra whitening on the very brightest texture ridges - the "wet" specular look.
GLOSSY_SPECULAR = 0.35


def _tone(ctx):
    """
    Split ctx.shade (0.6..1.4, see _shade()) into a highlight amount and a recess amount, both 0..1.
    """
    t = np.clip((ctx.shade - 1.0) / 0.4, 0, 1)   # 0..1 : natural highlight (raised / lit texture)
    d = np.clip((1.0 - ctx.shade) / 0.4, 0, 1)   # 0..1 : natural recess    (crease / seam / corner)
    return t, d


def _tinted(ctx, light_gain, dark_gain, spec_gain=0.0):
    """
    Paint the lips at a constant opacity (alpha * mask) everywhere, but let the LIP'S OWN
    TEXTURE brighten or darken the paint itself before it's laid down:
        highlight (t=1) -> paint pushed towards white by `light_gain`
        recess    (d=1) -> paint pulled towards black by `dark_gain`
    `ctx.shade` is a *local* ratio (a small blur divided by a wider blur of the same pixels), so a
    global lighting gradient across the face cancels out of it almost completely - only the lip's
    own raised/recessed geometry survives.  See _shade() for the measurement itself.
    """
    t, d = _tone(ctx)
    tint = ctx.color + (t * light_gain)[..., None] * (255.0 - ctx.color)
    if spec_gain:
        tint = tint + ((t ** 2) * spec_gain)[..., None] * (255.0 - tint)
    tint = tint * (1 - d * dark_gain)[..., None]
    w = (ctx.mask * ctx.alpha)[..., None]
    return ctx.img * (1 - w) + tint * w


def matte(ctx):
    light_gain, dark_gain = TONE["matte"]
    return _tinted(ctx, light_gain, dark_gain)


def glossy(ctx):
    light_gain, dark_gain = TONE["glossy"]
    return _soften(ctx, _tinted(ctx, light_gain, dark_gain, GLOSSY_SPECULAR), 0.35)   # softening fills fine cracks


def satin(ctx):
    light_gain, dark_gain = TONE["satin"]
    return _soften(ctx, _tinted(ctx, light_gain, dark_gain), 0.2)


def velvet(ctx):
    light_gain, dark_gain = TONE["velvet"]
    out = _soften(ctx, _tinted(ctx, light_gain, dark_gain), 0.5)
    grain = _blur(ctx.rng.normal(0, 1, ctx.mask.shape).astype(np.float32), 0.7) * 7.0
    out = out + (grain * ctx.mask)[..., None]                          # fine powdery grain
    edge = (1 - ctx.depth) ** 2                                         # soft darker rim
    out = out * (1 - 0.18 * edge * ctx.mask)[..., None]
    return _add_white(out, _band(ctx, 0.6, 0.3) * 0.05)                # very faint sheen


FINISHES = {
    "matte":    ("Matte",    "Flat, no shine",             matte),
    "glossy":   ("Glossy",   "Wet, high-shine",            glossy),
    "satin":    ("Satin",    "Soft, subtle sheen",         satin),
    "velvet":   ("Velvet",   "Powdery, soft-focus",        velvet),
}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def hex_to_rgb(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def decode_image(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Not a valid image file.")
    m = settings.LIP_MAX_SIDE                                           # cap size for speed
    if max(img.shape[:2]) > m:
        s = m / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def _shade(luma, mask):
    """
    Local texture lightness, ~0.6 (deep recess) .. 1.0 (neutral) .. 1.4 (natural highlight ridge).
    It is the ratio of a SMALL blur of the lips' own luma to a WIDER blur of the same pixels, both
    masked so skin and teeth can't leak in.  Dividing a small blur by a wide one is the classic
    "unsharp mask" trick: any large, slowly-varying pattern - such as one side of the face catching
    more room light than the other - shows up almost identically in both blurs and cancels out of
    the ratio.  What survives is the texture that changes over a distance shorter than the wide
    blur: the vermillion border, the cupid's bow ridge, the seam between the lips, the corners.
    That is "the brightness of the lip's own texture", independent of the room's lighting.
    """
    lip_w = float(np.ptp(np.where(mask > 0.5)[1])) if (mask > 0.5).any() else 1.0

    def masked_blur(sigma):
        k = int(sigma * 4) | 1
        num = cv2.GaussianBlur(luma * mask, (k, k), sigma)
        den = cv2.GaussianBlur(mask, (k, k), sigma)
        return num / np.maximum(den, 1e-3), den

    fine, _ = masked_blur(0.8)
    mid, den = masked_blur(max(2.0, lip_w * 0.06))
    shade = np.clip(fine / np.maximum(mid, 0.05), 0.6, 1.4)
    # Fade to neutral (1.0) wherever the wide blur barely overlaps the mask (right at its edge),
    # so the mask boundary itself doesn't get mistaken for a highlight/recess.
    conf = np.clip((den - 0.05) / 0.25, 0, 1)
    conf = conf * conf * (3 - 2 * conf)
    return 1.0 + (shade - 1.0) * conf


@dataclass
class Prepared:
    """Everything that depends only on the photo (computed ONCE per upload)."""
    img: np.ndarray       # uint8 BGR
    mask: np.ndarray
    depth: np.ndarray
    lip_w: float
    lip_mean: np.ndarray  # the user's own average lip colour (BGR, float32)
    shade: np.ndarray     # local texture lightness, see _shade()


def prepare_image(img_bgr) -> Prepared:
    """Slow part: face detection + lip mask.  Call once when the user uploads."""
    lms = detect_landmarks(img_bgr)
    mask = build_lip_mask(lms, img_bgr.shape)
    xs = [lms[i].x * img_bgr.shape[1] for i in OUTER_LIP]
    luma = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return Prepared(
        img=img_bgr, mask=mask, depth=_depth(mask), lip_w=float(np.ptp(xs)),
        lip_mean=img_bgr[mask > 0.5].mean(axis=0).astype(np.float32),
        shade=_shade(luma, mask),
    )


def render(prep: Prepared, hex_color, finish="matte", alpha=None):
    """Fast part: colour + finish.  Call as many times as the user changes settings."""
    if finish not in FINISHES:
        raise ValueError(f"Unknown finish: {finish}")
    lipstick = np.array(hex_to_rgb(hex_color)[::-1], np.float32)
    paint = LIP_MIX * lipstick + (1 - LIP_MIX) * prep.lip_mean         # lipstick coloured by the user's own lips
    ctx = Ctx(
        img=prep.img.astype(np.float32), mask=prep.mask, color=paint.reshape(1, 1, 3),
        alpha=settings.LIP_ALPHA if alpha is None else alpha,
        depth=prep.depth, shade=prep.shade, lip_w=prep.lip_w,
        rng=np.random.default_rng(42),
    )
    return np.clip(FINISHES[finish][2](ctx), 0, 255).astype(np.uint8)


def apply_lip_color(img_bgr, hex_color, finish="matte", alpha=None):
    """One-shot helper (prepare + render)."""
    return render(prepare_image(img_bgr), hex_color, finish, alpha)