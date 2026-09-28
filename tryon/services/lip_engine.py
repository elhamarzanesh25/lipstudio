"""
Lip colouring engine: MediaPipe FaceLandmarker (.task) -> lip mask -> finish.

Every finish starts from the same RGBA composite  out = img*(1-a*m) + color*(a*m)
(a = settings.LIP_ALPHA = 0.5) and then adds its own look on top.  Nothing
replaces pixels, so the natural lip texture always remains underneath.
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
    color: np.ndarray    # float32 (1,1,3) BGR
    alpha: float
    depth: np.ndarray    # 0 at lip edge -> 1 at lip centre
    luma: np.ndarray     # 0..1
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
def _light_map(ctx):
    """
    Smooth 0..1 map of how much light each lip pixel receives.
    0 = shadow, 1 = the brightest spot of the lips.  Heavily smoothed so it follows the
    broad lighting of the lips, not the fine cracks, and shaped so that only areas
    close to the maximum get close to 1.
    """
    lum = _blur(ctx.luma, max(1.5, ctx.lip_w * 0.03))
    sel = ctx.mask > 0.5
    lo, hi = np.percentile(lum[sel], [10, 99.7])
    t = np.clip((lum - lo) / max(hi - lo, 1e-3), 0, 1)
    t = t * t * (3 - 2 * t)                                       # smoothstep
    return t ** 2.2                                               # keep white for the top end only


# How much of the light-dependent white each finish gets (0 = none, 1 = full).
LIGHT_MATTE = 0.5
LIGHT_GLOSSY = 1.0
LIGHT_VELVET = 0.6


def _tint_by_light(ctx, strength=1.0):
    """
    Total tint weight stays constant (alpha * mask) everywhere, so there is no visible
    step in overall strength.  Only its split changes with the light:
        s = 0 (shadow)     -> all of it is lip colour
        s = 1 (max light)  -> all of it is white
        out = img*(1-w) + w*((1-s)*color + s*white)
    """
    s = (_light_map(ctx) * strength)[..., None]
    w = (ctx.mask * ctx.alpha)[..., None]
    tint = (1 - s) * ctx.color + s * 255.0
    return ctx.img * (1 - w) + tint * w


def matte(ctx):
    return _tint_by_light(ctx, LIGHT_MATTE)


def glossy(ctx):
    return _soften(ctx, _tint_by_light(ctx, LIGHT_GLOSSY), 0.35)     # softening fills fine cracks                    # fills fine cracks, wet look


def satin(ctx):
    out = _soften(ctx, _composite(ctx), 0.2)
    return _add_white(out, _highlights(ctx, 0.35) + _band(ctx, 0.55, 0.35) * 0.10)


def velvet(ctx):
    out = _soften(ctx, _tint_by_light(ctx, LIGHT_VELVET), 0.5)
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


@dataclass
class Prepared:
    """Everything that depends only on the photo (computed ONCE per upload)."""
    img: np.ndarray      # uint8 BGR
    mask: np.ndarray
    depth: np.ndarray
    luma: np.ndarray
    lip_w: float


def prepare_image(img_bgr) -> Prepared:
    """Slow part: face detection + lip mask.  Call once when the user uploads."""
    lms = detect_landmarks(img_bgr)
    mask = build_lip_mask(lms, img_bgr.shape)
    xs = [lms[i].x * img_bgr.shape[1] for i in OUTER_LIP]
    return Prepared(
        img=img_bgr, mask=mask, depth=_depth(mask),
        luma=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
        lip_w=float(np.ptp(xs)),
    )


def render(prep: Prepared, hex_color, finish="matte", alpha=None):
    """Fast part: colour + finish.  Call as many times as the user changes settings."""
    if finish not in FINISHES:
        raise ValueError(f"Unknown finish: {finish}")
    ctx = Ctx(
        img=prep.img.astype(np.float32), mask=prep.mask,
        color=np.array(hex_to_rgb(hex_color)[::-1], np.float32).reshape(1, 1, 3),
        alpha=settings.LIP_ALPHA if alpha is None else alpha,
        depth=prep.depth, luma=prep.luma, lip_w=prep.lip_w,
        rng=np.random.default_rng(42),
    )
    return np.clip(FINISHES[finish][2](ctx), 0, 255).astype(np.uint8)


def apply_lip_color(img_bgr, hex_color, finish="matte", alpha=None):
    """One-shot helper (prepare + render)."""
    return render(prepare_image(img_bgr), hex_color, finish, alpha)
