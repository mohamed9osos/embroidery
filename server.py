import io
import os
import time
import threading

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse

# ───────────────────────── الإعدادات ─────────────────────────
# أهم رقم: كام بكسل = 1 مم من التصميم الحقيقي.
# ده بيثبّت حجم الغرزة اللي الموديل بيشوفها. اضبطه على حجم الغرزة اللي
# الموديل اتدرب عليه (لو بيطلع غرز كبيرة زوّده، لو بيطلع غرز دقيقة قلّله).
PX_PER_MM = 12.0
MIN_DESIGN_MM, MAX_DESIGN_MM = 20.0, 85.0
DEFAULT_DESIGN_MM = 80.0

OUTLINE_MM = 1.6            # سمك خط الـ outline
COLOR_SIGMA_MM = 0.5        # حجم الضبابية لفصل اللون عن الملمس
LIGHT_TARGET = 175.0        # الألوان الأفتح من كده بتتغمّق قبل الموديل وترجع بعده
LIGHT_K_MIN = 0.6           # أقصى تغميق (k)
DEFRINGE_PX = 3             # عرض شريط الحافة اللي بيتنضف من لون الخلفية الرمادي
COLOR_FIDELITY = 0.8        # ثابت: ألوان الأصل + ملمس الموديل
CROSS_CELL_MM = 1.5         # حجم خلية غرزة الإكس (Cross-stitch)
APPLIQUE_BORDER_MM = 2.0    # عرض إطار الساتان حوالين قماش الـ Appliqué
APPLIQUE_GROW_PX = int(APPLIQUE_BORDER_MM * PX_PER_MM)  # الإطار بيطلع برا الشكل بنص عرضه
PATCH_GROW_PX = 44          # الباتش بيكبّر التصميم بحوالي 22px من كل ناحية

MIN_STROKE_MM = 0.5         # أي خط أرفع من كده بيتسمّك
THIN_GROW_PX = 1            # قد إيه يتسمّك من كل ناحية

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 40_000_000
VALID_STYLES = {"classic", "puff", "patch", "outline", "metallic", "cross", "applique"}

BG_RGB = np.array([235, 235, 235], dtype=np.uint8)

INTERNAL_SECRET = os.environ.get("INTERNAL_RENDER_KEY", "Super_Secret_Render_Key_2026")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"],
)

sess_opts = ort.SessionOptions()
sess_opts.enable_cpu_mem_arena = False
sess_opts.enable_mem_pattern = False
sess_opts.intra_op_num_threads = max(1, os.cpu_count() or 2)
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

EMB_SESSION = ort.InferenceSession(
    "embroidery_model.onnx",
    sess_options=sess_opts,
    providers=["CPUExecutionProvider"],
)

# طلب واحد في نفس الوقت عشان الرامات
_INFER_LOCK = threading.Semaphore(1)


RELIEF_EDGE_MM = 1.2        # عرض الـ bevel عند الحافة
STITCH_PITCH_MM = 0.45      # المسافة بين خطوط الغرز
STITCH_LEN_MM = 3.0         # طول الغرزة قبل التكسير
STITCH_ANGLE_DEG = 45
INTERIOR_START_MM = 1.5     # بعد كام مم من الحافة تبدأ الغرز الصناعية
INTERIOR_FULL_MM = 3.5      # عندها الغرز تبقى بكامل قوتها
RELIEF_STRENGTH = 0.9       # قوة الإضاءة الكلية


def add_uniform_relief(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """ريليف ثابت بالمليمتر: حافة بارزة + غرز في الداخل، بغض النظر عن حجم التصميم."""
    mask = (alpha > 0.4).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    H, W = mask.shape

    # 1) bevel ثابت عند الحافة
    r = RELIEF_EDGE_MM * PX_PER_MM
    bevel = 1.0 - np.exp(-dist / r)
    bevel = cv2.GaussianBlur(bevel, (0, 0), 1.5)

    # 2) غرز ساتان اصطناعية
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    th = np.deg2rad(STITCH_ANGLE_DEG)
    u = xx * np.cos(th) + yy * np.sin(th)     # عمودي على الغرز
    v = -xx * np.sin(th) + yy * np.cos(th)    # على طول الغرز
    pitch = STITCH_PITCH_MM * PX_PER_MM
    slen = STITCH_LEN_MM * PX_PER_MM

    row = np.floor(u / pitch).astype(np.int32)
    row -= row.min()
    rng = np.random.default_rng(11)
    phase = rng.uniform(0, slen, row.max() + 1).astype(np.float32)[row]

    ridge = 0.5 + 0.5 * np.cos(2 * np.pi * u / pitch)           # حرف الغرزة
    pos = np.mod(v + phase, slen)                                 # مكان التكسير
    gap = np.exp(-((pos / (0.25 * PX_PER_MM)) ** 2))              # نقطة دخول الإبرة
    stitch = ridge * (1.0 - 0.6 * gap)

    # الغرز تظهر بس في الداخل
    s, e = INTERIOR_START_MM * PX_PER_MM, INTERIOR_FULL_MM * PX_PER_MM
    w_in = np.clip((dist - s) / max(e - s, 1.0), 0.0, 1.0)
    w_in = w_in * w_in * (3 - 2 * w_in)

    height = 1.2 * bevel + 0.35 * stitch * w_in

    # 3) إضاءة موحدة
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    light = np.clip(-(gx * 0.65 + gy * 0.75) * RELIEF_STRENGTH * 0.5, -0.28, 0.30)
    light *= (alpha > 0.4)

    shaded = rgb.astype(np.float32) * (1.0 + light[:, :, None])
    spec = np.clip(light - 0.15, 0.0, 0.2) * 120.0
    return np.clip(shaded + spec[:, :, None], 0, 255).astype(np.uint8)


# ───────────────────────── أدوات مساعدة ─────────────────────────
def hex_to_rgb(hex_color: str, default=(255, 255, 255)) -> tuple:
    try:
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return default


def odd(n: int, lo: int = 3) -> int:
    n = max(lo, int(round(n)))
    return n if n % 2 == 1 else n + 1


def resize_rgba(rgba: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Resize لـ RGBA بطريقة premultiplied: بيمنع تلوّث الحواف بألوان البكسلات الشفافة
    (ده كان سبب الشريط الأخضر)."""
    h, w = rgba.shape[:2]
    interp = cv2.INTER_AREA if (new_w * new_h) < (w * h) else cv2.INTER_CUBIC
    a = rgba[:, :, 3].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32) * a[:, :, None]
    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
    a = np.clip(cv2.resize(a, (new_w, new_h), interpolation=interp), 0.0, 1.0)
    rgb = rgb / np.maximum(a[:, :, None], 1e-4)
    rgb[a < (1.0 / 255.0)] = 0
    return np.dstack([
        np.clip(rgb, 0, 255),
        np.clip(a * 255.0, 0, 255),
    ]).astype(np.uint8)


def masked_blur(img: np.ndarray, a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur بيتجاهل البكسلات الشفافة (normalized convolution)."""
    img = img.astype(np.float32)
    num = cv2.GaussianBlur(img * a[:, :, None], (0, 0), sigma)
    if num.ndim == 2:  # OpenCV بيشيل بعد القناة الواحدة
        num = num[:, :, None]
    den = cv2.GaussianBlur(a, (0, 0), sigma)
    return num / np.maximum(den[:, :, None], 1e-3)

def thicken_thin_parts(rgba: np.ndarray) -> np.ndarray:
    """بيسمّك الخطوط الرفيعة بس (زي الزخارف) عشان الموديل ما يضيّعهاش."""
    a = rgba[:, :, 3]
    binary = (a > 20).astype(np.uint8) * 255

    k = odd(MIN_STROKE_MM * PX_PER_MM)
    opened = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    thin = cv2.subtract(binary, opened)

    g = 2 * THIN_GROW_PX + 1
    thin_d = cv2.dilate(thin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g)))
    thin_d = cv2.GaussianBlur(thin_d, (3, 3), 0)

    new_a = np.maximum(a, thin_d)

    fill = masked_blur(rgba[:, :, :3], a.astype(np.float32) / 255.0, 2.0)
    need = new_a > a
    rgb = np.where(need[:, :, None], fill, rgba[:, :, :3])

    return np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), new_a])

# ───────────────────────── أنماط التطريز (قبل الموديل) ─────────────────────────
def convert_to_outline(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3]
    binary = (alpha > 40).astype(np.uint8) * 255
    t = odd(OUTLINE_MM * PX_PER_MM)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
    eroded = cv2.erode(binary, k, iterations=1)
    ring = cv2.subtract(binary, eroded)
    ring = cv2.GaussianBlur(ring, (3, 3), 0)
    out = rgba.copy()
    out[:, :, 3] = np.minimum(alpha, ring)
    return out


def wrap_as_patch(rgba: np.ndarray, bg_hex: str, border_hex: str) -> np.ndarray:
    extra_pad = 54
    padded = cv2.copyMakeBorder(rgba, extra_pad, extra_pad, extra_pad, extra_pad,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    alpha = (padded[:, :, 3] > 25).astype(np.uint8) * 255

    k_base = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    base_mask = cv2.dilate(alpha, k_base, iterations=1)
    base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, k_base)
    base_mask = cv2.GaussianBlur(base_mask, (15, 15), 0)
    _, base_bin = cv2.threshold(base_mask, 120, 255, cv2.THRESH_BINARY)

    k_border = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    inner_bin = cv2.erode(base_bin, k_border, iterations=1)
    border_ring = cv2.subtract(base_bin, inner_bin)

    bg_rgb = hex_to_rgb(bg_hex, (255, 255, 255))
    border_rgb = hex_to_rgb(border_hex, (17, 24, 39))

    canvas = np.zeros_like(padded)
    canvas[base_bin > 0, :3] = bg_rgb
    canvas[base_bin > 0, 3] = 255
    canvas[border_ring > 0, :3] = border_rgb

    fg_a = padded[:, :, 3:4].astype(np.float32) / 255.0
    canvas[:, :, :3] = np.clip(
        padded[:, :, :3].astype(np.float32) * fg_a
        + canvas[:, :, :3].astype(np.float32) * (1.0 - fg_a),
        0, 255,
    ).astype(np.uint8)
    canvas[:, :, 3] = np.maximum(canvas[:, :, 3], padded[:, :, 3])
    return canvas


# ───────────────────────── تأثيرات (بعد الموديل) ─────────────────────────
def transfer_color_from_input(out_rgb: np.ndarray, src_rgb: np.ndarray,
                              a: np.ndarray, strength: float) -> np.ndarray:
    """الألوان العامة (الترددات المنخفضة) من الأصل، والملمس (الترددات العالية) من الموديل."""
    if strength <= 0:
        return out_rgb
    sigma = max(1.0, COLOR_SIGMA_MM * PX_PER_MM)
    low_in = masked_blur(src_rgb, a, sigma)
    low_out = masked_blur(out_rgb, a, sigma)
    res = out_rgb.astype(np.float32) + strength * (low_in - low_out)
    return np.clip(res, 0, 255).astype(np.uint8)


def compute_light_k(rgb: np.ndarray, a: np.ndarray) -> np.ndarray:
    """معامل تغميق لكل منطقة: الأبيض والألوان الفاتحة بتتحول لدرجة متوسطة
    (قريبة من الرمادي اللي الموديل بيرسمه كويس) وبعد الموديل بنقسم على نفس المعامل."""
    gray = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    sigma = max(1.0, COLOR_SIGMA_MM * PX_PER_MM)
    lum = masked_blur(gray[:, :, None], a, sigma)[:, :, 0]
    k = np.clip(LIGHT_TARGET / np.maximum(lum, 1.0), LIGHT_K_MIN, 1.0)
    return k[:, :, None].astype(np.float32)


def defringe(rgb: np.ndarray, alpha: np.ndarray, band: int = DEFRINGE_PX) -> np.ndarray:
    """بيشيل الهالة الفاتحة حوالين الحواف: الموديل بيشوف التصميم على خلفية رمادي فاتح
    فبيطلع شريط رمادي على الحافة، وده بيبان أبيض على القماش الغامق.
    بنستبدل ألوان الشريط بألوان الداخل القريبة منه."""
    core = (alpha > 0.95).astype(np.uint8)
    core = cv2.erode(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1)))
    core_f = core.astype(np.float32)
    sigma = band * 0.8 + 1.0
    fill = masked_blur(rgb, core_f, sigma)
    valid = cv2.GaussianBlur(core_f, (0, 0), sigma) > 0.02
    use_fill = (core_f < 0.5) & valid
    out = np.where(use_fill[:, :, None], fill, rgb.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_3d_puff_shading(rgb: np.ndarray, mask_f: np.ndarray) -> np.ndarray:
    binary = (mask_f > 0.3).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    r = 3.0 * PX_PER_MM                      # قبة ثابتة بالمليمتر
    height = 1.0 - np.exp(-dist / r)
    height = cv2.GaussianBlur(height, (9, 9), 0)

    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=5)

    light = (-gx * 0.65 - gy * 0.75)
    light = np.clip(light * 0.42, -0.28, 0.32)

    shaded = rgb.astype(np.float32) * (1.0 + light[:, :, None])
    specular = np.clip(light - 0.12, 0.0, 0.25) * 140.0
    shaded = shaded + specular[:, :, None]
    return np.clip(shaded, 0, 255).astype(np.uint8)


def apply_metallic_effect(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB).astype(np.float32)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    highlights = np.power(np.clip((gray - 0.55) / 0.45, 0.0, 1.0), 1.8) * 65.0
    merged = merged * 1.06 + highlights[:, :, None]
    return np.clip(merged, 0, 255).astype(np.uint8)


# ───────────────────────── Cross-stitch (من غير موديل) ─────────────────────────
def render_cross_stitch(rgba: np.ndarray, has_transparency: bool) -> np.ndarray:
    """بيقسّم التصميم لشبكة ويرسم X من خيط في كل خلية بلون متوسط المنطقة."""
    cell = max(8, int(round(CROSS_CELL_MM * PX_PER_MM)))
    h, w = rgba.shape[:2]
    gw, gh = max(1, w // cell), max(1, h // cell)
    grid = resize_rgba(rgba, gw, gh)
    active = grid[:, :, 3] >= 128
    # ألوان محدودة شوية زي خيوط الـ cross-stitch الحقيقية
    colors = ((grid[:, :, :3].astype(np.int32) // 16) * 16 + 8).astype(np.float32)

    H, W = gh * cell, gw * cell
    if has_transparency:
        canvas = np.zeros((H, W, 3), np.uint8)
        alpha = np.zeros((H, W), np.uint8)
    else:
        # الصور العادية: قماش Aida فاتح تحت الغرز
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        weave = 1.0 + 0.03 * np.sin(2 * np.pi * xx / (cell / 2)) + 0.03 * np.sin(2 * np.pi * yy / (cell / 2))
        base = np.array([242, 238, 228], np.float32)
        canvas = np.clip(base[None, None, :] * weave[:, :, None], 0, 255).astype(np.uint8)
        alpha = np.full((H, W), 255, np.uint8)

    rng = np.random.default_rng(7)
    t = max(3, int(cell * 0.34))
    inset = max(1, int(cell * 0.12))
    cells = list(zip(*np.nonzero(active)))
    jitter = rng.uniform(0.94, 1.06, len(cells))

    # الغرزة اللي تحت "/" الأول وبعدين اللي فوق "\\" زي الشغل الحقيقي
    for diag in (0, 1):
        for (gy, gx), j in zip(cells, jitter):
            x0, y0 = gx * cell, gy * cell
            if diag == 0:
                p1, p2 = (x0 + inset, y0 + cell - inset), (x0 + cell - inset, y0 + inset)
            else:
                p1, p2 = (x0 + inset, y0 + inset), (x0 + cell - inset, y0 + cell - inset)
            c = np.clip(colors[gy, gx] * j, 0, 255)
            layers = (
                (tuple(int(v) for v in c * 0.62), t, 0),
                (tuple(int(v) for v in c), max(2, int(t * 0.72)), 0),
                (tuple(int(v) for v in np.clip(c * 1.22 + 14, 0, 255)), max(1, int(t * 0.26)), -1),
            )
            for col, th, off in layers:
                a, b = (p1[0] + off, p1[1] + off), (p2[0] + off, p2[1] + off)
                cv2.line(canvas, a, b, col, th, cv2.LINE_AA)
                if th >= 3:
                    cv2.circle(canvas, a, th // 2, col, -1, cv2.LINE_AA)
                    cv2.circle(canvas, b, th // 2, col, -1, cv2.LINE_AA)
            if has_transparency:
                cv2.line(alpha, p1, p2, 255, t, cv2.LINE_AA)
                cv2.circle(alpha, p1, t // 2, 255, -1, cv2.LINE_AA)
                cv2.circle(alpha, p2, t // 2, 255, -1, cv2.LINE_AA)

    return np.dstack([canvas, alpha])


# ───────────────────────── Appliqué (قماش + إطار ساتان) ─────────────────────────
def build_applique_layers(rgba: np.ndarray, border_hex: str):
    """بيرجّع طبقتين: (1) قطعة القماش بملمس twill بألوان التصميم،
    (2) حلقة الإطار اللي هتعدي على الموديل عشان تطلع غرز ساتان."""
    b = odd(APPLIQUE_BORDER_MM * PX_PER_MM)
    half = b // 2
    m = half + 4
    rgba = cv2.copyMakeBorder(rgba, m, m, m, m, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    a = (rgba[:, :, 3] > 25).astype(np.uint8) * 255
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (b, b))
    ring = cv2.subtract(cv2.dilate(a, k), cv2.erode(a, k))
    ring = cv2.GaussianBlur(ring, (3, 3), 0)

    border_rgb = hex_to_rgb(border_hex, (17, 24, 39))
    ring_rgba = np.zeros_like(rgba)
    ring_rgba[:, :, :3] = border_rgb
    ring_rgba[:, :, 3] = ring

    # ملمس twill قطري + حبيبات خفيفة
    H, W = a.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    period = max(4.0, 0.45 * PX_PER_MM)
    twill = np.sin((xx + yy) * 2 * np.pi / period)
    grain = cv2.GaussianBlur(np.random.default_rng(3).normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.8)
    tex = 1.0 + 0.05 * twill + 0.04 * grain
    fill_rgb = np.clip(rgba[:, :, :3].astype(np.float32) * tex[:, :, None], 0, 255).astype(np.uint8)
    fill_rgba = np.dstack([fill_rgb, cv2.GaussianBlur(a, (3, 3), 0)])
    return fill_rgba, ring_rgba


# ───────────────────────── الـ Pipeline الرئيسي ─────────────────────────
def run_embroidery_pipeline(
    pil_rgba: Image.Image,
    design_mm: float = DEFAULT_DESIGN_MM,
    style: str = "classic",
    patch_bg: str = "#ffffff",
    patch_border: str = "#111827",
    with_shadow: bool = True,
) -> Image.Image:
    rgba = np.array(pil_rgba.convert("RGBA"))
    has_transparency = int(rgba[:, :, 3].min()) < 250

    # 1. قص الفراغ الشفاف
    if has_transparency:
        ys, xs = np.where(rgba[:, :, 3] > 10)
        if len(xs) > 0 and len(ys) > 0:
            rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 2. مقاس ثابت بالمليمتر => حجم غرزة ثابت
    design_mm = float(np.clip(design_mm, MIN_DESIGN_MM, MAX_DESIGN_MM))
    target = design_mm * PX_PER_MM
    if style == "applique" and not has_transparency:
        style = "classic"  # الـ appliqué محتاج شكل شفاف
    if style == "patch" and has_transparency:
        target -= PATCH_GROW_PX  # الباتش هيكبّر الشكل، فنحسبها من الأول
    elif style == "applique":
        target -= APPLIQUE_GROW_PX
    h0, w0 = rgba.shape[:2]
    scale = target / max(h0, w0)
    if abs(scale - 1.0) > 0.02:
        rgba = resize_rgba(rgba, max(16, int(round(w0 * scale))), max(16, int(round(h0 * scale))))

    # شيل حافة 1 بكسل ملوثة
    if has_transparency:
        rgba = thicken_thin_parts(rgba)

    # 3. تعديلات ما قبل الموديل
    if style == "outline" and has_transparency:
        rgba = convert_to_outline(rgba)
    elif style == "patch" and has_transparency:
        rgba = wrap_as_patch(rgba, patch_bg, patch_border)
    elif style == "cross":
        rgba = render_cross_stitch(rgba, has_transparency)

    fill_rgba = None
    if style == "applique":
        fill_rgba, rgba = build_applique_layers(rgba, patch_border)  # rgba = حلقة الإطار بس

    pad = 32
    h, w = rgba.shape[0] + 2 * pad, rgba.shape[1] + 2 * pad
    pad_b = (4 - h % 4) % 4
    pad_r = (4 - w % 4) % 4

    def _pad(arr):
        arr = cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        if pad_b or pad_r:
            arr = cv2.copyMakeBorder(arr, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        return arr

    rgba_padded = _pad(rgba)

    a_norm = rgba_padded[:, :, 3:4].astype(np.float32) / 255.0

    # تغميق الألوان الفاتحة قبل الموديل (بيتعكس بعده)
    k_map = None
    rgb_model_in = rgba_padded[:, :, :3].astype(np.float32)
    if style != "cross":
        wa = a_norm[:, :, 0] if has_transparency else np.ones_like(a_norm[:, :, 0])
        k_map = compute_light_k(rgba_padded[:, :, :3], wa)
        rgb_model_in = rgb_model_in * k_map

    rgb_on_bg = rgb_model_in * a_norm + BG_RGB.astype(np.float32) * (1.0 - a_norm)
    rgb_on_bg = np.clip(rgb_on_bg, 0, 255).astype(np.uint8)

    # 4. تشغيل الموديل (الـ cross-stitch مرسوم برمجيًا ومش محتاج موديل)
    if style == "cross":
        out_rgb = rgba_padded[:, :, :3].copy()
    else:
        inp = rgb_on_bg.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
        with _INFER_LOCK:
            out = EMB_SESSION.run(["output"], {"input": inp})[0][0]
        out_rgb = (np.clip(out.transpose(1, 2, 0), 0.0, 1.0) * 255.0).astype(np.uint8)

        blur = cv2.GaussianBlur(out_rgb, (0, 0), 0.8)
        out_rgb = cv2.addWeighted(out_rgb, 1.05, blur, -0.05, 0)

    # 5. الألوان من الأصل + الملمس من الموديل
    orig_a = rgba_padded[:, :, 3].astype(np.float32) / 255.0
    if style not in ("metallic", "cross"):
        weight_a = orig_a if has_transparency else np.ones_like(orig_a)
        src = rgb_model_in if has_transparency else rgb_on_bg
        out_rgb = transfer_color_from_input(out_rgb, src, weight_a, COLOR_FIDELITY)

    # رجّع السطوع الأصلي للألوان الفاتحة (بيكبّر التباين والملمس كمان)
    if k_map is not None:
        out_rgb = np.clip(out_rgb.astype(np.float32) / k_map, 0, 255).astype(np.uint8)

    # Appliqué: إطار الساتان (من الموديل) فوق قطعة القماش
    if style == "applique":
        fill_p = _pad(fill_rgba)
        fill_a = fill_p[:, :, 3].astype(np.float32) / 255.0
        ring_a = orig_a
        out_rgb = np.clip(
            out_rgb.astype(np.float32) * ring_a[:, :, None]
            + fill_p[:, :, :3].astype(np.float32) * (1.0 - ring_a[:, :, None]),
            0, 255,
        ).astype(np.uint8)
        orig_a = np.maximum(ring_a, fill_a)

    # ريليف ثابت في كل الأحجام
    if has_transparency and style != "cross":
        out_rgb = add_uniform_relief(out_rgb, orig_a)

    # 6. تأثيرات ما بعد الموديل
    if style == "puff":
        out_rgb = apply_3d_puff_shading(out_rgb, orig_a)
    elif style == "metallic":
        out_rgb = apply_metallic_effect(out_rgb)

    if not has_transparency:
        res = out_rgb[pad:h - pad, pad:w - pad]
        return Image.fromarray(res, mode="RGB")

    if style != "cross":
        out_rgb = defringe(out_rgb, orig_a)

    # 7. حافة التطريز الطبيعية
    hh, ww = orig_a.shape
    grid_y, grid_x = np.mgrid[0:hh, 0:ww].astype(np.float32)
    wave_x = np.sin(grid_y * 0.75) * 0.5
    wave_y = np.sin(grid_x * 0.75) * 0.5
    if style == "cross":
        wavy_a = orig_a  # حواف الغرز الحادة من غير تموج
    else:
        wavy_a = cv2.remap(orig_a, grid_x + wave_x, grid_y + wave_y, cv2.INTER_LINEAR, borderValue=0.0)

    fg_a = np.clip((wavy_a - 0.10) / 0.60, 0.0, 1.0)
    if style != "cross":
        fg_a = cv2.erode(fg_a, np.ones((3, 3), np.uint8))  # شيل بكسل الحافة المتأثر بالخلفية
    fg_a_3d = fg_a[:, :, None]

    if with_shadow:
        if style == "puff":
            sh_dx, sh_dy, sh_blur, sh_op = 5.0, 7.0, 15, 0.62
        elif style in ("patch", "applique"):
            sh_dx, sh_dy, sh_blur, sh_op = 3.5, 4.5, 13, 0.52
        else:
            sh_dx, sh_dy, sh_blur, sh_op = 2.5, 3.5, 11, 0.42

        M_shift = np.float32([[1, 0, sh_dx], [0, 1, sh_dy]])
        sh_mask = cv2.warpAffine(fg_a, M_shift, (ww, hh), borderValue=0.0)
        sh_a = np.clip(cv2.GaussianBlur(sh_mask, (sh_blur, sh_blur), 0) * sh_op, 0.0, 1.0)[:, :, None]
        final_a = fg_a_3d + sh_a * (1.0 - fg_a_3d)
    else:
        # بدون ظل: مناسب لو هتحط الصورة على mockup فيه ظلاله الخاصة
        final_a = fg_a_3d

    final_rgb = (out_rgb.astype(np.float32) * fg_a_3d) / np.maximum(final_a, 1e-5)

    result_rgba = np.dstack([
        np.clip(final_rgb, 0, 255).astype(np.uint8),
        np.clip(final_a[:, :, 0] * 255.0, 0, 255).astype(np.uint8),
    ])

    if pad_b:
        result_rgba = result_rgba[:-pad_b, :]
    if pad_r:
        result_rgba = result_rgba[:, :-pad_r]

    return Image.fromarray(result_rgba, mode="RGBA")


# ───────────────────────── الـ API ─────────────────────────
@app.get("/")
def home():
    return {"status": "Embroidery ONNX Engine Ready 🚀"}


@app.post("/embroidery")
def apply_embroidery(
    file: UploadFile = File(...),
    style: str = Form("classic"),
    patch_bg: str = Form("#ffffff"),
    patch_border: str = Form("#111827"),
    design_mm: float = Form(DEFAULT_DESIGN_MM),
    with_shadow: bool = Form(True),
    x_internal_secret: str = Header(None),
):
    if x_internal_secret != INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if style not in VALID_STYLES:
        raise HTTPException(status_code=400, detail=f"style غير مدعوم: {style}")
    try:
        t0 = time.time()
        contents = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="الصورة كبيرة (الحد 15MB)")

        img = Image.open(io.BytesIO(contents))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGBA")

        emb_img = run_embroidery_pipeline(
            img,
            design_mm=design_mm,
            style=style,
            patch_bg=patch_bg,
            patch_border=patch_border,
            with_shadow=with_shadow,
        )

        buf = io.BytesIO()
        emb_img.save(buf, format="WEBP", quality=90, method=4)
        elapsed = time.time() - t0

        return Response(
            content=buf.getvalue(),
            media_type="image/webp",
            headers={"X-Process-Time": f"{elapsed:.2f}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
