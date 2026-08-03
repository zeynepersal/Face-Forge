"""
FaceForge Backend — Flask API
Endpoints:
  POST /api/landmarks     → detect landmarks, return JSON coords
  POST /api/expression    → warp face (smile, eyebrow, lips, slim)
  POST /api/aging         → aging / de-aging via frequency domain (Fixed)
  POST /api/golden_ratio  → analyze golden ratio + generate ideal overlay
"""

import base64, io, math, json, os, time
import numpy as np
import cv2
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import mediapipe as mp

FAL_KEY = '598e9ac3-4767-43a3-b16c-267b0c8303cb:c72903d9402cf49c000483c73f31a61f'
FAL_HEADERS = {'Authorization': f'Key {FAL_KEY}', 'Content-Type': 'application/json'}
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash-preview-image-generation").strip()
_wig_cache = {}
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

app = Flask(__name__)
CORS(app)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# ── Helpers ──────────────────────────────────────────────────────────────────

def decode_image(data_url):
    """Decode base64 data URL to numpy BGR array."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def encode_image(bgr):
    """Encode numpy BGR array to base64 PNG data URL."""
    _, buf = cv2.imencode(".png", bgr)
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def data_url_to_gemini_inline(image_data_url):
    header, encoded = image_data_url.split(",", 1)
    mime_type = "image/png"
    if header.startswith("data:") and ";" in header:
        mime_type = header[5:].split(";", 1)[0] or mime_type
    return {"inlineData": {"mimeType": mime_type, "data": encoded}}

def edit_image_with_gemini(image_data_url, prompt):
    if not GEMINI_API_KEY:
        return None
    url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_IMAGE_MODEL}:generateContent"
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            data_url_to_gemini_inline(image_data_url),
        ]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}
    }
    resp = requests.post(
        url,
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    resp.raise_for_status()
    parts = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            mime_type = inline.get("mimeType") or inline.get("mime_type") or "image/png"
            return f"data:{mime_type};base64,{inline['data']}"
    return None

def detect_apparent_gender(image_data_url):
    """One-shot Gemini vision call — returns 'female', 'male', or None."""
    if not GEMINI_API_KEY:
        return None
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        payload = {
            "contents": [{"parts": [
                {"text": "Look at this portrait photo. What is the apparent gender of the person? Reply with exactly one word: female or male."},
                data_url_to_gemini_inline(image_data_url),
            ]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 5},
        }
        resp = requests.post(
            url,
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        parts = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        for p in parts:
            t = p.get("text", "").strip().lower()
            if "female" in t or "woman" in t or "girl" in t:
                return "female"
            if "male" in t or "man" in t or "boy" in t:
                return "male"
    except Exception as e:
        print(f"Gender detection failed: {e}")
    return None


def get_landmarks(img_bgr):
    """Return list of (x,y) pixel coords for 478 landmarks (468 face + 10 iris)."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with mp_vision.FaceLandmarker.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_image)
    if not res.face_landmarks:
        return None
    lms = res.face_landmarks[0]
    return [(int(lm.x * w), int(lm.y * h)) for lm in lms]

def warp_triangle(src, dst, t_src, t_dst):
    """Affine warp a single triangle from src to dst."""
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    t1_off = [(p[0]-r1[0], p[1]-r1[1]) for p in t_src]
    t2_off = [(p[0]-r2[0], p[1]-r2[1]) for p in t_dst]
    M = cv2.getAffineTransform(np.float32(t1_off), np.float32(t2_off))
    patch = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    if patch.size == 0:
        return
    warped = cv2.warpAffine(patch, M, (r2[2], r2[3]),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(t2_off), 255)
    region = dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    if region.shape[:2] != warped.shape[:2]:
        return
    region[mask > 0] = warped[mask > 0]

def delaunay_warp(img, src_pts, dst_pts):
    """Warp img using Delaunay triangulation between src_pts and dst_pts."""
    h, w = img.shape[:2]
    result = img.copy()
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    dst_valid = []
    for p in dst_pts:
        px, py = float(np.clip(p[0], 1, w-2)), float(np.clip(p[1], 1, h-2))
        subdiv.insert((px, py))
        dst_valid.append((px, py))
    triangles = subdiv.getTriangleList().astype(np.int32)
    dst_arr = np.array(dst_valid)
    src_arr = np.array(src_pts)
    for tri in triangles:
        pts = [(tri[0],tri[1]),(tri[2],tri[3]),(tri[4],tri[5])]
        idxs = []
        valid = True
        for pt in pts:
            dists = np.linalg.norm(dst_arr - pt, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] > 3:
                valid = False; break
            idxs.append(idx)
        if not valid: continue
        t_src = [src_arr[i] for i in idxs]
        t_dst = [dst_arr[i] for i in idxs]
        warp_triangle(img, result, t_src, t_dst)
    return result

# ── MediaPipe landmark indices ────────────────────────────────────────────────
# Mouth corners
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
# Lips
UPPER_LIP_LEFT  = 61
UPPER_LIP_RIGHT = 291
# Eyebrows
L_BROW = [70,63,105,66,107]
R_BROW = [300,293,334,296,336]
# Jaw / cheeks
JAW_LEFT   = [234,227,137,177,215]
JAW_RIGHT  = [454,447,366,401,435]
# Eyes
L_EYE = [33,160,158,133,153,144]
R_EYE = [362,385,387,263,373,380]
# Nose tip
NOSE_TIP = 4
# Nose sides (for golden ratio / golden warp)
NOSE_LEFT_IDX, NOSE_RIGHT_IDX = 129, 358
# Chin
CHIN = 152
# Forehead (approx)
FOREHEAD = 10

# ── ROUTE: /api/landmarks ─────────────────────────────────────────────────────

@app.route("/api/landmarks", methods=["POST"])
def landmarks():
    data = request.get_json()
    img = decode_image(data["image"])
    pts = get_landmarks(img)
    if pts is None:
        return jsonify({"error": "No face detected"}), 400
    h, w = img.shape[:2]
    # Draw overlay
    overlay = img.copy()
    for x, y in pts:
        cv2.circle(overlay, (x, y), 1, (79, 142, 247), -1)
    return jsonify({
        "landmarks": pts,
        "count": len(pts),
        "overlay": encode_image(overlay),
        "width": w,
        "height": h
    })

# ── ROUTE: /api/expression ────────────────────────────────────────────────────

def local_translate(img, cx, cy, radius, tx, ty):
    """
    Locally translates pixels near (cx, cy) by the vector (tx, ty).
    Uses a smooth cosine-squared falloff for seamless blending.
    """
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Calculate distance from the center point
    dx = xs - cx
    dy = ys - cy
    dist = np.sqrt(dx*dx + dy*dy)
    
    # Smooth falloff: 1.0 at center, 0.0 outside radius
    factor = np.where(dist < radius,
                      np.cos(np.pi * dist / (2 * radius))**2,
                      0.0)
    
    # Inverse map: to move an output pixel by (tx, ty), we sample from (x - tx, y - ty)
    src_x = np.clip(xs - tx * factor, 0, w - 1).astype(np.float32)
    src_y = np.clip(ys - ty * factor, 0, h - 1).astype(np.float32)
    
    return cv2.remap(img, src_x, src_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT101)

@app.route("/api/expression", methods=["POST"])
def expression():
    data = request.get_json()
    img = decode_image(data["image"])
    h, w = img.shape[:2]

    smile_intensity   = float(data.get("smile",   0)) / 100.0
    eyebrow_intensity = float(data.get("eyebrow", 0)) / 100.0
    lip_intensity     = float(data.get("lips",    0)) / 100.0
    slim_intensity    = float(data.get("slim",    0)) / 100.0

    pts = get_landmarks(img)
    if pts is None:
        return jsonify({"error": "No face detected"}), 400

    src = list(pts)
    result = img.copy()

    # Dynamic scaling based on face size so the effect works on any resolution
    face_w = abs(src[454][0] - src[234][0]) if len(src) > 454 else w
    face_h_val = abs(src[152][1] - src[10][1]) if len(src) > 152 else h

    # ── SMILE ─────────────────────────────────────────────────────────────
    if smile_intensity > 0:
        ml = src[MOUTH_LEFT]    # index 61
        mr = src[MOUTH_RIGHT]   # index 291
        mouth_w = abs(mr[0] - ml[0])
        
        corner_radius = mouth_w * 0.8
        corner_lift   = smile_intensity * mouth_w * 0.25  # pixel translation amount

        # Left corner — pull UP (negative Y) and LEFT (negative X)
        result = local_translate(result, ml[0], ml[1], corner_radius, -corner_lift * 0.5, -corner_lift)
        # Right corner — pull UP (negative Y) and RIGHT (positive X)
        result = local_translate(result, mr[0], mr[1], corner_radius, corner_lift * 0.5, -corner_lift)

        # Cheek lift: push cheek volumes gently upward
        cheek_radius = face_w * 0.3
        cheek_lift   = smile_intensity * face_h_val * 0.03
        result = local_translate(result, src[205][0], src[205][1], cheek_radius, 0, -cheek_lift)
        result = local_translate(result, src[425][0], src[425][1], cheek_radius, 0, -cheek_lift)

    # ── EYEBROW RAISE ─────────────────────────────────────────────────────
    if eyebrow_intensity > 0:
        # Find the center of each eyebrow
        lb_cx = sum([src[i][0] for i in L_BROW]) // len(L_BROW)
        lb_cy = sum([src[i][1] for i in L_BROW]) // len(L_BROW)
        rb_cx = sum([src[i][0] for i in R_BROW]) // len(R_BROW)
        rb_cy = sum([src[i][1] for i in R_BROW]) // len(R_BROW)

        brow_radius = face_w * 0.3
        brow_lift = eyebrow_intensity * face_h_val * 0.06

        # Pull eyebrows straight UP (negative Y)
        result = local_translate(result, lb_cx, lb_cy, brow_radius, 0, -brow_lift)
        result = local_translate(result, rb_cx, rb_cy, brow_radius, 0, -brow_lift)

    # ── LIP WIDENING ──────────────────────────────────────────────────────
    if lip_intensity > 0:
        ml = src[MOUTH_LEFT]
        mr = src[MOUTH_RIGHT]
        mouth_w = abs(mr[0] - ml[0])
        lip_r = mouth_w * 0.6
        lip_push = lip_intensity * mouth_w * 0.15

        # Left corner moves strictly LEFT
        result = local_translate(result, ml[0], ml[1], lip_r, -lip_push, 0)
        # Right corner moves strictly RIGHT
        result = local_translate(result, mr[0], mr[1], lip_r, lip_push, 0)

    # ── FACE SLIMMING ─────────────────────────────────────────────────────
    if slim_intensity > 0:
        jaw_left_pt  = src[234]
        jaw_right_pt = src[454]
        nose = src[4]
        
        slim_r = face_w * 0.4
        slim_push = slim_intensity * face_w * 0.08

        # Calculate vector pointing from left jaw TOWARDS the nose
        vl_x, vl_y = nose[0] - jaw_left_pt[0], nose[1] - jaw_left_pt[1]
        norm_l = math.hypot(vl_x, vl_y) + 1e-6
        result = local_translate(result, jaw_left_pt[0], jaw_left_pt[1], slim_r, 
                                 (vl_x/norm_l)*slim_push, (vl_y/norm_l)*slim_push)

        # Calculate vector pointing from right jaw TOWARDS the nose
        vr_x, vr_y = nose[0] - jaw_right_pt[0], nose[1] - jaw_right_pt[1]
        norm_r = math.hypot(vr_x, vr_y) + 1e-6
        result = local_translate(result, jaw_right_pt[0], jaw_right_pt[1], slim_r, 
                                 (vr_x/norm_r)*slim_push, (vr_y/norm_r)*slim_push)

        # Repeat for the lower cheeks
        lc2, rc2 = src[172], src[397]
        vlc_x, vlc_y = nose[0] - lc2[0], nose[1] - lc2[1]
        norm_lc = math.hypot(vlc_x, vlc_y) + 1e-6
        result = local_translate(result, lc2[0], lc2[1], slim_r * 0.7, 
                                 (vlc_x/norm_lc)*slim_push*0.6, (vlc_y/norm_lc)*slim_push*0.6)

        vrc_x, vrc_y = nose[0] - rc2[0], nose[1] - rc2[1]
        norm_rc = math.hypot(vrc_x, vrc_y) + 1e-6
        result = local_translate(result, rc2[0], rc2[1], slim_r * 0.7, 
                                 (vrc_x/norm_rc)*slim_push*0.6, (vrc_y/norm_rc)*slim_push*0.6)

    final = np.clip(result, 0, 255).astype(np.uint8)

    return jsonify({"result": encode_image(final)})


# ── ROUTE: /api/makeup ────────────────────────────────────────────────────────

def apply_region_color(img, pts, region_indices, color_bgr, opacity,
                       blur_sigma=None, expand_px=0):
    """
    Blend a solid color into a landmark region with soft edges.
    img: float32 BGR
    Returns float32 BGR
    """
    h, w = img.shape[:2]
    rpts = np.array([pts[i] for i in region_indices if i < len(pts)], dtype=np.int32)
    if len(rpts) < 3:
        return img
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillConvexPoly(mask, rpts, 1.0)
    if expand_px > 0:
        k = expand_px * 2 + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    if blur_sigma and blur_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
    color_layer = np.full_like(img, color_bgr, dtype=np.float32)
    alpha = mask[:, :, np.newaxis] * opacity
    return img * (1 - alpha) + color_layer * alpha


def hex_or_rgb(val):
    """Accept [R,G,B] list, return BGR tuple."""
    r, g, b = val
    return (b, g, r)


@app.route("/api/makeup", methods=["POST"])
def makeup():
    data = request.get_json()
    img  = decode_image(data["image"])
    h, w = img.shape[:2]

    pts = get_landmarks(img)
    if pts is None:
        return jsonify({"error": "No face detected"}), 400

    face_w = abs(pts[454][0] - pts[234][0]) if len(pts) > 454 else w
    face_h = abs(pts[152][1] - pts[10][1])  if len(pts) > 152 else h

    result = img.copy().astype(np.float32)

    # ── 1. FOUNDATION: bilateral smooth + optional color tint ──────────────
    foundation = int(data.get("foundation", 0))
    foundation_color = data.get("foundation_color", None)  # [R,G,B] or None
    if foundation > 0:
        strength = foundation / 100.0
        sigma    = 20 + strength * 40
        smooth   = cv2.bilateralFilter(img, d=9, sigmaColor=sigma, sigmaSpace=sigma)
        contour_idx = [10,338,297,332,284,251,389,356,454,323,361,288,
                       397,365,379,378,400,377,152,148,176,149,150,136,
                       172,58,132,93,234,127,162,21,54,103,67,109,10]
        cpts = np.array([pts[i] for i in contour_idx if i < len(pts)], dtype=np.int32)
        fmask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(fmask, [cpts], 1.0)
        fmask = cv2.GaussianBlur(fmask, (0, 0), max(2.0, face_w * 0.03))
        fmask3 = np.stack([fmask] * 3, axis=2)
        blend  = smooth.astype(np.float32) * strength + img.astype(np.float32) * (1 - strength)
        result = result * (1 - fmask3 * strength) + blend * (fmask3 * strength)
        # Optional color tint on top of smoothed skin
        if foundation_color:
            f_bgr = hex_or_rgb(foundation_color)
            color_layer = np.full_like(result, f_bgr, dtype=np.float32)
            tint_alpha = strength * 0.30  # subtle tint, not full cover
            result = result * (1 - fmask3 * tint_alpha) + color_layer * (fmask3 * tint_alpha)
        result = np.clip(result, 0, 255)

    # ── 2. LIP COLOR ─────────────────────────────────────────────────────
    lip_opacity = int(data.get("lip_opacity", 60))
    lip_color   = data.get("lip_color", [192, 50, 74])
    lip_bgr     = hex_or_rgb(lip_color)

    # Upper lip + lower lip landmark indices (MediaPipe)
    UPPER_LIP = [61,185,40,39,37,0,267,269,270,409,291,308,415,310,311,312,13,82,81,80,191,78]
    LOWER_LIP = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,78]

    if lip_opacity > 0:
        alpha = lip_opacity / 100.0
        blur_s = max(1.5, face_w * 0.012)
        result = apply_region_color(result, pts, UPPER_LIP, lip_bgr, alpha, blur_sigma=blur_s)
        result = apply_region_color(result, pts, LOWER_LIP, lip_bgr, alpha, blur_sigma=blur_s)

    # ── 3. LIP GLOSS: specular highlight on center of lips ───────────────
    lip_gloss = int(data.get("lip_gloss", 0))
    if lip_gloss > 0:
        gloss_alpha = lip_gloss / 100.0 * 0.55
        gloss_layer = np.zeros((h, w), dtype=np.float32)
        # Center of lips
        lcx = (pts[13][0] + pts[14][0]) // 2
        lcy = (pts[13][1] + pts[14][1]) // 2
        cv2.ellipse(gloss_layer, (lcx, lcy),
                    (int(face_w * 0.08), int(face_h * 0.025)),
                    0, 0, 360, 1.0, -1)
        gloss_layer = cv2.GaussianBlur(gloss_layer, (0, 0), max(2.0, face_w * 0.015))
        gloss3 = np.stack([gloss_layer] * 3, axis=2)
        result = result + gloss3 * (gloss_alpha * 80)
        result = np.clip(result, 0, 255)

    # ── 4. EYELINER ───────────────────────────────────────────────────────
    eyeliner = int(data.get("eyeliner", 0))
    if eyeliner > 0:
        liner_alpha = eyeliner / 100.0
        liner_layer = np.zeros((h, w), dtype=np.float32)
        # Upper lash line landmarks
        L_UPPER_LASH = [33,246,161,160,159,158,157,173,133]
        R_UPPER_LASH = [362,398,384,385,386,387,388,466,263]
        for lash_set in [L_UPPER_LASH, R_UPPER_LASH]:
            lpts = np.array([pts[i] for i in lash_set if i < len(pts)], dtype=np.int32)
            cv2.polylines(liner_layer, [lpts], False, 1.0,
                          max(1, int(face_w * 0.007)))
        liner_layer = cv2.GaussianBlur(liner_layer, (0, 0), max(1.0, face_w * 0.006))
        liner3 = np.stack([liner_layer] * 3, axis=2)
        dark   = np.zeros_like(result)
        result = result * (1 - liner3 * liner_alpha * 0.85) + dark * (liner3 * liner_alpha * 0.85)
        result = np.clip(result, 0, 255)

    # ── 5. EYE COLOR (IRIS) — contact lens effect ─────────────────────────
    eye_color     = int(data.get("eye_color", 0))
    eye_color_rgb = data.get("eye_color_rgb", [100, 149, 237])
    eye_color_bgr = hex_or_rgb(eye_color_rgb)
    if eye_color > 0:
        strength = eye_color / 100.0 * 0.55  # max 55% — preserves iris texture

        for inner_idx, outer_idx, top_idx, bot_idx in [
            (133, 33,  159, 145),  # left eye
            (362, 263, 386, 374),  # right eye
        ]:
            if max(inner_idx, outer_idx, top_idx, bot_idx) >= len(pts):
                continue
            ix, iy = pts[inner_idx]; ox, oy = pts[outer_idx]
            tx, ty = pts[top_idx];   bx, by = pts[bot_idx]
            cx = (ix + ox) // 2
            cy = (ty + by) // 2
            # Iris is ~60% of eye width, ~80% of eye height
            iris_w = max(3, int(abs(ox - ix) * 0.28))
            iris_h = max(3, int(abs(by - ty) * 0.42))

            # Build soft iris mask
            iris_mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(iris_mask, (cx, cy), (iris_w, iris_h), 0, 0, 360, 1.0, -1)

            # Pupil mask (darker center — don't colorize)
            pupil_mask = np.zeros((h, w), dtype=np.float32)
            pw = max(2, int(iris_w * 0.42))
            ph = max(2, int(iris_h * 0.42))
            cv2.ellipse(pupil_mask, (cx, cy), (pw, ph), 0, 0, 360, 1.0, -1)
            pupil_mask = cv2.GaussianBlur(pupil_mask, (0,0), max(1.0, pw * 0.4))

            # Soft iris edge
            iris_mask = cv2.GaussianBlur(iris_mask, (0,0), max(1.0, iris_w * 0.18))

            # Remove pupil from iris zone
            iris_only = np.clip(iris_mask - pupil_mask * 0.9, 0, 1)
            iris3 = np.stack([iris_only] * 3, axis=2)

            # Multiply blend: tints the existing iris texture instead of painting over it
            color_norm = np.array([eye_color_bgr[0]/255, eye_color_bgr[1]/255, eye_color_bgr[2]/255], dtype=np.float32)
            for c in range(3):
                orig_ch   = result[:, :, c] / 255.0
                tinted_ch = orig_ch * (1 - strength) + orig_ch * color_norm[c] * strength * 2.2
                result[:, :, c] = np.clip(
                    result[:, :, c] * (1 - iris_only * strength) +
                    tinted_ch * 255 * iris_only * strength,
                    0, 255
                )

        result = np.clip(result, 0, 255)

    # ── 6. BLUSH ──────────────────────────────────────────────────────────
    blush       = int(data.get("blush", 0))
    blush_color = data.get("blush_color", [224, 96, 128])
    blush_bgr   = hex_or_rgb(blush_color)
    if blush > 0:
        alpha  = blush / 100.0 * 0.35
        blur_s = max(4.0, face_w * 0.06)
        # Cheek landmark centers (approx)
        for cheek_idx in [205, 425]:
            cx, cy = pts[cheek_idx]
            blush_layer = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(blush_layer, (cx, cy),
                        (int(face_w * 0.16), int(face_h * 0.12)),
                        0, 0, 360, 1.0, -1)
            blush_layer = cv2.GaussianBlur(blush_layer, (0, 0), blur_s)
            blush3 = np.stack([blush_layer] * 3, axis=2)
            color_layer = np.full_like(result, blush_bgr, dtype=np.float32)
            result = result * (1 - blush3 * alpha) + color_layer * (blush3 * alpha)
        result = np.clip(result, 0, 255)

    # ── 7. HIGHLIGHTER: nose bridge + brow bone + cupid's bow ────────────
    highlight = int(data.get("highlight", 0))
    if highlight > 0:
        hl_alpha = highlight / 100.0 * 0.45
        hl_layer = np.zeros((h, w), dtype=np.float32)
        # Nose bridge
        for idx in [6, 197, 195, 5]:
            cx, cy = pts[idx]
            cv2.circle(hl_layer, (cx, cy), int(face_w * 0.025), 1.0, -1)
        # Brow bone
        for idx in [107, 336]:
            cx, cy = pts[idx]
            cv2.ellipse(hl_layer, (cx, cy),
                        (int(face_w * 0.05), int(face_h * 0.025)),
                        0, 0, 360, 1.0, -1)
        # Cupid's bow
        for idx in [37, 267]:
            cx, cy = pts[idx]
            cv2.circle(hl_layer, (cx, cy), int(face_w * 0.015), 1.0, -1)
        hl_layer = cv2.GaussianBlur(hl_layer, (0, 0), max(2.0, face_w * 0.025))
        hl3 = np.stack([hl_layer] * 3, axis=2)
        white = np.full_like(result, 255.0)
        result = result * (1 - hl3 * hl_alpha) + white * (hl3 * hl_alpha)
        result = np.clip(result, 0, 255)

    final = np.clip(result, 0, 255).astype(np.uint8)
    return jsonify({"result": encode_image(final)})


# ── ROUTE: /api/aging ─────────────────────────────────────────────────────────

@app.route("/api/aging", methods=["POST"])
def aging():
    data = request.get_json()
    user_prompt = str(data.get("prompt", "")).strip()
    prompt_only = bool(data.get("promptOnly", False)) and bool(user_prompt)

    if prompt_only:
        if not GEMINI_API_KEY:
            return jsonify({"error": "Gemini API key not set. Export GEMINI_API_KEY before starting backend.py."}), 400
        ai_prompt = (
            f"Edit the provided portrait according to this instruction: {user_prompt}. "
            f"The instruction may be in Turkish or English. Apply the requested visual change clearly and realistically. "
            f"Keep the same person, identity, face shape, pose, clothing, lighting, and background unless explicitly asked to change them. "
            f"STRICT RULES that override all other instructions: "
            f"do NOT change the person's gender presentation; "
            f"do NOT add facial hair if the person does not already have it; "
            f"do NOT alter bone structure, jaw shape, or brow ridge; "
            f"if the subject looks female they must remain female, if male they must remain male. "
            f"Do not return an unchanged image."
        )
        try:
            result = edit_image_with_gemini(data["image"], ai_prompt)
            if result:
                return jsonify({"result": result, "ai": True, "provider": "gemini"})
            return jsonify({"error": "Gemini did not return an edited image."}), 502
        except Exception as e:
            print(f"Gemini prompt edit failed: {e}", flush=True)
            return jsonify({"error": f"Gemini prompt edit failed: {e}"}), 502

    img = decode_image(data["image"])
    mode = data.get("mode", "aging")
    intensity = float(data.get("intensity", 50)) / 100.0

    h, w = img.shape[:2]
    
    # 1. Get Landmarks & Mask First (Crucial to protect background)
    pts = get_landmarks(img)
    if pts is None:
        return jsonify({"error": "No face detected"}), 400
        
    result = img.copy().astype(np.float32)

    # 2. Handle Sagging via Landmarks (Not global image shifting)
    if mode == "aging" and intensity > 0:
        src = list(pts)
        dst = [list(p) for p in pts]
        sag_amount = int(intensity * h * 0.015)
        
        # Pull down jaw and lower cheeks
        for idx in JAW_LEFT + JAW_RIGHT + [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 323, 361, 288, 397, 365, 379, 378, 400, 377]:
            if idx < len(dst):
                dst[idx][1] += sag_amount
                
        # Pull down eye bags slightly
        for idx in [253, 254, 252, 256, 340, 23, 24, 22, 26, 110]:
             if idx < len(dst):
                dst[idx][1] += int(sag_amount * 0.4)
                
        src_pts = [(p[0], p[1]) for p in src]
        dst_pts = [(p[0], p[1]) for p in dst]
        result = delaunay_warp(result.astype(np.uint8), src_pts, dst_pts).astype(np.float32)

    if mode == "aging":
        # ── Step 1: Wrinkle texture via high-pass sharpening 
        blur_light = cv2.GaussianBlur(result, (0,0), 1.5)
        blur_heavy = cv2.GaussianBlur(result, (0,0), 5.0)
        hf = result - blur_heavy
        wrinkle_strength = intensity * 3.5
        result = result + hf * wrinkle_strength

        # ── Step 2: Skin texture noise (Fixed to look like pores/horizontal lines)
        noise = np.random.normal(0, intensity * 15, result.shape)
        h_noise = np.zeros_like(result)
        # Shift along axis=1 (horizontal) for forehead lines, and blur slightly
        for x_offset in range(-2, 3):
            shifted = np.roll(noise, x_offset, axis=1)
            h_noise += shifted * (1 - abs(x_offset)/3)
        h_noise = cv2.GaussianBlur(h_noise, (3, 1), 0) # soften the noise
        result = result + h_noise * 0.5

        # ── Step 3: Color grading 
        result = np.clip(result, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] *= max(0, 1 - intensity * 0.45)   
        hsv[:,:,2] *= max(0, 1 - intensity * 0.08)   
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        tint = result.astype(np.float32)
        tint[:,:,0] *= (1 - intensity * 0.12)  
        tint[:,:,2] *= (1 + intensity * 0.08)  
        result = np.clip(tint, 0, 255).astype(np.uint8)

        # ── Step 4: Contrast increase
        result = cv2.convertScaleAbs(result, alpha=1.0 + intensity * 0.25, beta=-intensity * 12)

    else:  # de-aging
        # ── Step 1: Bilateral filter
        passes = max(1, int(intensity * 4))
        d = 9
        sigma = 40 + intensity * 60
        result_uint = result.astype(np.uint8)
        for _ in range(passes):
            result_uint = cv2.bilateralFilter(result_uint, d, sigma, sigma)
        result = result_uint.astype(np.float32)

        # ── Step 2: Edge-preserved smoothing
        blur = cv2.GaussianBlur(result, (0,0), 2.0 + intensity * 3)
        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        edges = cv2.Laplacian(gray_orig, cv2.CV_32F)
        edge_mask = np.abs(edges)
        edge_mask = cv2.GaussianBlur(edge_mask, (0,0), 3)
        edge_mask = np.clip(edge_mask / (edge_mask.max() + 1e-6), 0, 1)
        edge_mask = np.stack([edge_mask]*3, axis=2)
        result = result * edge_mask + blur * (1 - edge_mask)

        result = np.clip(result, 0, 255).astype(np.uint8)

        # ── Step 3: Color grading 
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + intensity * 0.35), 0, 255)  
        hsv[:,:,2] = np.clip(hsv[:,:,2] * (1 + intensity * 0.12), 0, 255)  
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        tint = result.astype(np.float32)
        tint[:,:,0] = np.clip(tint[:,:,0] * (1 + intensity*0.06), 0, 255)  
        tint[:,:,2] = np.clip(tint[:,:,2] * (1 - intensity*0.04), 0, 255)  
        result = np.clip(tint, 0, 255).astype(np.uint8)
        
        result = cv2.convertScaleAbs(result, alpha=1.0 - intensity * 0.1, beta=intensity * 8)

    # ── Final Step: Apply Facial Mask ─────────────────────────────────────
    face_contour_idx = [10,338,297,332,284,251,389,356,454,323,361,288,
                        397,365,379,378,400,377,152,148,176,149,150,136,
                        172,58,132,93,234,127,162,21,54,103,67,109,10]
    contour_pts = np.array([pts[i] for i in face_contour_idx if i < len(pts)], dtype=np.int32)

    face_h = abs(pts[152][1] - pts[10][1]) if len(pts) > 152 else h
    chin_y = pts[152][1] if len(pts) > 152 else h // 2
    left_x = pts[234][0] if len(pts) > 234 else int(w * 0.25)
    right_x = pts[454][0] if len(pts) > 454 else int(w * 0.75)
    extend_y = chin_y + int(face_h * 0.22)
    beard_pts = np.array([[left_x, extend_y],[w//2, extend_y],[right_x, extend_y]], dtype=np.int32)
    contour_pts = np.vstack([contour_pts, beard_pts])

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [contour_pts], 1.0)
    
    # Exclude eyes from being overly blurred/wrinkled
    for eye in [L_EYE, R_EYE]:
        eye_pts = np.array([pts[i] for i in eye if i < len(pts)], dtype=np.int32)
        cv2.fillPoly(mask, [eye_pts], 0.0)

    mask = cv2.GaussianBlur(mask, (51, 51), 20)
    mask3 = np.stack([mask]*3, axis=2)
    
    final_result = (result.astype(np.float32) * mask3 +
                    img.astype(np.float32) * (1 - mask3))
    final_result = np.clip(final_result, 0, 255).astype(np.uint8)

    # ── AI enhancement via fal-ai/flux-pro/kontext ────────────────────────
    image_b64 = data["image"]
    # Detect gender upfront so the model cannot misread it
    detected_gender = detect_apparent_gender(data["image"])
    if detected_gender == "female":
        gender_anchor = "The person in this photo is a WOMAN. She must remain a woman."
        no_hair_rule  = "Do NOT add any facial hair, stubble, beard, or moustache — this is a woman."
    elif detected_gender == "male":
        gender_anchor = "The person in this photo is a MAN. He must remain a man."
        no_hair_rule  = "Preserve existing facial hair exactly as shown."
    else:
        gender_anchor = "Preserve the person's gender exactly as shown."
        no_hair_rule  = "Do NOT add facial hair if none is present."

    years = max(1, int(intensity * 40))
    if mode == "aging":
        ai_prompt = (
            f"{gender_anchor} "
            f"Age this person's face by {years} years using ONLY surface-level skin changes: "
            f"fine lines and wrinkles on the forehead, around the eyes (crow's feet), and nasolabial folds; "
            f"age spots; greying or whitening of the hair. "
            f"STRICT RULES: "
            f"(1) Do NOT change facial bone structure, jaw shape, brow ridge, or skull shape. "
            f"(2) {no_hair_rule} "
            f"(3) Do NOT make feminine features look masculine or vice versa. "
            f"(4) Do NOT change skin tone, ethnicity, eye colour, or nose shape. "
            f"(5) Keep background, clothing, lighting, and pose completely unchanged. "
            f"Only wrinkles, age spots, and hair greying are permitted."
        )
    else:
        younger = max(1, int(intensity * 20))
        ai_prompt = (
            f"{gender_anchor} "
            f"Make this person look {younger} years younger using ONLY surface-level skin changes: "
            f"smooth skin texture, remove wrinkles and fine lines, restore natural hair colour, "
            f"brighten the eyes, add youthful plumpness to cheeks. "
            f"STRICT RULES: "
            f"(1) Do NOT change facial bone structure, jaw shape, brow ridge, or skull shape. "
            f"(2) {no_hair_rule} "
            f"(3) Do NOT change skin tone, ethnicity, eye colour, or nose shape. "
            f"(4) Keep background, clothing, lighting, and pose completely unchanged. "
            f"Only skin smoothing, hair colour restoration, and eye brightening are permitted."
        )

    try:
        img_url = upload_image_to_fal(image_b64) or image_b64
        resp = requests.post(
            "https://fal.run/fal-ai/flux-pro/kontext",
            headers=FAL_HEADERS,
            json={"image_url": img_url, "prompt": ai_prompt,
                  "num_images": 1, "output_format": "jpeg"},
            timeout=90,
        )
        d = resp.json()
        if "images" in d and d["images"]:
            dl = requests.get(d["images"][0]["url"], timeout=30)
            arr = np.frombuffer(dl.content, np.uint8)
            ai_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if ai_img is not None:
                return jsonify({"result": encode_image(ai_img), "ai": True})
    except Exception as e:
        print(f"Fal aging failed, using local result: {e}")

    # Fallback: return local CV result
    return jsonify({"result": encode_image(final_result), "ai": False})


# ── ROUTE: /api/golden_ratio ──────────────────────────────────────────────────

PHI = (1 + math.sqrt(5)) / 2  # 1.618...

@app.route("/api/golden_ratio", methods=["POST"])
def golden_ratio():
    data = request.get_json()
    img = decode_image(data["image"])
    h, w = img.shape[:2]

    pts = get_landmarks(img)
    if pts is None:
        return jsonify({"error": "No face detected"}), 400

    def pt(idx): return pts[idx]

    # ── Key measurements ──────────────────────────────────────────────────
    face_left  = pt(234); face_right = pt(454)
    face_width = math.dist(face_left, face_right)

    forehead_pt = pt(FOREHEAD); chin_pt = pt(CHIN)
    face_height = math.dist(forehead_pt, chin_pt)

    nose_left = pt(129); nose_right = pt(358)
    nose_width = math.dist(nose_left, nose_right)

    mouth_l = pt(MOUTH_LEFT); mouth_r = pt(MOUTH_RIGHT)
    mouth_width = math.dist(mouth_l, mouth_r)

    leye_l = pt(33);  leye_r = pt(133)
    reye_l = pt(362); reye_r = pt(263)
    left_eye_w  = math.dist(leye_l, leye_r)
    right_eye_w = math.dist(reye_l, reye_r)
    avg_eye_w = (left_eye_w + right_eye_w) / 2

    l_pupil = pt(468) if len(pts) > 468 else ((leye_l[0]+leye_r[0])//2, (leye_l[1]+leye_r[1])//2)
    r_pupil = pt(473) if len(pts) > 473 else ((reye_l[0]+reye_r[0])//2, (reye_l[1]+reye_r[1])//2)
    pupil_dist = math.dist(l_pupil, r_pupil)

    nose_base = pt(2)
    chin_to_mouth = math.dist(chin_pt, mouth_l)
    mouth_to_nose  = math.dist(mouth_l, nose_base)
    nose_to_eyes   = math.dist(nose_base, pt(168))
    upper_face     = math.dist(pt(168), forehead_pt)

    # ── Golden ratio scores ───────────────────────────────────────────────
    scores = {}

    def ratio_score(a, b, label):
        if b == 0: return
        r = a / b
        ideal = PHI
        deviation = abs(r - ideal) / ideal
        pct = max(0, round((1 - deviation) * 100, 1))
        scores[label] = {"ratio": round(r, 3), "ideal": round(ideal, 3), "score": pct}

    ratio_score(face_height, face_width,   "Face Height / Width")
    ratio_score(face_width, nose_width * 3,"Face Width / (3 × Nose Width)")
    ratio_score(face_width, mouth_width * PHI, "Mouth × φ / Face Width")
    ratio_score(pupil_dist, nose_width,    "Pupil Distance / Nose Width")
    ratio_score(avg_eye_w,  nose_width,    "Eye Width / Nose Width")

    overall = round(sum(v["score"] for v in scores.values()) / len(scores), 1)

    # ── Draw overlay image ────────────────────────────────────────────────
    overlay = img.copy()

    fx, fy = face_left[0], forehead_pt[1]
    fw, fh_px = int(face_width), int(face_height)

    cv2.rectangle(overlay, (fx, fy), (fx+fw, fy+fh_px), (200, 160, 80), 1)

    line_y1 = fy + int(fh_px / PHI)
    line_y2 = fy + int(fh_px - fh_px / PHI)
    cv2.line(overlay, (fx, line_y1), (fx+fw, line_y1), (200, 160, 80), 1)
    cv2.line(overlay, (fx, line_y2), (fx+fw, line_y2), (200, 160, 80), 1)

    line_x1 = fx + int(fw / PHI)
    line_x2 = fx + int(fw - fw / PHI)
    cv2.line(overlay, (line_x1, fy), (line_x1, fy+fh_px), (200, 160, 80), 1)
    cv2.line(overlay, (line_x2, fy), (line_x2, fy+fh_px), (200, 160, 80), 1)

    key_pts = [face_left, face_right, forehead_pt, chin_pt,
               nose_left, nose_right, mouth_l, mouth_r,
               leye_l, leye_r, reye_l, reye_r]
    for p in key_pts:
        cv2.circle(overlay, p, 3, (79, 200, 150), -1)

    color = (60,180,100) if overall >= 80 else (60,130,200) if overall >= 60 else (60,80,200)
    cv2.putText(overlay, f"Golden Ratio Score: {overall}%",
                (fx, max(fy-10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)

    return jsonify({
        "scores": scores,
        "overall": overall,
        "phi": round(PHI, 4),
        "overlay": encode_image(overlay),
        "measurements": {
            "face_width":   round(face_width, 1),
            "face_height":  round(face_height, 1),
            "nose_width":   round(nose_width, 1),
            "mouth_width":  round(mouth_width, 1),
            "pupil_dist":   round(pupil_dist, 1),
            "avg_eye_width":round(avg_eye_w, 1),
        }
    })

# ── ROUTE: /api/golden_warp ───────────────────────────────────────────────────

def _compute_ideal_targets(face_width, face_height, nose_width, mouth_width, avg_eye_w):
    target_face_width  = face_height / PHI
    target_nose_width  = target_face_width / 4.0
    target_mouth_width = target_nose_width * 1.5
    target_eye_width   = target_nose_width * 1.0
    return target_face_width, target_nose_width, target_mouth_width, target_eye_width

@app.route("/api/golden_warp", methods=["POST"])
def golden_warp():
    data = request.get_json()
    img = decode_image(data["image"])
    h, w = img.shape[:2]

    pts = get_landmarks(img)
    if pts is None:
        return jsonify({"error": "No face detected"}), 400

    def pt(idx): return pts[idx]

    face_left, face_right = pt(234), pt(454)
    forehead_pt, chin_pt  = pt(FOREHEAD), pt(CHIN)
    nose_left, nose_right = pt(NOSE_LEFT_IDX), pt(NOSE_RIGHT_IDX)
    mouth_l, mouth_r      = pt(MOUTH_LEFT), pt(MOUTH_RIGHT)
    leye_l, leye_r = pt(33), pt(133)
    reye_l, reye_r = pt(362), pt(263)

    face_width  = math.dist(face_left, face_right)
    face_height = math.dist(forehead_pt, chin_pt)
    nose_width  = math.dist(nose_left, nose_right)
    mouth_width = math.dist(mouth_l, mouth_r)
    avg_eye_w   = (math.dist(leye_l, leye_r) + math.dist(reye_l, reye_r)) / 2

    t_face_w, t_nose_w, t_mouth_w, t_eye_w = _compute_ideal_targets(
        face_width, face_height, nose_width, mouth_width, avg_eye_w)

    raw_face_scale  = t_face_w  / face_width  if face_width  > 0 else 1.0
    raw_nose_scale  = t_nose_w  / nose_width  if nose_width  > 0 else 1.0
    raw_mouth_scale = t_mouth_w / mouth_width if mouth_width > 0 else 1.0

    MAX_FACE_DEFORM  = 0.18
    MAX_NOSE_DEFORM  = 0.14
    MAX_MOUTH_DEFORM = 0.15

    face_scale  = max(1-MAX_FACE_DEFORM,  min(1+MAX_FACE_DEFORM,  raw_face_scale))
    nose_scale  = max(1-MAX_NOSE_DEFORM,  min(1+MAX_NOSE_DEFORM,  raw_nose_scale))
    mouth_scale = max(1-MAX_MOUTH_DEFORM, min(1+MAX_MOUTH_DEFORM, raw_mouth_scale))

    face_pinch  = 1.0 - face_scale
    nose_pinch  = 1.0 - nose_scale
    mouth_pinch = 1.0 - mouth_scale

    cx = (face_left[0] + face_right[0]) / 2.0
    cy = (forehead_pt[1] + chin_pt[1]) / 2.0
    nose_cx = (nose_left[0] + nose_right[0]) / 2.0
    nose_cy = (nose_left[1] + nose_right[1]) / 2.0
    mouth_cx = (mouth_l[0] + mouth_r[0]) / 2.0
    mouth_cy = (mouth_l[1] + mouth_r[1]) / 2.0

    face_radius  = face_width * 0.62
    nose_radius  = nose_width * 0.9
    mouth_radius = mouth_width * 0.75

    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))

    def radial_falloff(center_x, center_y, radius, power=2.0):
        ddx = map_x - center_x
        ddy = map_y - center_y
        dist = np.sqrt(ddx * ddx + ddy * ddy)
        f = np.zeros_like(dist)
        inside = dist < radius
        f[inside] = (np.cos(np.pi * dist[inside] / (2 * radius))) ** power
        return ddx, f

    face_dx, face_f   = radial_falloff(cx, cy, face_radius)
    nose_dx, nose_f   = radial_falloff(nose_cx, nose_cy, nose_radius)
    mouth_dx, mouth_f = radial_falloff(mouth_cx, mouth_cy, mouth_radius)

    src_x = (map_x
             + face_dx  * (face_pinch  * face_f)
             + nose_dx  * (nose_pinch  * nose_f)
             + mouth_dx * (mouth_pinch * mouth_f))
    src_y = map_y

    map_x_final = np.clip(src_x, 0, w - 1).astype(np.float32)
    map_y_final = np.clip(src_y, 0, h - 1).astype(np.float32)

    result = cv2.remap(img, map_x_final, map_y_final, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)

    def zone_strength(x, y, center_x, center_y, radius, pinch, power=2.0):
        ddx, ddy = x - center_x, y - center_y
        d = math.hypot(ddx, ddy)
        f = (math.cos(math.pi * d / (2 * radius)) ** power) if d < radius else 0.0
        return ddx, pinch * f

    src_pts = [(float(p[0]), float(p[1])) for p in pts]
    dst_pts = []
    for (x, y) in src_pts:
        fdx, fs = zone_strength(x, y, cx, cy, face_radius, face_pinch)
        ndx, ns = zone_strength(x, y, nose_cx, nose_cy, nose_radius, nose_pinch)
        mdx, ms = zone_strength(x, y, mouth_cx, mouth_cy, mouth_radius, mouth_pinch)
        new_x = x - (fdx * fs + ndx * ns + mdx * ms)
        dst_pts.append((new_x, y))

    def dpt(idx): return dst_pts[idx]
    t_nfw = math.dist(dpt(234), dpt(454))
    t_nfh = math.dist(dpt(FOREHEAD), dpt(CHIN))
    t_nnw = math.dist(dpt(NOSE_LEFT_IDX), dpt(NOSE_RIGHT_IDX))
    t_nmw = math.dist(dpt(MOUTH_LEFT), dpt(MOUTH_RIGHT))
    t_nlw = math.dist(dpt(33), dpt(133))
    t_nrw = math.dist(dpt(362), dpt(263))
    t_eye_w2 = (t_nlw + t_nrw) / 2

    def score(a, b, ideal):
        if b == 0: return 0
        r = a / b
        dev = abs(r - ideal) / ideal
        return max(0, round((1 - dev) * 100, 1))

    target_scores = {
        "Face Height / Width":     score(t_nfh, t_nfw, PHI),
        "Eye Width / Nose Width":  score(t_eye_w2, t_nnw, 1.0),
        "Face Width / Nose Width": score(t_nfw, t_nnw, 4.0),
        "Mouth Width / Nose Width":score(t_nmw, t_nnw, 1.5),
    }
    target_overall = round(sum(target_scores.values()) / len(target_scores), 1)

    new_pts = get_landmarks(result)
    new_scores, new_overall = {}, None
    if new_pts is not None:
        def npt(idx): return new_pts[idx]
        nfw = math.dist(npt(234), npt(454))
        nfh = math.dist(npt(FOREHEAD), npt(CHIN))
        nnw = math.dist(npt(NOSE_LEFT_IDX), npt(NOSE_RIGHT_IDX))
        nmw = math.dist(npt(MOUTH_LEFT), npt(MOUTH_RIGHT))
        nlw = math.dist(npt(33), npt(133))
        nrw = math.dist(npt(362), npt(263))
        new_eye_w = (nlw + nrw) / 2
        new_scores = {
            "Face Height / Width":     score(nfh, nfw, PHI),
            "Eye Width / Nose Width":  score(new_eye_w, nnw, 1.0),
            "Face Width / Nose Width": score(nfw, nnw, 4.0),
            "Mouth Width / Nose Width":score(nmw, nnw, 1.5),
        }
        new_overall = round(sum(new_scores.values()) / len(new_scores), 1)

    return jsonify({
        "result":          encode_image(result),
        "new_scores":      target_scores,
        "new_overall":     target_overall,
        "measured_scores": new_scores,
        "measured_overall":new_overall,
    })

# ── ROUTE: /api/frequency ─────────────────────────────────────────────────────

@app.route("/api/frequency", methods=["POST"])
def frequency():
    data = request.get_json()
    img = decode_image(data["image"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spectrum_colored = cv2.applyColorMap(magnitude, cv2.COLORMAP_INFERNO)

    h, w = spectrum_colored.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(spectrum_colored, (cx, 0), (cx, h), (255, 255, 255), 1)
    cv2.line(spectrum_colored, (0, cy), (w, cy), (255, 255, 255), 1)

    mag_abs = np.abs(fshift)
    total_energy = float(np.sum(mag_abs))
    dc_energy    = float(mag_abs[cy, cx])
    dc_ratio     = round(dc_energy / (total_energy + 1e-9) * 100, 2)

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_r    = min(cx, cy)
    hf_mask  = dist_from_center > (max_r * 0.10)
    hf_energy = float(np.sum(mag_abs * hf_mask))
    lf_energy = float(np.sum(mag_abs * ~hf_mask))

    return jsonify({
        "spectrum":     encode_image(spectrum_colored),
        "total_energy": round(total_energy, 0),
        "hf_energy":    round(hf_energy, 0),
        "lf_energy":    round(lf_energy, 0),
        "hflf_ratio":   round(hf_energy / (lf_energy + 1e-9), 4),
        "stats": {
            "dc_energy_pct": dc_ratio,
            "high_freq_pct": round(hf_energy / (total_energy + 1e-9) * 100, 2),
            "low_freq_pct":  round(lf_energy  / (total_energy + 1e-9) * 100, 2),
            "image_size":    f"{w}×{h}",
        }
    })

# ── ROUTE: /api/evaluation ────────────────────────────────────────────────────

@app.route("/api/evaluation", methods=["POST"])
def evaluation():
    data = request.get_json()
    original_src    = data.get("original")
    transformed_src = data.get("transformed")
    if not original_src or not transformed_src:
        return jsonify({"error": "Both 'original' and 'transformed' images required"}), 400

    orig = decode_image(original_src).astype(np.float32)
    tran = decode_image(transformed_src).astype(np.float32)
    if orig.shape != tran.shape:
        tran = cv2.resize(tran, (orig.shape[1], orig.shape[0])).astype(np.float32)

    mse = float(np.mean((orig - tran) ** 2))
    psnr = 100.0 if mse == 0 else round(10 * math.log10((255.0 ** 2) / mse), 2)

    gray_orig = cv2.cvtColor(orig.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_tran = cv2.cvtColor(tran.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    C1 = (0.01 * 255) ** 2; C2 = (0.03 * 255) ** 2; ksize = (11, 11)
    mu1 = cv2.GaussianBlur(gray_orig, ksize, 1.5)
    mu2 = cv2.GaussianBlur(gray_tran, ksize, 1.5)
    mu1_sq = mu1*mu1; mu2_sq = mu2*mu2; mu1_mu2 = mu1*mu2
    sig1_sq = cv2.GaussianBlur(gray_orig*gray_orig, ksize, 1.5) - mu1_sq
    sig2_sq = cv2.GaussianBlur(gray_tran*gray_tran, ksize, 1.5) - mu2_sq
    sig12   = cv2.GaussianBlur(gray_orig*gray_tran, ksize, 1.5) - mu1_mu2
    ssim_map = ((2*mu1_mu2+C1)*(2*sig12+C2)) / ((mu1_sq+mu2_sq+C1)*(sig1_sq+sig2_sq+C2))
    ssim = round(float(np.mean(ssim_map)), 4)

    status = "GOOD" if psnr >= 35 else "MED" if psnr >= 28 else "LOW"
    return jsonify({"mse": round(mse, 2), "psnr": psnr, "ssim": ssim, "status": status})

# ── ROUTE: /api/model/<filename> — serves ML model files ─────────────────────
_ALLOWED_MODELS = {"face_landmarker.task", "hair_segmenter.tflite"}

@app.route("/api/model/<filename>")
def serve_model(filename):
    if filename not in _ALLOWED_MODELS:
        return jsonify({"error": "not found"}), 404
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), filename)


# ── ROUTE: /assets/<path> — serves PNG accessory assets ──────────────────────
@app.route("/assets/<path:filename>")
def serve_asset(filename):
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    return send_from_directory(assets_dir, filename)


# ── Hair segmenter lazy-loader ────────────────────────────────────────────────
_hair_seg = None

def get_hair_segmenter():
    global _hair_seg
    if _hair_seg is not None:
        return _hair_seg
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hair_segmenter.tflite")
    if not os.path.exists(path):
        return None
    try:
        opts = mp_vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            output_category_mask=True,
        )
        _hair_seg = mp_vision.ImageSegmenter.create_from_options(opts)
    except Exception as e:
        print(f"Hair segmenter init failed: {e}")
    return _hair_seg


# ── fal storage upload helper ─────────────────────────────────────────────────
def upload_image_to_fal(b64_data_url):
    """Upload a base64 data URL to fal storage, return a CDN URL or None."""
    try:
        _, b64data = b64_data_url.split(",", 1)
        img_bytes  = base64.b64decode(b64data)
        init = requests.post(
            "https://rest.fal.run/storage/upload/initiate",
            headers={"Authorization": f"Key {FAL_KEY}"},
            json={"content_type": "image/jpeg", "file_name": "snapshot.jpg"},
            timeout=15,
        )
        if init.status_code == 200:
            d = init.json()
            upload_url = d.get("upload_url")
            file_url   = d.get("file_url")
            if upload_url:
                put = requests.put(
                    upload_url, data=img_bytes,
                    headers={"Content-Type": "image/jpeg"}, timeout=30,
                )
                if put.status_code in (200, 204):
                    return file_url
    except Exception as e:
        print(f"fal upload error: {e}")
    return None


# ── ROUTE: /api/hair_color ────────────────────────────────────────────────────
@app.route("/api/hair_color", methods=["POST"])
def hair_color():
    data = request.get_json()
    img  = decode_image(data["image"])
    h, w = img.shape[:2]

    hair_rgb = data.get("hair_color", [40, 20, 10])
    opacity  = float(data.get("opacity", 70)) / 100.0
    hair_bgr = (int(hair_rgb[2]), int(hair_rgb[1]), int(hair_rgb[0]))

    # Try MediaPipe hair segmenter first (much more accurate)
    hair_f = None
    seg = get_hair_segmenter()
    if seg:
        try:
            rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result_s = seg.segment(mp_image)
            cat_mask = result_s.category_mask.numpy_view()  # 0=bg 1=hair
            raw      = (cat_mask == 1).astype(np.float32)
            if raw.shape != (h, w):
                raw = cv2.resize(raw, (w, h))
            hair_f = cv2.GaussianBlur(raw, (0, 0), max(2.0, w * 0.008))
        except Exception as e:
            print(f"Hair segmenter inference failed: {e}")

    # Fallback: landmark-based oval above the face
    if hair_f is None:
        pts = get_landmarks(img)
        if pts is None:
            return jsonify({"error": "No face detected"}), 400
        face_contour_idx = [10,338,297,332,284,251,389,356,454,323,361,288,
                            397,365,379,378,400,377,152,148,176,149,150,136,
                            172,58,132,93,234,127,162,21,54,103,67,109,10]
        face_pts = np.array([pts[i] for i in face_contour_idx if i < len(pts)], dtype=np.int32)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(face_mask, [face_pts], 255)
        face_cx    = pts[10][0]
        face_top_y = pts[10][1]
        face_w_px  = abs(pts[454][0] - pts[234][0]) if len(pts) > 454 else w // 2
        face_h_px  = abs(pts[152][1]  - pts[10][1]) if len(pts) > 152 else h // 2
        head_mask  = np.zeros((h, w), dtype=np.uint8)
        head_cy    = max(0, face_top_y - int(face_h_px * 0.18))
        cv2.ellipse(head_mask, (face_cx, head_cy),
                    (int(face_w_px * 0.62), int(face_h_px * 0.72)),
                    0, 0, 360, 255, -1)
        raw    = cv2.bitwise_and(head_mask, cv2.bitwise_not(face_mask)).astype(np.float32) / 255.0
        hair_f = cv2.GaussianBlur(raw, (0, 0), max(3.0, face_w_px * 0.04))

    result      = img.copy().astype(np.float32)
    color_layer = np.full_like(result, hair_bgr, dtype=np.float32)
    hair_3      = np.stack([hair_f] * 3, axis=2)
    result      = result * (1 - hair_3 * opacity) + color_layer * (hair_3 * opacity)

    return jsonify({"result": encode_image(np.clip(result, 0, 255).astype(np.uint8))})


# ── ROUTE: /api/shop/generate_wigs ───────────────────────────────────────────
@app.route("/api/shop/generate_wigs", methods=["POST"])
def generate_wigs():
    global _wig_cache
    if _wig_cache:
        return jsonify(_wig_cache)

    wig_prompts = {
        "black_straight": "long sleek straight black hair wig, isolated on plain white background, professional product photo, studio lighting, no person, top view",
        "curly_brown":    "voluminous curly natural brown afro hair wig, isolated on plain white background, product photo, studio lighting, no person",
        "bob_blonde":     "short chic blonde bob cut hair wig, isolated on plain white background, professional product photo, studio lighting, no person",
        "wavy_red":       "long wavy auburn red hair wig, isolated on plain white background, professional product photo, studio lighting, no person",
    }

    results = {}
    for wig_id, prompt in wig_prompts.items():
        try:
            resp = requests.post(
                "https://fal.run/fal-ai/flux/schnell",
                headers=FAL_HEADERS,
                json={"prompt": prompt, "image_size": "portrait_4_3",
                      "num_inference_steps": 4, "num_images": 1},
                timeout=45
            )
            d = resp.json()
            if "images" in d:
                results[wig_id] = d["images"][0]["url"]
        except Exception as e:
            print(f"Wig generation failed for {wig_id}: {e}")

    _wig_cache = results
    return jsonify(results)


# ── ROUTE: /api/shop/convince ─────────────────────────────────────────────────
@app.route("/api/shop/convince", methods=["POST"])
def shop_convince():
    data      = request.get_json()
    image_b64 = data.get("image", "")
    product   = data.get("product", "stylish accessory")

    kontext_prompt = (
        f"The same person, identical face and appearance, is now wearing {product}. "
        f"Funny luxury fashion advertisement, high-glam editorial look, keep the face completely unchanged."
    )

    # 1. Try flux-pro/kontext (best identity preservation)
    if image_b64:
        img_url = upload_image_to_fal(image_b64) or image_b64  # fall back to data URL
        try:
            resp = requests.post(
                "https://fal.run/fal-ai/flux-pro/kontext",
                headers=FAL_HEADERS,
                json={"image_url": img_url, "prompt": kontext_prompt,
                      "num_images": 1, "output_format": "jpeg"},
                timeout=90,
            )
            d = resp.json()
            if "images" in d:
                return jsonify({"image_url": d["images"][0]["url"], "mode": "kontext"})
        except Exception as e:
            print(f"kontext failed: {e}")

    # 2. Fallback: img2img
    if image_b64:
        try:
            resp = requests.post(
                "https://fal.run/fal-ai/flux/dev/image-to-image",
                headers=FAL_HEADERS,
                json={"image_url": image_b64, "prompt": kontext_prompt,
                      "strength": 0.65, "num_inference_steps": 25, "num_images": 1},
                timeout=60,
            )
            d = resp.json()
            if "images" in d:
                return jsonify({"image_url": d["images"][0]["url"], "mode": "img2img"})
        except Exception:
            pass

    # 3. Fallback: text only
    funny_prompt = (
        f"hilarious over-the-top luxury fashion advertisement, person wearing {product}, "
        f"dramatic editorial pose, high fashion magazine cover, vibrant saturated colors, "
        f"funny exaggerated expression, cinematic lighting"
    )
    resp = requests.post(
        "https://fal.run/fal-ai/flux/schnell",
        headers=FAL_HEADERS,
        json={"prompt": funny_prompt, "image_size": "square_hd",
              "num_inference_steps": 4, "num_images": 1},
        timeout=30,
    )
    d = resp.json()
    return jsonify({"image_url": d["images"][0]["url"], "mode": "txt2img"})


# ── ROUTE: /api/shop/checkout ─────────────────────────────────────────────────
@app.route("/api/shop/checkout", methods=["POST"])
def shop_checkout():
    data      = request.get_json()
    image_b64 = data.get("image", "")
    cart      = data.get("cart", [])

    items_str   = ", ".join(cart) if cart else "various fashion items"
    haul_prompt = (
        f"The same person, identical face, now wearing ALL of these at once: {items_str}. "
        f"Chaotic luxury fashion haul, too much glam, funny fashion disaster, "
        f"vibrant editorial photography. Keep the face completely unchanged."
    )

    # 1. Try flux-pro/kontext for identity-preserving haul photo
    if image_b64:
        img_url = upload_image_to_fal(image_b64) or image_b64
        try:
            resp = requests.post(
                "https://fal.run/fal-ai/flux-pro/kontext",
                headers=FAL_HEADERS,
                json={"image_url": img_url, "prompt": haul_prompt,
                      "num_images": 1, "output_format": "jpeg"},
                timeout=90,
            )
            d = resp.json()
            if "images" in d:
                return jsonify({"image_url": d["images"][0]["url"]})
        except Exception as e:
            print(f"kontext checkout failed: {e}")

    # 2. Fallback: img2img
    if image_b64:
        try:
            resp = requests.post(
                "https://fal.run/fal-ai/flux/dev/image-to-image",
                headers=FAL_HEADERS,
                json={"image_url": image_b64, "prompt": haul_prompt,
                      "strength": 0.70, "num_inference_steps": 25, "num_images": 1},
                timeout=60,
            )
            d = resp.json()
            if "images" in d:
                return jsonify({"image_url": d["images"][0]["url"]})
        except Exception:
            pass

    # 3. Fallback: text only
    fallback_prompt = (
        f"hilarious luxury fashion haul photo, model wearing ALL of these: {items_str}. "
        f"Chaotic high fashion, too much glam, over-the-top styling, funny fashion disaster, "
        f"vibrant editorial photography"
    )
    resp = requests.post(
        "https://fal.run/fal-ai/flux/schnell",
        headers=FAL_HEADERS,
        json={"prompt": fallback_prompt, "image_size": "square_hd",
              "num_inference_steps": 4, "num_images": 1},
        timeout=30,
    )
    d = resp.json()
    return jsonify({"image_url": d["images"][0]["url"]})


from flask import send_file as _send_file

_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def serve_index():
    return _send_file(os.path.join(_ROOT, "index.html"))

@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(os.path.join(_ROOT, "js"), filename)

@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(os.path.join(_ROOT, "css"), filename)

if __name__ == "__main__":
    print("FaceForge backend running on http://localhost:5050")
    app.run(port=5050, debug=True)