import os
import time
import urllib.request
from collections import Counter, deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

W, H = 900, 500
N = 6000

WEB_TEXT = "My Website\nmohamed-khaled.site"

GESTURE_WORDS = {
    "ONE":  "Hi",
    "FIVE": WEB_TEXT,
    "TWO":  "I Love You",
    "CALL": "My Name",
    "OK":   "Mohamed",
}
HEART_GESTURES = {"THREE"}
SQUARE_GESTURES = {"FOUR"}

SCATTER_COLOR = (255, 210, 140)
WORD_COLORS = {
    "Hi":         (255, 230, 80),
    "I Love You": (180, 105, 255),
    "My Name":    (255, 110, 190),
    "Mohamed":    (120, 255, 150),
    WEB_TEXT:     (245, 245, 255),
}
SQUARE_COLOR = (60, 230, 255)
PLANET_COLOR = (40, 140, 255)
RING_COLOR = (170, 215, 255)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def fit_text(line, max_w, max_h):
    scale, thick = 1, 2
    while True:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        if tw > max_w or th > max_h:
            break
        scale += 0.2
        thick = max(2, int(scale * 2))
    scale -= 0.2
    thick = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    return scale, thick, tw, th


def word_targets(word):
    """يحول الكلمة (أو سطرين) لمواضع نقط."""
    lines = word.split("\n")
    mask = np.zeros((H, W), np.uint8)
    if len(lines) == 1:
        limits = [(W * 0.55, H * 0.4)]
    else:
        limits = [(W * 0.55, H * 0.2), (W * 0.8, H * 0.09)]
    fits = [fit_text(l, *lim) for l, lim in zip(lines, limits)]
    gap = 30
    total = sum(f[3] for f in fits) + gap * (len(lines) - 1)
    y = (H - total) // 2
    for line, (scale, thick, tw, th) in zip(lines, fits):
        y += th
        cv2.putText(mask, line, ((W - tw) // 2, y),
                    cv2.FONT_HERSHEY_DUPLEX, scale, 255, thick)
        y += gap
    ys, xs = np.nonzero(mask)
    idx = np.random.choice(len(xs), N, replace=len(xs) < N)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)


def get_gesture(lm, aspect):
    """يرجّع اسم الحركة (أو None) وعدد الأصابع."""
    def d(a, b):
        return np.hypot((a.x - b.x) * aspect, a.y - b.y)

    wrist = lm[0]
    scale = d(lm[0], lm[9]) + 1e-6

    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    idx_, mid_, rng_, pnk_ = [
        d(lm[t], wrist) > d(lm[p], wrist) * 1.15 for t, p in zip(tips, pips)
    ]
    palm_w = d(lm[5], lm[17]) + 1e-6
    thumb = (d(lm[4], lm[5]) > 0.6 * scale
             or d(lm[4], lm[17]) > 1.25 * palm_w)
    count = int(idx_) + int(mid_) + int(rng_) + int(pnk_) + int(thumb)

    if d(lm[4], lm[8]) < 0.3 * scale and mid_ and rng_ and pnk_:
        return "OK", count

    key = (idx_, mid_, rng_, pnk_)
    if key == (False, False, False, False):
        return "PLANET", count
    if key == (True, False, False, False):
        return "ONE", count
    if key == (True, True, False, False):
        return "TWO", count
    if key == (True, True, True, False):
        return "THREE", count
    if key == (True, True, True, True):
        return ("FIVE" if thumb else "FOUR"), count
    if key == (False, False, False, True):
        return ("CALL" if thumb else None), count
    return None, count


def mode_of(gesture):
    """يحول الحركة لوضع العرض: PLANET / HEART / SQUARE / كلمة / None."""
    if gesture == "PLANET":
        return "PLANET"
    if gesture in HEART_GESTURES:
        return "HEART"
    if gesture in SQUARE_GESTURES:
        return "SQUARE"
    return GESTURE_WORDS.get(gesture)


def draw_hand(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)


def _rot_x(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]], np.float32)


def _rot_y(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]], np.float32)


def _rot_z(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]], np.float32)


R = 75
NS = int(N * 0.6)
_i = np.arange(NS) + 0.5
_phi = np.arccos(1 - 2 * _i / NS)
_th = np.pi * (1 + 5 ** 0.5) * _i
SPH = (np.stack([np.cos(_th) * np.sin(_phi), np.cos(_phi),
                 np.sin(_th) * np.sin(_phi)], 1) * R).astype(np.float32)

_NR = N - NS
_ang = np.random.rand(_NR) * 2 * np.pi
_rad = np.where(np.random.rand(_NR) < 0.55,
                np.random.uniform(1.40, 1.65, _NR),
                np.random.uniform(1.78, 2.00, _NR)) * R
RING = np.stack([_rad * np.cos(_ang), np.zeros(_NR), _rad * np.sin(_ang)],
                1).astype(np.float32)
TILT = _rot_z(-0.35) @ _rot_x(0.5)


def planet_frame(angle):
    """مواضع نقط الكوكب في لحظة معينة + العمق (للتظليل)."""
    sph = SPH @ _rot_y(angle).T
    ring = RING @ _rot_y(angle * 0.4).T
    pts = np.vstack([sph, ring]) @ TILT.T
    tgt = np.stack([W / 2 + pts[:, 0], H / 2 + pts[:, 1]], 1).astype(np.float32)
    depth = np.clip(0.5 + pts[:, 2] / (4 * R), 0, 1).astype(np.float32)
    return tgt, depth


HS = 9


def _heart_curve(t):
    x = 16 * np.sin(t) ** 3
    y = (13 * np.cos(t) - 5 * np.cos(2 * t)
         - 2 * np.cos(3 * t) - np.cos(4 * t))
    return x, y + 2.5


_xh, _yh = _heart_curve(np.random.rand(N) * 2 * np.pi)
_thick = np.random.uniform(0.94, 1.0, N)
HEART = np.stack([
    _xh * _thick * HS,
    -_yh * _thick * HS,
    (np.random.rand(N) - 0.5) * 40,
], 1).astype(np.float32)


def heart_frame(t):
    beat = 1 + 0.07 * np.sin(t * 7)
    sway = 0.5 * np.sin(t * 1.5)
    pts = (HEART * beat) @ _rot_y(sway).T
    tgt = np.stack([W / 2 + pts[:, 0], H / 2 + pts[:, 1]], 1).astype(np.float32)
    depth = np.clip(0.5 + pts[:, 2] / 280, 0, 1).astype(np.float32)
    return tgt, depth


SQ = 130
_side = np.random.randint(0, 4, N)
_u = np.random.uniform(-1, 1, N)
_sx = np.select([_side == 0, _side == 1, _side == 2], [_u, 1, _u], -1)
_sy = np.select([_side == 0, _side == 1, _side == 2], [-1, _u, 1], _u)
_sthick = np.random.uniform(0.95, 1.0, N)
SQUARE = np.stack([_sx * _sthick * SQ, _sy * _sthick * SQ,
                   (np.random.rand(N) - 0.5) * 20], 1).astype(np.float32)
SQ_TILT = _rot_x(0.35)


def square_frame(t):
    pts = (SQUARE @ _rot_y(t * 1.5).T) @ SQ_TILT.T
    tgt = np.stack([W / 2 + pts[:, 0], H / 2 + pts[:, 1]], 1).astype(np.float32)
    depth = np.clip(0.5 + pts[:, 2] / (4 * SQ), 0, 1).astype(np.float32)
    return tgt, depth


def solid(color):
    return np.tile(np.array(color, np.float32), (N, 1))


scatter_col = solid(SCATTER_COLOR)
planet_col = np.vstack([solid(PLANET_COLOR)[:NS], solid(RING_COLOR)[:N - NS]])
square_col = solid(SQUARE_COLOR)
heart_col = np.stack([np.random.uniform(40, 150, N),
                      np.random.uniform(30, 110, N),
                      np.full(N, 255.0)], 1).astype(np.float32)
word_col = {w: solid(c) for w, c in WORD_COLORS.items()}
jitter = (0.7 + 0.3 * np.random.rand(N)).astype(np.float32)

pos = np.random.rand(N, 2).astype(np.float32) * [W, H]
vel = (np.random.rand(N, 2).astype(np.float32) - 0.5) * 2
cols_now = scatter_col.copy()
targets_cache = {w: word_targets(w) for w in set(GESTURE_WORDS.values())}

offset = np.zeros(2, np.float32)
raw_off = np.zeros(2, np.float32)

cap = cv2.VideoCapture(0)
history = deque(maxlen=5)
current = None
angle = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect_for_video(mp_img, int(time.time() * 1000))

    gesture, count = None, 0
    if res.hand_landmarks:
        lm = res.hand_landmarks[0]
        draw_hand(frame, lm)
        fh, fw = frame.shape[:2]
        gesture, count = get_gesture(lm, fw / fh)
        raw_off = np.array([(lm[9].x - 0.5) * W * 0.9,
                            (lm[9].y - 0.5) * H * 0.9], np.float32)

    history.append(mode_of(gesture))
    voted, votes = Counter(history).most_common(1)[0]
    if votes >= 3 and voted != current:
        current = voted
        if voted is None:
            vel = (np.random.rand(N, 2).astype(np.float32) - 0.5) * 4

    offset += (raw_off - offset) * 0.25
    depth = np.ones(N, np.float32)
    if current == "PLANET":
        angle += 0.05
        tgt, depth = planet_frame(angle)
        pos += ((tgt + offset) - pos) * 0.2
        pos += (np.random.rand(N, 2).astype(np.float32) - 0.5) * 0.3
        cols_target = planet_col
    elif current == "HEART":
        tgt, depth = heart_frame(time.time())
        pos += ((tgt + offset) - pos) * 0.2
        pos += (np.random.rand(N, 2).astype(np.float32) - 0.5) * 0.3
        cols_target = heart_col
    elif current == "SQUARE":
        tgt, depth = square_frame(time.time())
        pos += ((tgt + offset) - pos) * 0.2
        pos += (np.random.rand(N, 2).astype(np.float32) - 0.5) * 0.3
        cols_target = square_col
    elif current in targets_cache:
        pos += ((targets_cache[current] + offset) - pos) * 0.12
        pos += (np.random.rand(N, 2).astype(np.float32) - 0.5) * 0.6
        cols_target = word_col[current]
    else:
        pos += vel
        vel += (np.random.rand(N, 2).astype(np.float32) - 0.5) * 0.3
        vel = np.clip(vel, -2.5, 2.5)
        for a, lim in ((0, W), (1, H)):
            out = (pos[:, a] < 0) | (pos[:, a] > lim)
            vel[out, a] *= -1
            pos[:, a] = np.clip(pos[:, a], 0, lim)
        cols_target = scatter_col

    cols_now += (cols_target - cols_now) * 0.12
    shade = (0.35 + 0.65 * depth) * jitter

    canvas = np.zeros((H, W, 3), np.uint8)
    p = pos.astype(int)
    p[:, 0] = np.clip(p[:, 0], 0, W - 1)
    p[:, 1] = np.clip(p[:, 1], 0, H - 1)
    order = np.argsort(depth)
    cols = np.clip(cols_now * shade[:, None], 0, 255).astype(np.uint8)
    canvas[p[order, 1], p[order, 0]] = cols[order]

    cv2.putText(frame, f"Fingers: {count}  Mode: {current or '-'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)
    cv2.imshow("Words", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
