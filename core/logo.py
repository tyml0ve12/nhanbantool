"""Logo kenh: xoa nen + vi tri/kich thuoc (luu theo TI LE khung hinh -> dung
cho moi do phan giai). Chi dung khi xuat mp4, draft CapCut khong doi."""
import hashlib
import os
from dataclasses import dataclass, asdict

import numpy as np
from PySide6.QtGui import QImage

from core.paths import get_app_dir

BG_TOLERANCE = 38          # sai khac mau (0-255) van coi la mau nen
MIN_UNIFORM_BORDER = 0.75  # >= 75% diem vien cung 1 mau -> coi la nen 1 mau


@dataclass
class LogoPlacement:
    """Toa do theo ti le khung hinh video (0..1), goc tren-trai cua logo."""
    x: float = 0.85
    y: float = 0.04
    width: float = 0.12            # be ngang logo / be ngang video
    remove_bg: bool = True

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        if not d:
            return LogoPlacement()
        return LogoPlacement(**{k: d[k] for k in ("x", "y", "width", "remove_bg") if k in d})

    @staticmethod
    def corner(corner: str, width: float, logo_aspect: float, frame_aspect: float, margin: float = 0.03):
        """Dat vao 1 goc ('tl','tr','bl','br'), cach mep `margin` be ngang video
        (khoang cach doc quy doi cung so pixel)."""
        height = width * frame_aspect / max(logo_aspect, 1e-6)   # chieu cao logo theo ti le chieu cao khung
        my = margin * frame_aspect
        x = margin if corner[1] == "l" else 1 - margin - width
        y = my if corner[0] == "t" else 1 - my - height
        return LogoPlacement(x=x, y=y, width=width)


# ---- Quy doi voi thong so "Bien doi" cua CapCut (Ty le %, Vi tri X, Y) ----
# Da doi chieu voi project that (10 logo, khung 1920x1080): draft luu clip.scale va
# clip.transform (x, y); giao dien CapCut hien:
#   Ty le % = scale x 100 ; X = transform.x x be ngang khung ; Y = transform.y x be cao khung
#   (vd transform (0.891566, 0.778571) -> X 1712, Y 840 tren khung 1920x1080).
# transform tinh theo NUA khung -> tam logo lech khoi tam khung (X/2, Y/2) diem anh, Y duong = len.
# Ty le 100% = anh co VUA KHUNG (khop theo canh, ke ca anh nho hon khung).
CAPCUT_CANVASES = {"1080p (1920 × 1080)": (1920, 1080), "4K (3840 × 2160)": (3840, 2160)}


def _capcut_base(logo_w, logo_h, canvas_w, canvas_h) -> float:
    return min(canvas_w / logo_w, canvas_h / logo_h)


def from_capcut(scale_pct, x, y, logo_w, logo_h, canvas, remove_bg=True) -> LogoPlacement:
    cw, ch = canvas
    base = _capcut_base(logo_w, logo_h, cw, ch)
    w_px, h_px = logo_w * base * scale_pct / 100, logo_h * base * scale_pct / 100
    cx, cy = cw / 2 + x / 2, ch / 2 - y / 2
    return LogoPlacement(x=(cx - w_px / 2) / cw, y=(cy - h_px / 2) / ch, width=w_px / cw, remove_bg=remove_bg)


def to_capcut(p: LogoPlacement, logo_w, logo_h, canvas) -> tuple:
    """(ty_le_%, X, Y) nhu CapCut hien thi."""
    cw, ch = canvas
    base = _capcut_base(logo_w, logo_h, cw, ch)
    w_px = p.width * cw
    h_px = w_px * logo_h / max(1, logo_w)
    scale_pct = w_px / (logo_w * base) * 100
    cx, cy = p.x * cw + w_px / 2, p.y * ch + h_px / 2
    return scale_pct, (cx - cw / 2) * 2, (ch / 2 - cy) * 2


def _to_rgba_array(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    ptr = img.constBits()
    arr = np.frombuffer(ptr, np.uint8, img.sizeInBytes()).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return arr[:, :img.width()].copy()


def _from_rgba_array(arr: np.ndarray) -> QImage:
    h, w, _ = arr.shape
    arr = np.ascontiguousarray(arr)
    return QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


def remove_background(img: QImage) -> QImage:
    """Anh da co nen trong suot -> giu nguyen. Nen 1 mau (trang/den/...) -> xoa
    bang cach LOANG TU VIEN vao: chi xoa vung nen noi voi mep anh, chu/hinh ben
    trong trung mau nen khong bi xoa. Nen phuc tap -> giu nguyen (nen dung PNG trong suot)."""
    arr = _to_rgba_array(img)
    h, w, _ = arr.shape
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if (border[:, 3] < 250).mean() > 0.3:
        return img                                   # da trong suot san
    rgb = arr[:, :, :3].astype(np.int16)
    bg = np.median(border[:, :3], axis=0).astype(np.int16)
    close = np.abs(rgb - bg).max(axis=2) <= BG_TOLERANCE
    border_close = np.concatenate([close[0], close[-1], close[:, 0], close[:, -1]])
    if border_close.mean() < MIN_UNIFORM_BORDER:
        return img                                   # nen khong phai 1 mau

    # Loang tu vien (BFS theo tung dong quet, dung numpy cho nhanh)
    mask = np.zeros((h, w), bool)
    mask[0] |= close[0]; mask[-1] |= close[-1]; mask[:, 0] |= close[:, 0]; mask[:, -1] |= close[:, -1]
    while True:
        grown = mask.copy()
        grown[1:] |= mask[:-1]; grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]; grown[:, :-1] |= mask[:, 1:]
        grown &= close
        if (grown == mask).all():
            break
        mask = grown
    out = arr.copy()
    out[mask, 3] = 0
    # Nen "lom dom" (chấm/hat nhieu mau tren nen den...): sau khi xoa nen con lai
    # cac manh li ti lo lung -> xoa not manh roi rac qua nho. Phan that cua logo
    # (chu, vong, hinh) la manh lon lien khoi nen khong bi anh huong.
    opaque = out[:, :, 3] > 0
    small = _small_components(opaque, max(40, int(SPECK_MAX_AREA * h * w)))
    out[small, 3] = 0
    return _from_rgba_array(out)


SPECK_MAX_AREA = 0.0015    # manh roi < 0.15% dien tich anh -> coi la hat nhieu cua nen


def _small_components(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Tra ve mask cac vung lien thong (4 huong) co dien tich <= max_area.
    Gom theo DOAN tren tung hang + union-find giua cac hang (khong can scipy)."""
    h, w = mask.shape
    parent = []
    area = []
    run_rows = []          # moi hang: list (start, end, id)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    padded = np.zeros((h, w + 2), bool)
    padded[:, 1:-1] = mask
    diff = np.diff(padded.astype(np.int8), axis=1)
    prev = []
    for y in range(h):
        starts = np.flatnonzero(diff[y] == 1)
        ends = np.flatnonzero(diff[y] == -1)
        cur = []
        j = 0
        for s, e in zip(starts.tolist(), ends.tolist()):
            rid = len(parent)
            parent.append(rid)
            area.append(e - s)
            # noi voi cac doan hang tren co cot giao nhau
            while j < len(prev) and prev[j][1] <= s:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] < e:
                a, b = find(rid), find(prev[k][2])
                if a != b:
                    parent[a] = b
                    area[b] += area[a]
                k += 1
            cur.append((s, e, rid))
        run_rows.append(cur)
        prev = cur

    out = np.zeros_like(mask)
    for y, runs in enumerate(run_rows):
        for s, e, rid in runs:
            if area[find(rid)] <= max_area:
                out[y, s:e] = True
    return out


def load_logo(path: str, remove_bg: bool) -> QImage:
    img = QImage(path)
    if img.isNull():
        raise ValueError(f"Không đọc được ảnh logo: {os.path.basename(path)}")
    return remove_background(img) if remove_bg else img.convertToFormat(QImage.Format.Format_RGBA8888)


def prepared_logo_png(path: str, remove_bg: bool) -> str:
    """File PNG da xu ly (xoa nen neu can) de ffmpeg chen vao video; cache theo noi dung."""
    st = os.stat(path)
    key = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{remove_bg}".encode()).hexdigest()
    out_dir = os.path.join(get_app_dir(), "cache", "logos")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, key + ".png")
    if not os.path.isfile(out):
        if not load_logo(path, remove_bg).save(out, "PNG"):
            raise ValueError("Không lưu được logo đã xử lý.")
    return out


def grab_frame(video_path: str) -> str:
    """1 khung hinh that cua video (giay 30, video ngan thi 10% thoi luong) lam nen
    cho hop thoai dat logo. Tra ve duong dan PNG hoac '' neu khong lay duoc."""
    from core import audio
    from core.paths import find_ffmpeg
    if not video_path or not os.path.isfile(video_path):
        return ""
    st = os.stat(video_path)
    key = hashlib.sha1(f"{os.path.abspath(video_path)}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
    out_dir = os.path.join(get_app_dir(), "cache", "frames")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, key + ".png")
    if not os.path.isfile(out):
        try:
            t = min(30.0, audio.probe_duration(video_path) * 0.1)
        except Exception:
            t = 0.0
        audio._run([find_ffmpeg(), "-hide_banner", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", video_path,
                    "-frames:v", "1", out])
    return out if os.path.isfile(out) else ""


def overlay_geometry(placement: LogoPlacement, logo_w: int, logo_h: int, video_w: int, video_h: int) -> tuple:
    """(x, y, w, h) theo pixel tren video, so chan (yeu cau cua bo nen)."""
    """KHONG ep logo vao trong khung: logo lan ra ngoai mep (nhu CapCut cho phep) thi
    render se cat phan thua - giu dung vi tri da dat."""
    w = max(2, int(round(placement.width * video_w / 2)) * 2)
    h = max(2, int(round(w * logo_h / max(1, logo_w) / 2)) * 2)
    x = int(round(placement.x * video_w))
    y = int(round(placement.y * video_h))
    return x, y, w, h


def visible_part(x, y, w, h, video_w, video_h):
    """Phan logo nam trong khung: (cat_x, cat_y, cat_w, cat_h, dat_x, dat_y) - so chan.
    None neu logo nam han ngoai khung."""
    cx0, cy0 = max(0, -x), max(0, -y)
    cw = min(w, video_w - x) - cx0
    ch = min(h, video_h - y) - cy0
    cx0, cy0 = cx0 // 2 * 2, cy0 // 2 * 2
    cw, ch = cw // 2 * 2, ch // 2 * 2
    if cw <= 0 or ch <= 0:
        return None
    return cx0, cy0, cw, ch, max(0, x), max(0, y)
