"""1 ngon ngu co the gom NHIEU PHAN audio (vd 1.mp3 ... 10.mp3, ten file khong
co ten ngon ngu - nguoi dung tu dat ten ngon ngu khi them). Module nay: sap theo
so, bao so thu tu bi hong, ghep cac phan thanh 1 file."""
import hashlib
import json
import os
import re
import tempfile
import unicodedata

from core import audio
from core.capcut_project import AUDIO_EXTS
from core.paths import MAX_PATH, find_ffmpeg, long_path_message, path_too_long

NUMBER_RE = re.compile(r"(\d+)(?!.*\d)")    # so CUOI CUNG trong ten file


def part_number(path: str):
    m = NUMBER_RE.search(os.path.splitext(os.path.basename(path))[0])
    return int(m.group(1)) if m else None


def natural_key(path: str):
    """Sap 1, 2, ... 10 (khong phai 1, 10, 2)."""
    name = os.path.basename(path).lower()
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def sort_parts(paths: list) -> list:
    return sorted(paths, key=natural_key)


def is_merged_file(path: str) -> bool:
    """File ghep cua tool: "<ten cac phan> Ghép.mp3" (T-030) hoac kieu cu "<ngon ngu> (gộp).mp3" - khong lay lai lam 1 phan.
    Ke ca file DO DANG cua lan ghep bi ngat (tool tat giua chung / Drive khoa khong xoa duoc): "... Ghép.mp3.part.mp3",
    "... (gộp).mp3.part.mp3" va file ghi thu ".nhanban_ghi_thu.tmp" (QA T-030: file do dang bi lay lam phan 1)."""
    name = unicodedata.normalize("NFC", os.path.basename(path))
    low = name.lower()
    return ("(gộp)" in name or os.path.splitext(name)[0].endswith(MERGED_TAG) or low.endswith(TEMP_SUFFIX)
            or (MERGED_TAG + MERGED_EXT + ".") in name or ("(gộp)" + MERGED_EXT + ".") in name
            or low == PROBE_NAME)


def audio_files_in(folder: str) -> list:
    return sort_parts([os.path.join(folder, n) for n in os.listdir(folder)
                       if os.path.splitext(n)[1].lower() in AUDIO_EXTS and not is_merged_file(n)])


def numbering_problem(paths: list) -> str:
    """'' neu so thu tu lien tuc; nguoc lai mo ta (vd 'thiếu phần 4, 7').
    Khong biet duoc thieu phan CUOI (1..9 thieu 10) - nguoi dung tu doi chieu
    so phan + tong thoi luong, va kiem tra noi dung sau khi chay se bao."""
    if len(paths) < 2:
        return ""
    numbers = [part_number(p) for p in paths]
    if any(x is None for x in numbers):
        return "có file không đánh số — kiểm tra thứ tự"
    dup = sorted({x for x in numbers if numbers.count(x) > 1})
    if dup:
        return f"trùng phần {', '.join(map(str, dup))}"
    present = set(numbers)
    missing = [x for x in range(1, max(present) + 1) if x not in present]
    return f"thiếu phần {', '.join(map(str, missing))}" if missing else ""


# File ghep (T-030, nguoi dung chot 26/10; hotfix ten 26/10): "<ten cac phan bo so thu tu> Ghép.mp3" ngay trong thu muc do (ke ca Google
# Drive); chi khi ghi loi (khong quyen / duong dan qua dai) moi luu vao <project>\voice nhan ban\ + canh bao.
MERGED_TAG = " Ghép"
MERGED_EXT = ".mp3"
OLD_SUFFIX = " (gộp).mp3"     # kieu cu (<= v1.2.10): van dung lai neu cung chu ky
TEMP_SUFFIX = ".part.mp3"     # hau to dai nhat cua file canh file gop (.json ngan hon)
PROBE_NAME = ".nhanban_ghi_thu.tmp"     # file ghi thu quyen ghi thu muc (_writable)


FALLBACK_SUBDIR = "voice nhan ban"     # thu muc trong project CapCut khi khong ghi duoc canh cac phan


def _safe(name: str) -> str:
    return "".join("_" if ch in '<>:"/\\|?*' else ch for ch in name).strip() or "audio"


def _fit_in(folder: str, stem: str, suffix: str):
    """<folder>/<stem><suffix> vua gioi han Windows (ke ca file tam .part.mp3), hoac None neu thu muc qua sau.
    Ten dai -> giu phan CUOI ten + 6 ky tu ma bam."""
    out = os.path.join(folder, f"{stem}{suffix}")
    if not path_too_long(out, len(TEMP_SUFFIX)):
        return out
    tag = "~" + hashlib.sha1(stem.encode("utf-8")).hexdigest()[:6]
    room = MAX_PATH - 1 - len(os.path.abspath(folder)) - 1 - len(tag + suffix + TEMP_SUFFIX)
    if room < 8:
        return None
    return os.path.join(folder, f"{stem[-room:].lstrip()}{tag}{suffix}")


def merged_name(parts: list) -> str:
    """Ten file ghep (khong duoi) = ten cac phan BO so thu tu phan (hotfix 26/10: giu ten giong doc). Vd "289 tây 1_full",
    "289 tây 2_full" -> "289 tây_full"; "176 Đức 1" -> "176 Đức"; "44.1 tbn", "44.12 tbn" -> "44 tbn". Cac phan chi la
    so (1.mp3, 2.mp3) -> ten thu muc chua cac phan."""
    stems = [unicodedata.normalize("NFC", os.path.splitext(os.path.basename(p))[0]) for p in parts]
    head = os.path.commonprefix(stems)
    tail = os.path.commonprefix([x[::-1] for x in stems])[::-1] if len(stems) > 1 else ""
    if len(head) + len(tail) > min(len(x) for x in stems):     # 1 phan / ten trung: phan dau va duoi chong nhau
        tail = ""
    head = head.rstrip("0123456789").rstrip(" ._-(")
    tail = tail.lstrip("0123456789").lstrip(" .-)")
    name = head + tail if tail.startswith("_") else f"{head} {tail}"
    name = re.sub(r"\s+", " ", name).strip(" ._-")
    if not any(ch.isalpha() for ch in name):                   # chi con so / rong -> ten thu muc
        name = os.path.basename(os.path.dirname(os.path.abspath(parts[0])))
    return _safe(name)


def merged_target(parts: list, folder: str = None):
    """File ghep "<ten cac phan bo so thu tu> Ghép.mp3" (merged_name) trong `folder` (mac dinh thu muc phan 1); None neu
    qua dai."""
    src_folder = os.path.dirname(os.path.abspath(parts[0]))
    return _fit_in(folder or src_folder, merged_name(parts), MERGED_TAG + MERGED_EXT)


def _signature(parts: list) -> list:
    return [[os.path.abspath(p), os.path.getsize(p), int(os.path.getmtime(p))] for p in parts]


def _same_parts(out: str, signature: list) -> bool:
    """File ghep da co + sidecar .json cung chu ky cac phan (cac phan khong doi) -> dung lai."""
    if not out or not os.path.isfile(out) or not os.path.isfile(out + ".json"):
        return False
    try:
        with open(out + ".json", encoding="utf-8") as f:
            return json.load(f) == signature
    except (OSError, ValueError):
        return False


def _writable(folder: str) -> bool:
    """Ghi thu 1 file nho vao thu muc (khong quyen / thu muc chi doc / o mang loi -> False)."""
    try:
        os.makedirs(folder, exist_ok=True)
        probe = os.path.join(folder, PROBE_NAME)
        with open(probe, "wb") as f:
            f.write(b"0")
        os.remove(probe)
        return True
    except OSError:
        return False


def _concat(parts: list, out: str, signature: list):
    """Ghep cac phan (dung thu tu) thanh mp3 `out` + sidecar chu ky."""
    # Danh sach cho ffmpeg concat ghi vao thu muc tam (duong dan ngan), khong ghi canh cac phan
    fd, list_file = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(list_file, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '" + os.path.abspath(p).replace("'", "'\\''") + "'\n")
    tmp = out + TEMP_SUFFIX
    base = [find_ffmpeg(), "-hide_banner", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", list_file]
    try:
        # Cac phan deu la mp3 -> noi thang (vai giay); khong duoc thi nen lai thanh mp3
        ok = all(p.lower().endswith(".mp3") for p in parts) and \
            audio._run(base + ["-c", "copy", tmp]).returncode == 0
        if not ok:
            r = audio._run(base + ["-vn", "-c:a", "libmp3lame", "-b:a", "192k", tmp])
            if r.returncode != 0:
                raise audio.AudioError(f"Không ghép được các phần audio:\n{r.stderr[-400:]}")
        os.replace(tmp, out)
    finally:
        for f in (list_file, tmp):
            if os.path.exists(f):
                os.remove(f)
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(signature, f)


def merge_parts(parts: list, language: str, fallback_dir: str = "") -> tuple:
    """Ghep cac phan voice nhan ban thanh 1 file mp3 - CapCut can 1 file lam material. Tra ve (duong_dan, da_dung_lai,
    canh_bao). Dich: "<ten cac phan bo so thu tu> Ghép.mp3" ngay trong thu muc do (ke ca Google Drive). Khong ghi duoc
    (khong quyen / duong dan qua dai / loi ghi) -> `fallback_dir`/voice nhan ban/ + canh bao. Dung lai file ghep da co
    cung chu ky (ca kieu cu "<ngon ngu> (gộp).mp3") -> giu duoc cache nhan dang giong noi."""
    signature = _signature(parts)
    folder = os.path.dirname(os.path.abspath(parts[0]))
    target = merged_target(parts)
    fb_dir = os.path.join(os.path.abspath(fallback_dir), FALLBACK_SUBDIR) if fallback_dir else ""
    backup = merged_target(parts, fb_dir) if fb_dir else None
    old = [_fit_in(d, _safe(language), OLD_SUFFIX) for d in (folder, fb_dir) if d]
    for cand in [target, backup] + old:
        if _same_parts(cand, signature):
            in_project = fb_dir and os.path.normcase(os.path.dirname(cand)) == os.path.normcase(fb_dir)
            return cand, True, (f"File ghép đang ở project (lần trước không ghi được vào thư mục {folder}): {cand}"
                                if in_project else "")
    if target and _writable(folder):
        try:
            _concat(parts, target, signature)
            return target, False, ""
        except (OSError, audio.AudioError) as e:
            if not backup:
                raise
            reason = f"lỗi ghi: {e}"
    else:
        reason = "đường dẫn quá dài" if target is None else "không có quyền ghi"
    if not backup:
        raise audio.AudioError(long_path_message(os.path.join(folder, merged_name(parts) + MERGED_TAG + MERGED_EXT))
                               if target is None else f"Không ghi được file ghép vào thư mục {folder} ({reason}).")
    os.makedirs(fb_dir, exist_ok=True)
    _concat(parts, backup, signature)
    return backup, False, f"Không ghi được vào thư mục {folder} ({reason}) → đã lưu ở project: {backup}"


def merge_for_editor(parts: list) -> tuple:
    """Ghep cac phan GIONG GOC thanh "<ten cac phan bo so thu tu> Ghép.mp3" trong thu muc cac phan - CHI de nguoi dung dung khi dung
    video (tool van doc tung file goc, draft khong doi). Tra ve (duong_dan, da_co_san). Loi -> AudioError / OSError."""
    signature = _signature(parts)
    target = merged_target(parts)
    if target is None:
        raise audio.AudioError("đường dẫn quá dài")
    if _same_parts(target, signature):
        return target, True
    if not _writable(os.path.dirname(target)):
        raise audio.AudioError("không có quyền ghi")
    _concat(parts, target, signature)
    return target, False
