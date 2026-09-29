"""1 ngon ngu co the gom NHIEU PHAN audio (vd 1.mp3 ... 10.mp3, ten file khong
co ten ngon ngu - nguoi dung tu dat ten ngon ngu khi them). Module nay: sap theo
so, bao so thu tu bi hong, ghep cac phan thanh 1 file."""
import json
import os
import re

from core import audio
from core.capcut_project import AUDIO_EXTS
from core.paths import find_ffmpeg

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
    return "(gộp)" in os.path.basename(path)


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


def merged_path(parts: list, language: str) -> str:
    safe = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in language).strip() or "audio"
    return os.path.join(os.path.dirname(parts[0]), f"{safe} (gộp).mp3")


def merge_parts(parts: list, language: str) -> tuple:
    """Ghep cac phan (dung thu tu) thanh 1 file mp3 canh cac phan - CapCut can
    1 file lam material. Dung lai file gop cu neu cac phan khong doi (giu duoc
    cache nhan dang giong noi). Tra ve (duong_dan, da_dung_lai)."""
    out = merged_path(parts, language)
    sidecar = out + ".json"
    signature = [[os.path.abspath(p), os.path.getsize(p), int(os.path.getmtime(p))] for p in parts]
    if os.path.isfile(out) and os.path.isfile(sidecar):
        try:
            with open(sidecar, encoding="utf-8") as f:
                if json.load(f) == signature:
                    return out, True
        except (OSError, ValueError):
            pass

    list_file = out + ".txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '" + os.path.abspath(p).replace("'", "'\\''") + "'\n")
    tmp = out + ".part.mp3"
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
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(signature, f)
    return out, False
