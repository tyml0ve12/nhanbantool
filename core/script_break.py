"""Tao kich ban ngat (T-033): chen ky hieu ngat nghi cua cong cu doc (ai33 / Google voice / Minimax...) vao kich ban.

- Chen ngay sau . ! ? 。 ！ ？ ؟ (sau ca ngoac / nhay dong neu co) va cuoi moi dong co chu.
- Bo qua: so (3.14, "1." dau dong), viet tat (Mr. / U.S. / e.g. / chu thuong ngay sau dau cham), dau ba cham.
- Chi CHEN them ky hieu, khong doi bat ky chu nao khac; chay lai nhieu lan khong chen trung.
- Danh sach cong cu + ky hieu luu rieng o <app>\\config\\cong_cu_ngat.json (cap nhat app khong dong vao).
"""
import json
import os

from core.paths import get_app_dir

DEFAULT_TOOLS = [
    {"name": "ai33", "marker": "<#0.5#>"},
    {"name": "Google voice", "marker": '<break time="500ms"/>'},
    {"name": "Minimax", "marker": "<#0.5#>"},
]

END_MARKS = ".!?。！？؟"
CLOSERS = "\"'”’»)]}）」』】"
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "eg", "ie", "vol", "inc", "ltd",
    "sra", "dra", "srta", "mme", "mlle", "hr", "fr", "nr", "bzw", "usw", "zb", "approx", "dept", "fig", "tp",
}


def config_path() -> str:
    return os.path.join(get_app_dir(), "config", "cong_cu_ngat.json")


def load_tools() -> list:
    """[{name, marker}] - file cau hinh hong / chua co thi dung mac dinh."""
    try:
        with open(config_path(), encoding="utf-8") as f:
            return [{"name": str(t["name"]), "marker": str(t["marker"])} for t in json.load(f)]
    except (OSError, ValueError, KeyError, TypeError):
        return [dict(t) for t in DEFAULT_TOOLS]


def save_tools(tools: list):
    path = config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(tools, f, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)


def _is_sentence_end(line: str, i: int) -> bool:
    """line[i] la dau cham / hoi / than: co phai het cau khong."""
    ch = line[i]
    nxt = line[i + 1] if i + 1 < len(line) else ""
    if nxt and nxt in END_MARKS + "…":                      # chuoi dau "?!", "..." -> chi xet dau cuoi chuoi
        return False
    if ch != ".":
        return True
    if i > 0 and line[i - 1] in ".…":                         # dau ba cham "..."
        return False
    if nxt and not nxt.isspace() and nxt not in CLOSERS:     # 3.14 / example.com / U.S
        return False
    j = i
    while j > 0 and line[j - 1].isalnum():
        j -= 1
    word = line[j:i]
    if word.isdigit() and not line[:j].strip():              # "1." danh so dau dong
        return False
    if word.isalpha() and len(word) == 1 and word.isupper():  # "J. K. Rowling"
        return False
    if j > 0 and line[j - 1] == "." and word.isalpha():       # "e.g.", "U.S." (chu giua cac dau cham)
        return False
    if word.lower() in ABBREVIATIONS:
        return False
    k = i + 1
    while k < len(line) and (line[k] in CLOSERS or line[k] in " \t"):
        k += 1
    if k < len(line) and line[k].islower():                  # chu thuong sau dau cham -> chua het cau
        return False
    return True


def _break_line(line: str, marker: str) -> str:
    out, i, n = [], 0, len(line)
    while i < n:
        if line[i] in END_MARKS and _is_sentence_end(line, i):
            j = i + 1
            while j < n and line[j] in CLOSERS:
                j += 1
            out.append(line[i:j])
            if not line.startswith(marker, j):
                out.append(marker)
            i = j
        else:
            out.append(line[i])
            i += 1
    body = "".join(out)
    stripped = body.rstrip(" \t\r")
    if stripped.strip() and not stripped.endswith(marker):   # cuoi dong co chu ma chua ngat
        body = stripped + marker + body[len(stripped):]
    return body


def insert_breaks(text: str, marker: str) -> str:
    """Chen marker sau moi cau va cuoi moi dong co chu. Khong doi chu nao khac."""
    if not marker:
        return text
    return "\n".join(_break_line(line, marker) for line in text.split("\n"))
