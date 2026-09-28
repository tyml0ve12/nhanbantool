"""Cac buoc xu ly thuan (khong phu thuoc giao dien) - dung chung cho app va CLI."""
import json
import os
from dataclasses import dataclass

from core.capcut_project import ProjectError, US, AUDIO_EXTS

SEARCH_DEPTH = 3


@dataclass
class OriginalClip:
    target_start_us: int
    target_duration_us: int
    source_path: str        # duong dan ghi trong draft (co the cu)
    source_start_us: int
    source_duration_us: int


def read_original_clips(draft: dict, track_index: int) -> list:
    try:
        segs = draft["tracks"][track_index]["segments"]
        audios = {m["id"]: m for m in draft["materials"].get("audios", [])}
        clips = []
        for s in segs:
            m = audios.get(s["material_id"], {})
            clips.append(OriginalClip(
                int(s["target_timerange"]["start"]), int(s["target_timerange"]["duration"]),
                m.get("path", ""),
                int(s["source_timerange"]["start"]), int(s["source_timerange"]["duration"]),
            ))
    except (KeyError, IndexError, TypeError, ValueError):
        raise ProjectError("Không đọc được track thoại gốc trong draft (cấu trúc lạ).")
    return sorted(clips, key=lambda c: c.target_start_us)


def _walk_limited(root, depth):
    root_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count(os.sep) - root_depth >= depth:
            dirnames[:] = []
        yield dirpath, filenames


def resolve_media(path_in_draft: str, search_roots: list) -> str:
    """Duong dan trong draft hay bi cu (project da chuyen o dia/may). Tim lai
    file cung ten trong thu muc project."""
    if path_in_draft and os.path.isfile(path_in_draft):
        return os.path.normpath(path_in_draft)
    name = os.path.basename(path_in_draft.replace("\\", "/"))
    if not name:
        return ""
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, filenames in _walk_limited(root, SEARCH_DEPTH):
            if name in filenames:
                return os.path.normpath(os.path.join(dirpath, name))
    return ""


def resolve_original_sources(clips: list, search_roots: list) -> dict:
    """{duong_dan_trong_draft: duong_dan_that_hoac_rong}"""
    return {p: resolve_media(p, search_roots) for p in {c.source_path for c in clips}}


def original_texts(clips: list, source_map: dict, words_by_file: dict) -> list:
    """Noi dung tung cau goc = cac tu (whisper) nam trong source_timerange cua clip."""
    texts = []
    for c in clips:
        words = words_by_file.get(source_map.get(c.source_path, ""), [])
        a = c.source_start_us / US
        b = a + c.source_duration_us / US
        texts.append("".join(w.text for w in words if a - 0.1 <= (w.start + w.end) / 2 <= b + 0.1).strip())
    return texts


def _text_of(material: dict) -> str:
    content = material.get("content", "")
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return str(data.get("text", ""))
    except (ValueError, TypeError):
        pass
    return str(content or material.get("recognize_text", ""))


def subtitle_texts(draft: dict, clips: list) -> tuple:
    """Noi dung tung cau goc lay tu track phu de (thuong la phu de tieng Viet
    do CapCut tao, dat khop thoi gian voi giong goc). Moi doan phu de gan vao
    cau goc chua diem giua cua no.
    Tra ve (danh_sach_text, ti_le_cau_co_phu_de)."""
    texts_by_id = {m.get("id"): m for m in draft.get("materials", {}).get("texts", [])}
    best, best_cover = None, 0.0
    for track in draft.get("tracks", []):
        if track.get("type") != "text":
            continue
        buckets = [[] for _ in clips]
        for seg in track.get("segments", []):
            tr = seg.get("target_timerange") or {}
            mid = int(tr.get("start", 0)) + int(tr.get("duration", 0)) // 2
            i = _clip_at(clips, mid)
            if i is not None:
                text = _text_of(texts_by_id.get(seg.get("material_id"), {})).strip()
                if text:
                    buckets[i].append((int(tr.get("start", 0)), text))
        cover = sum(1 for b in buckets if b) / len(clips) if clips else 0.0
        if cover > best_cover:
            best_cover = cover
            best = [" ".join(t for _, t in sorted(b)) for b in buckets]
    return (best or [""] * len(clips)), best_cover


def _clip_at(clips, t_us):
    """Cau goc chua thoi diem t (tinh ca khoang trong ngay sau cau do toi cau ke)."""
    lo, hi = 0, len(clips) - 1
    if not clips or t_us < clips[0].target_start_us - US:
        return None
    while lo < hi:
        m = (lo + hi + 1) // 2
        if clips[m].target_start_us <= t_us:
            lo = m
        else:
            hi = m - 1
    c = clips[lo]
    # Phu de nam han trong khoang lang giua 2 cau (xa cau truoc) -> bo qua
    if t_us > c.target_start_us + c.target_duration_us + US and lo + 1 < len(clips):
        return None
    return lo


def compute_placements(sentences: list, clips: list, timeline_end_us: int) -> tuple:
    """Moi cau moi dat dung moc bat dau cau goc; dai qua khoang trong toi cau ke
    tiep thi CAT DUOI (khong bao gio chong 2 giong).
    Tra ve ([(target_us, source_us, duration_us)], so_cau_bi_cat_duoi)."""
    placements = []
    truncated = 0
    for i, (sent, clip) in enumerate(zip(sentences, clips)):
        next_start = clips[i + 1].target_start_us if i + 1 < len(clips) else timeline_end_us
        room = max(0, next_start - clip.target_start_us)
        length = int(round((sent.end - sent.start) * US))
        if length > room:
            truncated += 1
            length = room
        if length > 0:
            placements.append((clip.target_start_us, int(round(sent.start * US)), length))
    return placements, truncated


def is_audio_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS
