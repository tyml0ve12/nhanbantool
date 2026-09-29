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


def reading_order(clips: list) -> list:
    """Thu tu DOC cua cac cau trong file giong goc (vi tri source_timerange) - co
    the KHAC thu tu tren timeline (vd cau tieu de doc dau tien nhung duoc keo ra
    giua video). Audio ngon ngu moi doc theo cung kich ban -> cung thu tu DOC nay.
    Nhieu file nguon: file nao xuat hien som tren timeline truoc, trong file theo vi tri.
    Tra ve danh sach chi so clip (theo timeline) xep theo thu tu doc."""
    first_seen = {}
    for i, c in enumerate(clips):
        first_seen.setdefault(c.source_path, i)
    return sorted(range(len(clips)), key=lambda i: (first_seen[clips[i].source_path], clips[i].source_start_us, i))


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


FAR_SHIFT_US = 5 * US    # cau bi dich xa moc goc hon muc nay -> canh bao


class PlacementError(Exception):
    pass


@dataclass
class PlacementReport:
    pushed_later: int = 0          # TH1: so cau bi day ra sau
    pulled_earlier: int = 0        # TH2: so cau bi keo som len
    far: list = None               # [(so_thu_tu_cau, lech_giay)] lech > 5s


def compute_placements(sentences: list, clips: list, timeline_end_us: int) -> tuple:
    """Dat cau moi i vao moc bat dau cau goc i. KHONG BAO GIO CAT CAU, khong doi toc do;
    tranh 2 giong de nhau bang cach DICH VI TRI (quy tac da chot voi nguoi dung 2026-09-29):
      TH1: cau truoc dai de sang cau sau -> cau sau bat dau NOI SAT ngay khi cau truoc
           het, day day chuyen ra sau.
      TH2: cau cuoi vuot qua cuoi video -> keo som len de ket thuc DUNG cuoi video, cau
           nao bi de thi keo som len theo, day chuyen nguoc ve truoc.
      Tong giong dai hon ca video -> PlacementError (bao loi ngon ngu do).
    Tra ve ([(target_us, source_us, duration_us)], PlacementReport)."""
    # Cau goc da GOP vao cau truoc (None, can 2:1) -> khong co doan rieng tren timeline
    idx = [i for i, s in enumerate(sentences[:len(clips)]) if s is not None]
    sentences = [sentences[i] for i in idx]
    n = len(sentences)
    lengths = [max(1, int(round((s.end - s.start) * US))) for s in sentences]
    origin = [clips[i].target_start_us for i in idx]
    starts = list(origin)

    # TH1: quet tu dau -> cuoi, day cau bi de ra sau
    for i in range(1, n):
        starts[i] = max(origin[i], starts[i - 1] + lengths[i - 1])

    # TH2: quet tu cuoi -> dau, keo cau vuot cuoi video (va cau bi de) som len
    if n and starts[-1] + lengths[-1] > timeline_end_us:
        starts[-1] = timeline_end_us - lengths[-1]
        for i in range(n - 2, -1, -1):
            if starts[i] + lengths[i] > starts[i + 1]:
                starts[i] = starts[i + 1] - lengths[i]
    if n and starts[0] < 0:
        over = -starts[0] / US
        raise PlacementError(f"Tổng giọng mới dài hơn video {over:.1f}s — không xếp được mà không chồng tiếng. "
                             "Kiểm tra lại audio (thừa phần/ thừa câu?).")

    report = PlacementReport(far=[])
    placements = []
    for i in range(n):
        delta = starts[i] - origin[i]
        if delta > 0:
            report.pushed_later += 1
        elif delta < 0:
            report.pulled_earlier += 1
        if abs(delta) > FAR_SHIFT_US:
            report.far.append((idx[i] + 1, delta / US))
        placements.append((starts[i], int(round(sentences[i].start * US)), lengths[i]))
    return placements, report


def is_audio_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS
