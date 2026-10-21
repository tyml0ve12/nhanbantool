"""Cac buoc xu ly thuan (khong phu thuoc giao dien) - dung chung cho app va CLI."""
import json
import os
import re
from dataclasses import dataclass

import numpy as np

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


def reading_order(clips: list, by_timeline=frozenset()) -> list:
    """Thu tu DOC cua cac cau trong file giong goc (vi tri source_timerange) - co
    the KHAC thu tu tren timeline (vd cau tieu de doc dau tien nhung duoc keo ra
    giua video). Audio ngon ngu moi doc theo cung kich ban -> cung thu tu DOC nay.
    Nhieu file nguon: file nao xuat hien som tren timeline truoc, trong file theo vi tri.
    by_timeline: clip xep theo TIMELINE (ngay sau clip khac dung truoc no tren timeline) - xem short_insert_clips.
    Tra ve danh sach chi so clip (theo timeline) xep theo thu tu doc."""
    first_seen = {}
    for i, c in enumerate(clips):
        first_seen.setdefault(c.source_path, i)
    order = sorted((i for i in range(len(clips)) if i not in by_timeline),
                   key=lambda i: (first_seen[clips[i].source_path], clips[i].source_start_us, i))
    for i in sorted(by_timeline):           # clip i - 1 (ke ca clip chen) da co trong order
        order.insert(order.index(i - 1) + 1 if i else 0, i)
    return order


SHORT_INSERT_MAX_CLIPS = 2


def short_insert_clips(clips: list) -> set:
    """Clip cua FILE PHU CHEN NGAN (vd file "keu goi binh luan" chen giua video, bai 120): file khong phai file doc dau,
    <= SHORT_INSERT_MAX_CLIPS clip, moi clip nam GIUA cac clip cua file chinh (file nhieu clip nhat) tren timeline.
    Giong moi co the khong doc (loi keu goi rieng tung ngon ngu) -> cau TUY CHON, vi tri doc theo timeline."""
    by_file = {}
    for i, c in enumerate(clips):
        by_file.setdefault(c.source_path, []).append(i)
    if len(by_file) < 2:
        return set()
    main = max(by_file.values(), key=len)
    first = clips[reading_order(clips)[0]].source_path
    out = set()
    for path, idx in by_file.items():
        if path != first and idx is not main and len(idx) <= SHORT_INSERT_MAX_CLIPS                 and all(main[0] < i < main[-1] for i in idx):
            out.update(idx)
    return out


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


# Giong moi THUA cau (chot 2026-10-20): cau goc da bi XOA khoi timeline nhung van nam trong file giong goc (vd bai 44
# file 44.1 doan 21.5-27.9s "En Kenia, incluso…") -> giong moi van doc. Nhan ra nho doan co tieng trong file giong
# goc nam giua 2 clip lien nhau (theo vi tri trong file) ma khong thuoc clip nao.
EXTRA_MIN_GAP_SEC = 0.3       # doan ho ngan hon -> bo qua (cho cat giua 2 clip)
EXTRA_MIN_SPEECH_SEC = 1.0    # tong khung co tieng (nang luong) trong doan ho
EXTRA_MIN_WORDS = 3           # so chu whisper co trung diem trong doan ho
EXTRA_REREAD_SIM = 0.7        # giong noi dung 1 clip bat ky -> "doc lai / loi voice", khong phai cau thua
EXTRA_REREAD_WARN = 0.55      # cau thua giong 1 clip tu muc nay (duoi EXTRA_REREAD_SIM) -> canh bao "co the doc lai"
EXTRA_SENTENCE_END = re.compile(r"[.!?。！？…]+[\"'”’»)\]]*$")


@dataclass
class ExtraSpeech:
    """Doan co tieng trong file giong goc khong thuoc clip nao."""
    path: str
    start: float              # giay trong file
    end: float
    speech: float             # tong thoi gian co tieng (giay)
    text: str                 # chu whisper trong doan
    after: int                # chi so clip (timeline) ngay TRUOC doan trong file; -1 = dau file
    before: int               # chi so clip ngay SAU; -1 = cuoi file
    kind: str                 # "thua" | "doc lai" | "manh" | "dau file" | "cuoi file"
    reread_sim: float = 0.0   # do giong lon nhat voi noi dung 1 clip (doc lai)


def find_extra_speech(clips: list, source_map: dict, words_by_file: dict, levels_by_file: dict,
                      clip_texts: list, embed=None, frame_sec: float = 0.01) -> list:
    """Tim doan co tieng trong file giong goc khong thuoc clip nao (moi file: truoc clip dau, giua 2 clip lien nhau
    theo vi tri trong file, sau clip cuoi). Phan loai:
      thua      - giua 2 clip, >= EXTRA_MIN_SPEECH_SEC co tieng, >= EXTRA_MIN_WORDS chu, ket bang dau ket cau,
                  khong giong noi dung clip nao (cau goc da xoa khoi timeline -> giong moi co the doc thua);
      doc lai   - giong >= EXTRA_REREAD_SIM noi dung 1 clip (ke ca clip file khac: file loi voice / bo sung) -> bo qua;
      manh      - co tieng nhung ngan / it chu / khong ket cau (cat lem) -> chi canh bao;
      dau file / cuoi file - truoc clip dau / sau clip cuoi cua file -> chi canh bao.
    embed(texts) -> vector (MiniLM); None -> khong kiem "doc lai"."""
    out = []
    by_file = {}
    for i, c in enumerate(clips):
        path = source_map.get(c.source_path, "")
        if path in words_by_file and path in levels_by_file:
            by_file.setdefault(path, []).append(i)
    clip_vec = None
    for path, idx in by_file.items():
        levels = levels_by_file[path]
        if not len(levels):
            continue
        floor = level_floor(levels)
        # nguong co tieng theo dinh CA FILE (doan ho co the chi co lang -> dinh doan ho khong dung lam moc)
        threshold = max(floor + SPEECH_ABOVE_FLOOR_DB, float(np.percentile(levels, 95)) - SPEECH_BELOW_PEAK_DB)
        idx = sorted(idx, key=lambda i: clips[i].source_start_us)
        edges = [(-1, 0.0)] + [(i, (clips[i].source_start_us + clips[i].source_duration_us) / US) for i in idx]
        starts = [(i, clips[i].source_start_us / US) for i in idx] + [(-1, len(levels) * frame_sec)]
        covered_to = 0.0
        for (after, gap_start), (before, gap_end) in zip(edges, starts):
            gap_start = max(gap_start, covered_to)
            if after >= 0:
                covered_to = max(covered_to, (clips[after].source_start_us + clips[after].source_duration_us) / US)
            if gap_end - gap_start < EXTRA_MIN_GAP_SEC:
                continue
            fa, fb = int(round(gap_start / frame_sec)), min(len(levels) - 1, int(round(gap_end / frame_sec)))
            speech = float((levels[fa:fb + 1] > threshold).sum()) * frame_sec
            ws = [w for w in words_by_file[path] if w.text.strip() and gap_start <= (w.start + w.end) / 2 < gap_end]
            text = "".join(w.text for w in ws).strip()
            if speech < 0.3:      # chi chu whisper (moc chu som / lech) ma khong co tieng -> khong phai doan noi
                continue
            item = ExtraSpeech(path, gap_start, gap_end, speech, text, after, before, "")
            if after < 0:
                item.kind = "dau file"
            elif before < 0:
                item.kind = "cuoi file"
            elif speech < EXTRA_MIN_SPEECH_SEC or len(ws) < EXTRA_MIN_WORDS or not EXTRA_SENTENCE_END.search(text):
                item.kind = "manh"
            else:
                item.kind = "thua"
                if embed is not None and text:
                    if clip_vec is None:
                        clip_vec = embed([t or "" for t in clip_texts])
                    v = embed([text])[0]
                    item.reread_sim = float(max(clip_vec @ v)) if len(clip_vec) else 0.0
                    if item.reread_sim >= EXTRA_REREAD_SIM:
                        item.kind = "doc lai"
            out.append(item)
    return out


def _text_of(material: dict) -> str:
    content = material.get("content", "")
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return str(data.get("text", ""))
    except (ValueError, TypeError):
        pass
    return str(content or material.get("recognize_text", ""))


def _visible_text(material: dict, duration_us: int) -> str:
    """Chu THUC SU hien trong doan phu de. CapCut cat doi 1 dong phu de thi ca 2
    nua van giu nguyen ca cau trong "text", chi danh dau bang moc tung chu (words):
    chu co thoi luong 0 hoac bat dau sau khi doan ket thuc la chu bi an.
    Vd "o Sahara cuoc song khong duoc cho di" cat doi -> "o Sahara cuoc song" | "song khong duoc cho di"."""
    words = material.get("words") or {}
    texts, starts, ends = words.get("text") or [], words.get("start_time") or [], words.get("end_time") or []
    if not texts or not (len(texts) == len(starts) == len(ends)):
        return _text_of(material)
    limit_ms = duration_us / 1000
    shown = [t for t, a, b in zip(texts, starts, ends) if b > a and a < limit_ms]
    return " ".join(shown) if shown else _text_of(material)


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
                text = _visible_text(texts_by_id.get(seg.get("material_id"), {}), int(tr.get("duration", 0))).strip()
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


EXTEND_SEC = 0.5      # moi doan voice keo dai them toi da 0.5s truoc chu dau / sau chu cuoi (nguoi dung chot 2026-10-19)


@dataclass
class VoiceSpan:
    """Vung cua 1 cau trong file voice (giay). Ngoai [audible_start, audible_end] la tieng cau ben canh
    hoac lang -> TAT TIENG (am luong 0) neu nam trong phan keo dai."""
    audible_start: float     # cho lang nhat giua cau truoc va cau nay (bat dau nghe duoc)
    speech_start: float      # chu dau tien bat dau
    speech_end: float        # chu cuoi cung ket thuc
    audible_end: float       # cho lang nhat giua cau nay va cau sau (het nghe duoc)
    ext_start: float         # mep doan keo dai (speech_start - 0.5s)
    ext_end: float           # (speech_end + 0.5s)
    # Khung con AM (> SOUND_BELOW_PEAK_DB) dau tien / cuoi cung trong vung nghe duoc: am nhe / tieng bat cuoi chu
    # thap hon nguong luc noi. LOI cua TH1 luon bao tron [sound_start, sound_end] -> khong bao gio rut mat am.
    sound_start: float = None
    sound_end: float = None


SPEECH_ABOVE_FLOOR_DB = 10   # khung to hon nen (khung 10% nho nhat cua file) muc nay moi co the la tieng noi
# Nguong "co tieng" CHUNG cho giong goc va giong moi: max(nen + 10 dB, dinh cua vung - 25 dB), dinh = phan vi 95
# cua vung (clip goc / vung nghe duoc cua cau). Khong phu thuoc nen lang so (giong goc 41 nen -54 dBFS, giong
# Anh -70 dBFS: cung "nen + 10 dB" thi giong Anh nhay hon 16 dB, bat ca tieng tho). Do 2026-10-20 (132u, 41, 107,
# so voi track nguoi dung tay 107): lech > 0.15s con 6/131 cau (nen + 10 dB: 22/131, whisper: 111/131).
SPEECH_BELOW_PEAK_DB = 25
SPEECH_MIN_FRAMES = 3        # >= 3 khung 10ms lien tiep co tieng moi tinh (bo tieng tach / tho ngan)
TH1_MARGIN_SEC = 0.15        # TH1/TH2 tinh tren tieng noi + le 0.15s moi ben (khong phai ca vung nghe duoc)
# ... nhung LOI luon bao tron moi khung con am (> dinh - 35 dB, 1 khung du) + 50ms: duoi am thap / tieng bat
# ngan (< 3 khung) sau luc het noi khong bi rut mat khi TH1 phai rut phan nghe duoc (QA 8-A: 107 day cau 9 mat "d").
SOUND_BELOW_PEAK_DB = 35
SOUND_PAD_SEC = 0.05


def speech_threshold(levels, floor: float, fa: int, fb: int) -> float:
    """Nguong co tieng cho vung khung [fa, fb] (xem SPEECH_BELOW_PEAK_DB)."""
    seg = levels[fa:fb + 1]
    peak = float(np.percentile(seg, 95)) if len(seg) else floor
    return max(floor + SPEECH_ABOVE_FLOOR_DB, peak - SPEECH_BELOW_PEAK_DB)


def first_speech(levels, threshold: float, fa: int, fb: int):
    """Khung dau tien cua doan >= SPEECH_MIN_FRAMES khung lien tiep vuot nguong trong [fa, fb]; None neu khong co."""
    run = 0
    for i in range(fa, fb + 1):
        run = run + 1 if levels[i] > threshold else 0
        if run >= SPEECH_MIN_FRAMES:
            return i - SPEECH_MIN_FRAMES + 1
    return None


def last_speech(levels, threshold: float, fa: int, fb: int):
    """Khung cuoi cung cua doan >= SPEECH_MIN_FRAMES khung lien tiep vuot nguong trong [fa, fb]; None neu khong co."""
    run = 0
    for i in range(fb, fa - 1, -1):
        run = run + 1 if levels[i] > threshold else 0
        if run >= SPEECH_MIN_FRAMES:
            return i + SPEECH_MIN_FRAMES - 1
    return None


def level_floor(levels) -> float:
    """Nen am luong cua ca file: khung 10% nho nhat."""
    return float(sorted(levels)[len(levels) // 10]) if len(levels) else -100.0


def voice_spans(sentences: list, levels, frame_sec: float, audio_duration: float, debug: dict = None) -> dict:
    """Tinh VoiceSpan cho moi cau (theo thu tu trong file voice).
    Diem cat cu (s.start / s.end) nam trong KHOANG LANG THAT do ffmpeg do -> giu lam chuan va chi
    NOI RONG: giua 2 cau, cat o CHO LANG NHAT trong khoang [cuoi cau truoc, dau cau sau] cu
    -> cau sau chi bat dau SOM HON hoac bang, cau truoc chi ket thuc MUON HON hoac bang cach cu
    (khong bao gio te hon), lay lai am dau/cuoi nho bi nguong -35 dB cua ffmpeg tinh la lang.
    KHONG dung moc tung chu cua whisper (lech nhieu voi tieng Phap / A Rap / Nhat: do tren bai 107,
    cat theo moc whisper lam cau cat vao giua tieng noi tang 0 -> 26..55 cau).
    Luc bat dau / het noi do bang nang luong (speech_threshold, cung thuoc voi giong goc), tim trong CA cau:
    ranh gioi doi theo moc chu whisper (_refine_boundaries) co the som hon tieng that > 0.1s (132 cau 44:
    lay diem cat cu lam luc bat dau noi -> dat muon 0.31s).
    debug (dict, tuy chon): dien them thong tin de ghi log chan doan - "floor" va
    {id(sentence): {...}} (cach chon cho cat, muc dB cho cat...). Khong anh huong ket qua.
    Tra ve {id(sentence): VoiceSpan}."""
    present = sorted((s for s in sentences if s is not None), key=lambda s: s.start)
    if not present:
        return {}
    floor = level_floor(levels)
    if debug is not None:
        debug["floor"] = floor

    def frame(t):
        return min(len(levels) - 1, max(0, int(round(t / frame_sec))))

    def valley(t0, t1):
        """Cho cat trong khoang lang [t0, t1]: trong cac khung du lang (<= lang nhat + 3 dB), chon khung
        GAN GIUA khoang nhat -> ca cau truoc lan cau sau deu con le (khong dinh sat mep cau nao)."""
        a, b = frame(t0), frame(t1)
        if b <= a:
            return t0
        seg = levels[a:b + 1]
        quiet = [i for i in range(len(seg)) if seg[i] <= seg.min() + 3]
        mid = (len(seg) - 1) / 2
        return (a + min(quiet, key=lambda i: abs(i - mid))) * frame_sec

    cuts, cut_how = [], []   # cho cat giua cau k va k+1: cho lang nhat trong khoang trong giua 2 diem cat cu
    for k in range(len(present) - 1):
        lo, hi = present[k].end, present[k + 1].start
        cuts.append(valley(lo, hi) if hi > lo else (lo + hi) / 2)
        cut_how.append(f"valley({hi - lo:.2f}s)" if hi > lo else "giua(hi<=lo)")
    spans = {}
    for k, s in enumerate(present):
        a = cuts[k - 1] if k > 0 else max(0.0, s.start - EXTEND_SEC)
        b = cuts[k] if k + 1 < len(present) else min(audio_duration, s.end + EXTEND_SEC)
        a, b = min(a, s.start), max(b, s.end)
        threshold = speech_threshold(levels, floor, frame(a), frame(b)) if len(levels) else 0.0
        # luc bat dau noi: doan co tieng dau tien tu a toi het cau; luc het noi: doan co tieng cuoi cung tu dau cau toi b
        on = first_speech(levels, threshold, frame(a), frame(s.end)) if len(levels) else None
        speech_start = on * frame_sec if on is not None else s.start
        off = last_speech(levels, threshold, frame(s.start), frame(b)) if len(levels) else None
        speech_end = (off + 1) * frame_sec if off is not None else s.end
        speech_start, speech_end = max(a, speech_start), min(b, max(speech_end, speech_start))
        ext_start = max(0.0, min(a, speech_start - EXTEND_SEC))
        ext_end = min(audio_duration, max(b, speech_end + EXTEND_SEC))
        # con am (nguong thap hon, 1 khung du): tu dau vung toi luc bat dau noi / tu luc het noi toi cuoi vung
        # (khong ap nen + 10 dB: file nen cao -> moi khung deu tinh la con am -> loi = ca vung nghe duoc, khong rut gi)
        fa, fs, fe, fb = frame(a), frame(speech_start), frame(speech_end), frame(b)
        sound = float(np.percentile(levels[fa:fb + 1], 95)) - SOUND_BELOW_PEAK_DB if len(levels) else 0.0
        first = next((i for i in range(fa, fs) if levels[i] > sound), None) if len(levels) else None
        last = next((i for i in range(fb, fe - 1, -1) if levels[i] > sound), None) if len(levels) else None
        sound_start = min(speech_start, first * frame_sec) if first is not None else speech_start
        sound_end = max(speech_end, (last + 1) * frame_sec) if last is not None else speech_end
        spans[id(s)] = VoiceSpan(a, speech_start, speech_end, b, ext_start, ext_end, sound_start, sound_end)
        if debug is not None and len(levels):
            debug[id(s)] = dict(
                cut_start=cut_how[k - 1] if k > 0 else "mep (-0.5s)",
                cut_end=cut_how[k] if k + 1 < len(present) else "mep (+0.5s)",
                db_start=float(levels[frame(a)]) - floor, db_end=float(levels[frame(b)]) - floor,
                threshold=threshold - floor, onset_by_energy=on is not None, offset_by_energy=off is not None)
    return spans


def original_onsets(clips: list, source_map: dict, words_by_file: dict, debug: list = None,
                    levels_by_file: dict = None, frame_sec: float = 0.01) -> list:
    """Do tre tu mep dau clip goc toi luc giong goc bat dau noi (giay).
    Co muc am luong file giong goc (levels_by_file) -> do bang NANG LUONG trong clip, CUNG thuoc voi giong moi
    (speech_threshold). Moc chu dau whisper chi la DU PHONG khi clip khong co doan nao du to: whisper som co he
    thong ~0.2-0.3s va bi kep ve 0 khi chu dau bat dau truoc mep clip (132: 62/76 clip, 41: 57/124).
    Khong co file giong goc -> 0 (coi mep clip la luc bat dau noi).
    debug (list, tuy chon): them nguon cua tung onset de ghi log chan doan."""
    out = []
    floors = {}
    for c in clips:
        path = source_map.get(c.source_path, "")
        words = words_by_file.get(path, [])
        a = c.source_start_us / US
        b = a + c.source_duration_us / US
        inside = [w for w in words if w.text.strip() and a - 0.1 <= (w.start + w.end) / 2 <= b + 0.1]
        whisper = (f"whisper '{inside[0].text.strip()[:15]}' @{inside[0].start:.3f}s"
                   + (" (kep ve 0)" if inside[0].start < a else "")) if inside else "whisper khong co chu"
        levels = (levels_by_file or {}).get(path)
        on = None
        if levels is not None and len(levels):
            if path not in floors:
                floors[path] = level_floor(levels)
            fa = min(len(levels) - 1, max(0, int(round(a / frame_sec))))
            fb = min(len(levels) - 1, max(0, int(round(b / frame_sec))))
            on = first_speech(levels, speech_threshold(levels, floors[path], fa, fb), fa, fb)
        if on is not None:
            out.append(min(max(0.0, on * frame_sec - a), c.source_duration_us / US))
            src = f"nang luong; {whisper}"
        elif inside:
            out.append(min(max(0.0, inside[0].start - a), c.source_duration_us / US))
            src = f"du phong {whisper}"
        else:
            out.append(0.0)
            src = "mac dinh 0 (khong co chu)" if words else "mac dinh 0 (khong co file giong goc)"
        if debug is not None:
            debug.append(src)
    return out


def stack_placements(placements: list) -> list:
    """Track phu (cau thua, da tat tieng): doan nao chong doan truoc thi DAY ra sau cho vua cham (cat tu file giu
    nguyen) - track phu duoc lech vi tat tieng, mien khong chong (validate_dubbed)."""
    out = []
    for p in sorted(placements):
        p = list(p)
        if out and p[0] < out[-1][0] + out[-1][2]:
            p[0] = out[-1][0] + out[-1][2]
        out.append(tuple(p))
    return out


def compute_placements(sentences: list, clips: list, timeline_end_us: int,
                       spans: dict = None, onsets: list = None, debug: list = None) -> tuple:
    """Dat cau moi i vao moc cau goc i: LUC BAT DAU NOI cua cau moi = luc bat dau noi cua giong goc
    (mep clip goc + onsets[i]). KHONG BAO GIO CAT CAU, khong doi toc do; tranh 2 giong de nhau
    bang cach DICH VI TRI (quy tac da chot voi nguoi dung 2026-09-29), tinh tren LOI = tieng noi + le
    TH1_MARGIN_SEC moi ben (giao vung nghe duoc):
      TH1: cau truoc dai de sang cau sau -> cau sau bat dau NOI SAT ngay khi cau truoc
           het, day day chuyen ra sau.
      TH2: cau cuoi vuot qua cuoi video -> keo som len de ket thuc DUNG cuoi video, cau
           nao bi de thi keo som len theo, day chuyen nguoc ve truoc.
      Tong giong dai hon ca video -> PlacementError (bao loi ngon ngu do).
    (Truoc 2026-10-20 tinh tren ca vung nghe duoc valley->valley: project day cau bi day nhieu gap doi,
    107 day 17 -> 40 cau. Le 0.15s: day 21 cau, 2 cau bi day van ngat >= 0.3s; le 0.1s bat dau mat ngat cau.)
    Phan nghe duoc ngoai loi giu nguyen neu con cho; dung loi cau ben canh thi rut cho vua cham.
    Sau do moi doan keo dai them (toi da EXTEND_SEC quanh chu dau/cuoi); phan keo dai nam ngoai vung
    nghe duoc se TAT TIENG. Dung doan ben canh tren timeline -> rut bot phan keo dai cho vua cham
    (phan rut bot chi la lang / da tat tieng).
    spans=None -> cach cu (cat dung [start, end], khong keo dai).
    debug (list, tuy chon): them 1 dict / placement (chi so clip, TH1/TH2, phan keo dai bi rut...)
    de ghi log chan doan. Khong anh huong ket qua.
    Tra ve ([(target_us, source_us, duration_us, mute_head_us, mute_tail_us)], PlacementReport)."""
    # Cau goc da GOP vao cau truoc (None, can 2:1) -> khong co doan rieng tren timeline
    idx = [i for i, s in enumerate(sentences[:len(clips)]) if s is not None]
    sentences = [sentences[i] for i in idx]
    n = len(sentences)
    if spans is None:
        spans = {id(s): VoiceSpan(s.start, s.start, s.end, s.end, s.start, s.end) for s in sentences}
    sp = [spans[id(s)] for s in sentences]
    margin, pad = int(round(TH1_MARGIN_SEC * US)), int(round(SOUND_PAD_SEC * US))
    aud_start = [int(round(v.audible_start * US)) for v in sp]
    aud_end = [max(a + 1, int(round(v.audible_end * US))) for v, a in zip(sp, aud_start)]
    # LOI = [bat dau noi - le, het noi + le] + moi khung con am (+50ms), trong vung nghe duoc; vi tri tren timeline tinh theo loi
    src_start = [max(a, min(int(round(v.speech_start * US)) - margin,
                            int(round((v.speech_start if v.sound_start is None else v.sound_start) * US)) - pad))
                 for v, a in zip(sp, aud_start)]
    lengths = [max(1, min(e, max(int(round(v.speech_end * US)) + margin,
                                 int(round((v.speech_end if v.sound_end is None else v.sound_end) * US)) + pad)) - s)
               for v, s, e in zip(sp, src_start, aud_end)]
    lead = [int(round(v.speech_start * US)) - s for v, s in zip(sp, src_start)]    # le truoc chu dau
    onset = [clips[i].target_start_us + int(round((onsets[i] if onsets else 0.0) * US)) for i in idx]
    origin = [o - l for o, l in zip(onset, lead)]
    starts = list(origin)

    # TH1: quet tu dau -> cuoi, day cau bi de ra sau
    for i in range(1, n):
        starts[i] = max(origin[i], starts[i - 1] + lengths[i - 1])
    after_th1 = list(starts)

    # TH2: quet tu cuoi -> dau, keo cau vuot cuoi video (va cau bi de) som len
    if n and starts[-1] + lengths[-1] > timeline_end_us:
        starts[-1] = timeline_end_us - lengths[-1]
        for i in range(n - 2, -1, -1):
            if starts[i] + lengths[i] > starts[i + 1]:
                starts[i] = starts[i + 1] - lengths[i]
    after_th2 = list(starts)
    trimmed0 = 0
    if n and starts[0] < 0:
        # cau dau ra truoc 0 chi vi phan le truoc chu dau -> bo bot phan le do (khong dung chu)
        trim = -starts[0]
        trimmed0 = trim
        if trim <= lead[0]:
            src_start[0] += trim
            lengths[0] -= trim
            lead[0] -= trim
            starts[0] = 0
        else:
            over = (trim - lead[0]) / US
            raise PlacementError(f"Tổng giọng mới dài hơn video {over:.1f}s — không xếp được mà không chồng tiếng. "
                                 "Kiểm tra lại audio (thừa phần/ thừa câu?).")

    report = PlacementReport(far=[])
    for i in range(n):
        delta = starts[i] + lead[i] - onset[i]      # lech LUC BAT DAU NOI so voi giong goc
        if delta > 0:
            report.pushed_later += 1
        elif delta < 0:
            report.pulled_earlier += 1
        if abs(delta) > FAR_SHIFT_US:
            report.far.append((idx[i] + 1, delta / US))

    def fit(gap, want_left, want_right):
        """Chia khoang trong gap cho phan muon them cua doan trai / doan phai (dung nhau -> rut cho vua cham)."""
        if want_left + want_right <= gap:
            return want_left, want_right
        left = min(want_left, max(gap // 2, gap - want_right))
        return left, min(want_right, gap - left)

    # Phan nghe duoc ngoai loi (am nhe / lang that truoc-sau tieng noi), roi phan keo dai (tat tieng).
    # Cach 1: dung doan ben canh tren timeline thi rut bot cho vua cham, khong chong nhau.
    aud_head_want = [s - a for s, a in zip(src_start, aud_start)]
    aud_tail_want = [max(0, e - (s + ln)) for s, ln, e in zip(src_start, lengths, aud_end)]
    head_want = [max(0, a - int(round(v.ext_start * US))) for v, a in zip(sp, aud_start)]
    tail_want = [max(0, int(round(v.ext_end * US)) - e) for v, e in zip(sp, aud_end)]
    aud_head, aud_tail, head, tail = list(aud_head_want), list(aud_tail_want), list(head_want), list(tail_want)
    if n:
        aud_head[0] = min(aud_head[0], starts[0])
        head[0] = min(head[0], starts[0] - aud_head[0])
        room = max(0, timeline_end_us - (starts[-1] + lengths[-1]))
        aud_tail[-1] = min(aud_tail[-1], room)
        tail[-1] = min(tail[-1], room - aud_tail[-1])
    for i in range(n - 1):
        gap = starts[i + 1] - (starts[i] + lengths[i])
        aud_tail[i], aud_head[i + 1] = fit(gap, aud_tail_want[i], aud_head_want[i + 1])
        gap -= aud_tail[i] + aud_head[i + 1]
        tail[i], head[i + 1] = fit(gap, tail_want[i], head_want[i + 1])
    placements = []
    for i in range(n):
        before, after = aud_head[i] + head[i], aud_tail[i] + tail[i]
        placements.append((starts[i] - before, src_start[i] - before, lengths[i] + before + after, head[i], tail[i]))
        if debug is not None:
            debug.append(dict(clip=idx[i], onset_us=onset[i], speech_us=starts[i] + lead[i],
                              core_us=(starts[i], starts[i] + lengths[i]), lead_us=lead[i],
                              th1_us=after_th1[i] - origin[i], th2_us=after_th2[i] - after_th1[i],
                              trim_start_us=trimmed0 if i == 0 else 0,
                              aud_head_cut_us=aud_head_want[i] - aud_head[i], aud_tail_cut_us=aud_tail_want[i] - aud_tail[i],
                              head_cut_us=head_want[i] - head[i], tail_cut_us=tail_want[i] - tail[i]))
    return placements, report


def is_audio_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS
