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


SPEECH_ABOVE_FLOOR_DB = 10   # khung to hon nen (khung 10% nho nhat) muc nay = dang co tieng noi


def voice_spans(sentences: list, levels, frame_sec: float, audio_duration: float, debug: dict = None) -> dict:
    """Tinh VoiceSpan cho moi cau (theo thu tu trong file voice).
    Diem cat cu (s.start / s.end) nam trong KHOANG LANG THAT do ffmpeg do -> giu lam chuan va chi
    NOI RONG: giua 2 cau, cat o CHO LANG NHAT trong khoang [cuoi cau truoc, dau cau sau] cu
    -> cau sau chi bat dau SOM HON hoac bang, cau truoc chi ket thuc MUON HON hoac bang cach cu
    (khong bao gio te hon), lay lai am dau/cuoi nho bi nguong -35 dB cua ffmpeg tinh la lang.
    KHONG dung moc tung chu cua whisper (lech nhieu voi tieng Phap / A Rap / Nhat: do tren bai 107,
    cat theo moc whisper lam cau cat vao giua tieng noi tang 0 -> 26..55 cau).
    Luc bat dau / het noi do bang nang luong (khung dau tien / cuoi cung co tieng).
    debug (dict, tuy chon): dien them thong tin de ghi log chan doan - "floor" va
    {id(sentence): {...}} (cach chon cho cat, muc dB cho cat...). Khong anh huong ket qua.
    Tra ve {id(sentence): VoiceSpan}."""
    present = sorted((s for s in sentences if s is not None), key=lambda s: s.start)
    if not present:
        return {}
    floor = float(sorted(levels)[len(levels) // 10]) if len(levels) else -100.0
    loud = levels > floor + SPEECH_ABOVE_FLOOR_DB
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

    cuts = []   # cho cat giua cau k va k+1: cho lang nhat trong khoang trong giua 2 diem cat cu
    for k in range(len(present) - 1):
        lo, hi = present[k].end, present[k + 1].start
        cuts.append(valley(lo, hi) if hi > lo else (lo + hi) / 2)
    spans = {}
    for k, s in enumerate(present):
        a = cuts[k - 1] if k > 0 else max(0.0, s.start - EXTEND_SEC)
        b = cuts[k] if k + 1 < len(present) else min(audio_duration, s.end + EXTEND_SEC)
        a, b = min(a, s.start), max(b, s.end)
        # luc bat dau noi: khung co tieng dau tien tu a (khong muon hon diem cat cu 0.1s)
        fa, fb = frame(a), frame(max(a, s.start + 0.1))
        on = next((i for i in range(fa, fb + 1) if loud[i]), None)
        speech_start = on * frame_sec if on is not None else s.start
        # luc het noi: khung co tieng cuoi cung truoc b (khong som hon diem cat cu 0.1s)
        fa, fb = frame(min(b, s.end - 0.1)), frame(b)
        off = next((i for i in range(fb, fa - 1, -1) if loud[i]), None)
        speech_end = (off + 1) * frame_sec if off is not None else s.end
        speech_start, speech_end = max(a, speech_start), min(b, max(speech_end, speech_start))
        ext_start = max(0.0, min(a, speech_start - EXTEND_SEC))
        ext_end = min(audio_duration, max(b, speech_end + EXTEND_SEC))
        spans[id(s)] = VoiceSpan(a, speech_start, speech_end, b, ext_start, ext_end)
        if debug is not None and len(levels):
            debug[id(s)] = dict(
                cut_start=_cut_how(present, k - 1) if k > 0 else "mep (-0.5s)",
                cut_end=_cut_how(present, k) if k + 1 < len(present) else "mep (+0.5s)",
                db_start=float(levels[frame(a)]) - floor, db_end=float(levels[frame(b)]) - floor,
                onset_by_energy=on is not None, offset_by_energy=off is not None)
    return spans


def _cut_how(present, k) -> str:
    """Cach chon cho cat giua cau k va k+1 trong voice_spans (chi de ghi log)."""
    lo, hi = present[k].end, present[k + 1].start
    return f"valley({hi - lo:.2f}s)" if hi > lo else "giua(hi<=lo)"


def original_onsets(clips: list, source_map: dict, words_by_file: dict, debug: list = None) -> list:
    """Do tre tu mep dau clip goc toi luc giong goc bat dau noi (giay), theo chu dau tien
    (whisper) nam trong clip. Khong co file giong goc -> 0 (coi mep clip la luc bat dau noi).
    debug (list, tuy chon): them nguon cua tung onset de ghi log chan doan."""
    out = []
    for c in clips:
        words = words_by_file.get(source_map.get(c.source_path, ""), [])
        a = c.source_start_us / US
        b = a + c.source_duration_us / US
        inside = [w for w in words if w.text.strip() and a - 0.1 <= (w.start + w.end) / 2 <= b + 0.1]
        out.append(min(max(0.0, inside[0].start - a), c.source_duration_us / US) if inside else 0.0)
        if debug is not None:
            debug.append(f"whisper chu dau '{inside[0].text.strip()[:15]}' @{inside[0].start:.3f}s" if inside
                         else ("mac dinh 0 (khong co chu)" if words else "mac dinh 0 (khong co file giong goc)"))
    return out


def compute_placements(sentences: list, clips: list, timeline_end_us: int,
                       spans: dict = None, onsets: list = None, debug: list = None) -> tuple:
    """Dat cau moi i vao moc cau goc i: LUC BAT DAU NOI cua cau moi = luc bat dau noi cua giong goc
    (mep clip goc + onsets[i]). KHONG BAO GIO CAT CAU, khong doi toc do; tranh 2 giong de nhau
    bang cach DICH VI TRI (quy tac da chot voi nguoi dung 2026-09-29), tinh tren VUNG NGHE DUOC:
      TH1: cau truoc dai de sang cau sau -> cau sau bat dau NOI SAT ngay khi cau truoc
           het, day day chuyen ra sau.
      TH2: cau cuoi vuot qua cuoi video -> keo som len de ket thuc DUNG cuoi video, cau
           nao bi de thi keo som len theo, day chuyen nguoc ve truoc.
      Tong giong dai hon ca video -> PlacementError (bao loi ngon ngu do).
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
    src_start = [int(round(v.audible_start * US)) for v in sp]
    lengths = [max(1, int(round(v.audible_end * US)) - a) for v, a in zip(sp, src_start)]
    lead = [int(round(v.speech_start * US)) - a for v, a in zip(sp, src_start)]    # lang truoc chu dau
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
        # cau dau ra truoc 0 chi vi phan lang truoc chu dau -> bo bot phan lang do (khong dung chu)
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

    # Keo dai moi doan (phan ngoai vung nghe duoc -> tat tieng). Cach 1: dung doan ben canh tren
    # timeline thi rut bot cho vua cham, khong chong nhau.
    head_want = [max(0, a - int(round(v.ext_start * US))) for v, a in zip(sp, src_start)]
    tail_want = [max(0, int(round(v.ext_end * US)) - (a + ln)) for v, a, ln in zip(sp, src_start, lengths)]
    head = list(head_want)
    tail = list(tail_want)
    if n:
        head[0] = min(head[0], starts[0])
        tail[-1] = min(tail[-1], max(0, timeline_end_us - (starts[-1] + lengths[-1])))
    for i in range(n - 1):
        gap = starts[i + 1] - (starts[i] + lengths[i])
        if tail[i] + head[i + 1] > gap:
            tail[i] = min(tail_want[i], max(gap // 2, gap - head_want[i + 1]))
            head[i + 1] = min(head_want[i + 1], gap - tail[i])
    placements = []
    for i in range(n):
        placements.append((starts[i] - head[i], src_start[i] - head[i], lengths[i] + head[i] + tail[i],
                           head[i], tail[i]))
        if debug is not None:
            debug.append(dict(clip=idx[i], onset_us=onset[i], speech_us=starts[i] + lead[i],
                              th1_us=after_th1[i] - origin[i], th2_us=after_th2[i] - after_th1[i],
                              trim_start_us=trimmed0 if i == 0 else 0,
                              head_cut_us=head_want[i] - head[i], tail_cut_us=tail_want[i] - tail[i]))
    return placements, report


def is_audio_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS
