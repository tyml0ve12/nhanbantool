"""Tach audio ngon ngu dich thanh DUNG N cau, dung thu tu, khop voi N cau goc.

Van de: file doc lien mach co hang tram cho ngat dai gan bang nhau, va track
goc thuong bi cat GIUA cau (tai cho nguoi doc ban goc ngung) -> khong the chi
dua vao khoang lang hay dau cham cau.

Cach lam - quy hoach dong (giong can song ngu Vecalign), chon (N-1) cho ngat
trong audio moi sao cho tong diem cao nhat, moi cau i duoc cham:
  + NGHIA: do giong nghia giua noi dung cau goc i va doan audio moi (model
    nhung cau da ngon ngu) - yeu to quyet dinh, chong lech cau.
  + DO DAI: do dai doan moi ~ ti_le_toc_do x do dai cau goc i.
  + CHO NGAT: uu tien cat o cho ngat dai / co dau cham cau.
Khong co noi dung cau goc (khong tim thay file giong goc) -> chi dung do dai
va cho ngat, kem canh bao.
"""
import math
import re
from dataclasses import dataclass, field

import numpy as np

SENTENCE_END_RE = re.compile(r"[.!?。！？…]+[\"'”’»)\]]*$")
PHRASE_END_RE = re.compile(r"[.,;:!?…。！？、，»”’\"')\]-]+$")   # tu co dau cau bat ky o cuoi

MIN_PAUSE = 0.12          # cho ngat ngan hon muc nay khong lam ranh gioi (giay)
MAX_UNITS = 16            # 1 cau toi da trai qua bao nhieu doan giua 2 cho ngat
REFINE_RADIUS = 1         # buoc so nghia chinh xac: xe dich moi ranh gioi +-1 cho ngat
MERGE_PENALTY = 2.0       # phat khi gop 2 cau goc vao 1 doan (chi gop khi ro rang tot hon)
MERGE_GAP_WEIGHT = 1.5    # + phat moi giay 2 cau goc cach nhau tren timeline (gop -> cau sau phat som hon hinh)
MERGE_FAR_GAP = 10.0      # cau sau KHONG nam ngay sau tren timeline -> coi nhu cach xa 10s
LOG_RATIO_SIGMA = 0.35    # do lech do dai chap nhan (log ti le)
SIM_WEIGHT = 12.0         # trong so do giong nghia
PAUSE_WEIGHT = 1.0        # diem moi giay ngat o ranh gioi
# Cat sau dau ket cau duoc uu tien ro, cat GIUA cum tu (tu truoc khong co dau cau nao)
# bi phat - thieu 2 muc nay DP hay cat o cho nguoi doc ngung nhe giua cum (vd
# "In the | Sahara") cho khop do dai -> 1-3 chu dau cau sau dinh vao cau truoc.
# Do tren project 232: 0.4 / 0 sai 12 cho, 1.0 / 0.3 (va 1.2 / 0.5) dung ca 12.
PUNCT_BONUS = 1.0
MID_PHRASE_PENALTY = 0.3
MERGE_MAX_INNER_PAUSE = 0.5   # doan gop 2 cau co cho ngung >= muc nay o khoang giua -> nguoi doc van tach 2 cau, khong gop
REFINE_SIM_WEIGHT = 6.0       # tinh chinh ranh gioi: trong so do giong cau con sat ranh gioi voi dau/duoi phu de
REFINE_MIN_GAIN = 0.5         # chi doi ranh gioi khi tot hon ro rang

MIN_CORRELATION = 0.60    # do khop do dai duoi muc nay -> bao loi
WARN_CORRELATION = 0.80
LOW_SIMILARITY = 0.35     # cau co do giong nghia duoi muc nay -> canh bao
MAX_LOW_SIMILARITY_SHARE = 0.10   # qua 10% so cau khong khop noi dung -> bao loi
OUTLIER_LOG_RATIO = 0.70

PAD_BEFORE = 0.05         # giu them truoc tieng dau cau (giay)
PAD_AFTER = 0.15          # giu them sau tieng cuoi cau


class SegmentError(Exception):
    pass


@dataclass
class Sentence:
    start: float        # vi tri cat trong file audio (giay)
    end: float
    text: str
    similarity: float = 0.0


@dataclass
class SegmentResult:
    sentences: list                 # dung N phan tu; None = cau goc da gop vao cau truoc (can 2:1)
    correlation: float
    speed_ratio: float
    used_meaning: bool
    warnings: list = field(default_factory=list)
    merged: list = field(default_factory=list)   # [(nhan cau dau, nhan cau bi gop)]


def _speech_bounds(words, silences, duration):
    first, last = words[0].start, words[-1].end
    for s, e in silences:
        if s <= 0.05 and abs(e - first) < 1.0:
            first = e
        if e >= duration - 0.05 and abs(s - last) < 1.0:
            last = s
    return first, last


def _candidates(words, silences, t0, t1):
    """Cac cho ngat trong audio: (het tieng, bat dau tieng lai, diem thuong).
    Nguon 1: khoang lang do duoc (ffmpeg). Nguon 2: DAU CHAM CAU whisper nhan ra
    ma khoang ngat qua ngan de do duoc (vd "…em silêncio total. Elas caçam…") -
    thieu cho cat nay thi 2 cau bi gop nham, cau sau phat som hon hinh."""
    cands = []
    wi = 0
    for s, e in silences:
        if e - s < MIN_PAUSE or s <= t0 or e >= t1:
            continue
        while wi + 1 < len(words) and words[wi + 1].end <= s + 0.35:
            wi += 1
        before = words[wi].text.strip() if words[wi].end <= s + 0.35 else ""
        score = PAUSE_WEIGHT * (e - s)
        if SENTENCE_END_RE.search(before):
            score += PUNCT_BONUS
        elif before and not PHRASE_END_RE.search(before):
            score -= MID_PHRASE_PENALTY
        cands.append((s, e, score))

    covered = sorted(cands)
    extra = []
    for w, nxt in zip(words, words[1:]):
        if not SENTENCE_END_RE.search(w.text.strip()) or w.end <= t0 or nxt.start >= t1:
            continue
        # da co khoang lang do duoc ngay cho nay -> khong them trung
        if any(s - 0.35 <= w.end <= e + 0.35 for s, e, _ in covered):
            continue
        end, start = w.end, max(w.end, nxt.start)
        extra.append((end, start, PAUSE_WEIGHT * (start - end) + PUNCT_BONUS))
    return sorted(cands + extra)


def split_sentences(words, original_durations, silences, audio_duration,
                    original_texts=None, embedder=None, on_log=None, labels=None,
                    pair_gaps=None) -> SegmentResult:
    """labels: so thu tu hien thi cua tung cau trong canh bao (vd so thu tu tren
    timeline khi cau duoc can theo thu tu doc). Mac dinh 1..N.
    pair_gaps[k]: khoang cach tren timeline (giay) giua cau goc k va k+1 (thu tu doc)
    - dung de phat viec gop 2 cau cach xa nhau."""
    n = len(original_durations)
    labels = labels or list(range(1, n + 1))
    merge_cost = [MERGE_PENALTY + MERGE_GAP_WEIGHT * min(MERGE_FAR_GAP, max(0.0, g))
                  for g in (pair_gaps or [0.0] * max(0, n - 1))]
    words = [w for w in words if w.text.strip()]
    if n <= 0:
        raise SegmentError("Track gốc không có câu nào.")
    if not words:
        raise SegmentError("Không nhận dạng được giọng nói nào trong file audio.")

    t0, t1 = _speech_bounds(words, silences, audio_duration)
    cands = _candidates(words, silences, t0, t1)
    if len(cands) < n - 1:
        raise SegmentError(f"Chỉ tìm được {len(cands)} chỗ ngắt nghỉ, cần ít nhất {n - 1} để tách {n} câu. "
                           "Audio này có thể thiếu câu hoặc đọc quá liền mạch.")

    node_start = [t0] + [c[1] for c in cands] + [t1]     # tieng bat dau sau node
    node_end = [t0] + [c[0] for c in cands] + [t1]       # tieng ket thuc tai node
    node_score = [0.0] + [c[2] for c in cands] + [0.0]
    last_node = len(node_start) - 1

    orig_total = sum(original_durations)
    typical_pause = sorted((c[1] - c[0] for c in cands), reverse=True)[max(0, n - 2)]
    ratio = max(0.1, t1 - t0 - (n - 1) * typical_pause) / orig_total

    # Bo cac doan dai bat hop ly (dai hon 1.8 lan cau goc dai nhat) - giam
    # manh so doan phai so nghia.
    max_len = 1.8 * ratio * max(original_durations)
    spans = [(a, b) for a in range(last_node) for b in range(a + 1, min(last_node, a + MAX_UNITS) + 1)
             if node_end[b] - node_start[a] <= max_len]
    sim = sim2 = None
    use_meaning = bool(original_texts) and embedder is not None and len(original_texts) == n
    if use_meaning:
        mids = np.array([(w.start + w.end) / 2 for w in words])

        def span_text(a, b):
            lo, hi = np.searchsorted(mids, node_start[a] - 0.1), np.searchsorted(mids, node_end[b] + 0.1)
            return "".join(w.text for w in words[lo:hi]).strip()

        texts = list(original_texts)
        orig_vec = embedder.encode(texts)
        # 2 cau goc LIEN NHAU (theo thu tu doc) ma ban dich doc lien thanh 1 cau
        # (vd "Leones del Serengeti." + "¿Reinarán o caerán?") -> can 2:1
        pair_vec = embedder.encode([f"{texts[k]} {texts[k + 1]}" for k in range(n - 1)]) if n > 1 else None
        # Buoc 1 (nhanh): vector moi doan nho giua 2 cho ngat, vector doan dai =
        # ghep cac doan nho (tong token / so token) -> chi chay model ~vai tram lan.
        sums, counts = embedder.encode_token_sums([span_text(u, u + 1) for u in range(last_node)])
        csum = np.vstack([np.zeros((1, sums.shape[1]), np.float32), np.cumsum(sums, 0)])
        ccount = np.concatenate([[0.0], np.cumsum(counts)])
        a_idx = np.array([a for a, _ in spans])
        b_idx = np.array([b for _, b in spans])
        vec = (csum[b_idx] - csum[a_idx]) / np.maximum(ccount[b_idx] - ccount[a_idx], 1)[:, None]
        vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        coarse, coarse2 = vec @ orig_vec.T, (vec @ pair_vec.T if pair_vec is not None else None)
        sim = {span: coarse[k] for k, span in enumerate(spans)}
        sim2 = {span: coarse2[k] for k, span in enumerate(spans)} if coarse2 is not None else None
        steps = _align_iter(n, node_start, node_end, node_score, original_durations, ratio, sim, sim2, orig_total, merge_cost)

        # Buoc 2 (chinh xac): chay model tren nguyen doan cho cac cach cat lan
        # can phuong an buoc 1 (moi ranh gioi xe dich +-1 cho ngat), chon lai.
        if steps is not None:
            near = []
            for a0, b0, _, _ in steps:
                for a in range(a0 - REFINE_RADIUS, a0 + REFINE_RADIUS + 1):
                    for b in range(b0 - REFINE_RADIUS, b0 + REFINE_RADIUS + 1):
                        if 0 <= a < b <= last_node and (a == 0) == (a0 == 0) and (b == last_node) == (b0 == last_node):
                            near.append((a, b))
            near = sorted(set(near))
            if on_log:
                on_log(f"So nghĩa chính xác {len(near)} cách cắt…")
            near_vec = embedder.encode([span_text(a, b) for a, b in near])
            exact = near_vec @ orig_vec.T
            exact2 = near_vec @ pair_vec.T if pair_vec is not None else None
            fine = {span: exact[k] for k, span in enumerate(near)}
            fine2 = {span: exact2[k] for k, span in enumerate(near)} if exact2 is not None else None
            refined = _align_iter(n, node_start, node_end, node_score, original_durations, ratio,
                                  fine, fine2, orig_total, merge_cost)
            if refined is not None:
                steps, sim, sim2 = refined, fine, fine2
    else:
        steps = _align_iter(n, node_start, node_end, node_score, original_durations, ratio, None, None, orig_total, merge_cost)
    if steps is None:
        raise SegmentError("Không căn được câu mới với câu gốc (độ dài quá chênh lệch). "
                           "Kiểm tra audio có đúng kịch bản của project này không.")
    ratio = sum(node_end[b] - node_start[a] for a, b, _, _ in steps) / orig_total

    # sentences[k] = cau moi cua cau goc k; cau goc duoc GOP vao cau truoc -> None
    sentences = [None] * n
    expected = [0.0] * n            # do dai goc tuong ung (cong ca cau bi gop)
    sent_words = [None] * n         # cac tu whisper cua tung cau moi (de tinh chinh ranh gioi)
    merged = []
    joined = []                     # cap doc lien khong ngung da tach tai ranh gioi tu: [(nhan, nhan, chu o cho tach)]
    for a, b, k, step in steps:
        s, e = node_start[a], node_end[b]
        span_words = [w for w in words if s - 0.3 <= (w.start + w.end) / 2 <= e + 0.1]
        if step == 2:
            # 2 cau goc la 2 clip rieng -> van tra ve 2 cau: tach tai ranh gioi tu hop nhat
            pair = _split_joined(span_words, max(0.0, s - PAD_BEFORE), min(audio_duration, e + PAD_AFTER),
                                 original_durations[k:k + 2], ratio,
                                 texts[k:k + 2] if use_meaning else None, embedder if use_meaning else None)
            if pair:
                for j, (ws, start, end) in enumerate(pair):
                    sentences[k + j] = Sentence(start, end, _words_text(ws))
                    sent_words[k + j] = ws
                    expected[k + j] = original_durations[k + j]
                joined.append((labels[k], labels[k + 1], f"{pair[0][0][-1].text.strip()} | {pair[1][0][0].text.strip()}"))
                continue
        table = sim if step == 1 else sim2
        similarity = float(table[(a, b)][k]) if table else 0.0
        sentences[k] = Sentence(max(0.0, s - PAD_BEFORE), min(audio_duration, e + PAD_AFTER), _words_text(span_words), similarity)
        sent_words[k] = span_words
        expected[k] = sum(original_durations[k:k + step])
        if step == 2:
            merged.append((labels[k], labels[k + 1]))
    present = [k for k in range(n) if sentences[k] is not None]
    if use_meaning:
        moved = _refine_boundaries(present, sentences, sent_words, texts, expected, ratio, embedder)
        if on_log and moved:
            on_log(f"Tinh chỉnh ranh giới {moved} chỗ theo nội dung đầu/cuối câu.")
        # do giong nghia theo noi dung cuoi cung (sau khi tach / tinh chinh); cau con gop -> so voi ca 2 cau goc
        vec = embedder.encode([sentences[k].text for k in present])
        ref = embedder.encode([texts[k] + (f" {texts[k + 1]}" if k + 1 < n and sentences[k + 1] is None else "")
                               for k in present])
        for j, k in enumerate(present):
            sentences[k].similarity = float(vec[j] @ ref[j])
    for x, y in zip(present, present[1:]):
        if sentences[x].end > sentences[y].start:
            sentences[x].end = sentences[y].start = (sentences[x].end + sentences[y].start) / 2

    if use_meaning:
        low = sorted(labels[k] for k in present if sentences[k].similarity < LOW_SIMILARITY)
        if len(low) > MAX_LOW_SIMILARITY_SHARE * n:
            raise SegmentError(
                f"{len(low)}/{n} câu có nội dung không khớp câu gốc (vd câu {', '.join(map(str, low[:5]))}). "
                "Audio này có thể thiếu/thừa câu hoặc không cùng kịch bản.")
    corr = _correlation([sentences[k].end - sentences[k].start for k in present], [expected[k] for k in present])
    if corr < MIN_CORRELATION:
        raise SegmentError(f"Độ khớp độ dài câu mới với câu gốc quá thấp ({corr:.2f}). "
                           "Audio này có thể không cùng kịch bản, hoặc thiếu/thừa câu.")
    warnings = [f"Câu {x} + {y} được đọc liền không ngừng → đã tách tại ranh giới từ ({cut}) — nên nghe kiểm tra."
                for x, y, cut in joined]
    if not use_meaning:
        warnings.append("Không có nội dung câu gốc → chỉ căn theo độ dài, có thể lệch ở đoạn cắt giữa câu. "
                        "Nên mở CapCut nghe kiểm tra.")
    if corr < WARN_CORRELATION:
        warnings.append(f"Độ khớp độ dài câu chỉ {corr:.2f} — nên mở CapCut nghe kiểm tra.")
    for k in present:
        sent, od = sentences[k], expected[k]
        if use_meaning and sent.similarity < LOW_SIMILARITY:
            warnings.append(f"Câu {labels[k]}: nội dung ít giống câu gốc ({sent.similarity:.2f}) — nên nghe kiểm tra.")
        elif abs(math.log(max(0.05, sent.end - sent.start) / (ratio * od))) > OUTLIER_LOG_RATIO:
            warnings.append(f"Câu {labels[k]} dài {sent.end - sent.start:.1f}s, lệch nhiều so với câu gốc "
                            f"({od:.1f}s) — nên nghe kiểm tra.")
    return SegmentResult(sentences, corr, ratio, use_meaning, warnings, merged)


def _align_iter(n, node_start, node_end, node_score, original_durations, ratio, sim, sim2, orig_total, merge_cost):
    """Chay DP 2 lan: lan 2 dung ti le toc do tinh lai tu ket qua lan 1."""
    steps = None
    for _ in range(2):
        steps = _align(n, node_start, node_end, node_score, original_durations, ratio, sim, sim2, merge_cost)
        if steps is None:
            return None
        ratio = sum(node_end[b] - node_start[a] for a, b, _, _ in steps) / orig_total
    return steps


def _align(n, node_start, node_end, node_score, original_durations, ratio, sim, sim2, merge_cost):
    """Quy hoach dong. Moi buoc = 1 doan audio moi (node a -> node b) phu:
      - 1 cau goc (binh thuong), hoac
      - 2 cau goc LIEN NHAU (can 2:1, chi khi co so nghia): ban dich doc lien 2 cau.
    Tra ve [(a, b, cau_goc_dau, so_cau_goc)] hoac None."""
    last_node = len(node_start) - 1
    INF = float("inf")
    costs = [{0: (0.0, None, 0)}]          # costs[i][node] = (chi phi, node truoc, so cau goc cua buoc)
    for i in range(1, n + 1):
        cur = {}
        targets = [last_node] if i == n else range(1, last_node)
        for step in (1, 2):
            if i - step < 0 or (step == 2 and sim2 is None):
                continue
            prev = costs[i - step]
            if not prev:
                continue
            k = i - step
            expected = ratio * sum(original_durations[k:i])
            table = sim if step == 1 else sim2
            for b in targets:
                best = cur.get(b, (INF,))[0]
                for a in range(max(0, b - MAX_UNITS), b):
                    pa = prev.get(a)
                    if pa is None:
                        continue
                    length = node_end[b] - node_start[a]
                    if length <= 0.2:
                        continue
                    if table is not None:
                        v = table.get((a, b))
                        if v is None:
                            continue
                    dev = math.log(length / expected) / LOG_RATIO_SIGMA
                    cost = pa[0] + 0.5 * dev * dev - node_score[b]
                    if table is not None:
                        # doan gop phu 2 cau goc -> tinh diem nghia cho ca 2 (so sanh cong bang voi tach rieng)
                        cost -= SIM_WEIGHT * float(v[k]) * step
                    if step == 2:
                        if _inner_pause(a, b, node_start, node_end) >= MERGE_MAX_INNER_PAUSE:
                            continue
                        cost += merge_cost[k]
                    if cost < best:
                        best = cost
                        cur[b] = (cost, a, step)
        if not cur:
            return None
        costs.append(cur)
    if last_node not in costs[n]:
        return None
    steps, i, b = [], n, last_node
    while i > 0:
        _, a, step = costs[i][b]
        steps.append((a, b, i - step, step))
        i, b = i - step, a
    return list(reversed(steps))


def _words_text(ws) -> str:
    return "".join(w.text for w in ws).strip()


def _len_cost(length, original, ratio) -> float:
    return 0.5 * (math.log(max(0.05, length) / (ratio * original)) / LOG_RATIO_SIGMA) ** 2


def _boundary_bonus(before: str, gap: float) -> float:
    """Diem cho 1 cho cat giua 2 tu (khong can khoang lang): ngung + dau cau."""
    score = PAUSE_WEIGHT * max(0.0, gap)
    if SENTENCE_END_RE.search(before):
        score += PUNCT_BONUS
    elif PHRASE_END_RE.search(before):
        score += PUNCT_BONUS / 2
    else:
        score -= MID_PHRASE_PENALTY
    return score


def _cut_times(left_last, right_first):
    """Moc cat giua 2 tu lien nhau: chia doi khoang ngung (co the = 0), toi da bang le thuong."""
    gap = max(0.0, right_first.start - left_last.end)
    return left_last.end + min(PAD_AFTER, gap / 2), right_first.start - min(PAD_BEFORE, gap / 2)


def _split_joined(ws, start, end, originals, ratio, pair_texts, embedder):
    """2 cau goc (2 clip rieng) ma ban dich doc LIEN, khong co khoang lang de cat ->
    van tach thanh 2 cau tai ranh gioi tu hop nhat: do dai 2 phan ~ 2 cau goc,
    ngung/dau cau o cho cat, va (neu co) noi dung 2 phan giong 2 cau goc.
    Tra ve [(tu_phan_1, start, end), (tu_phan_2, start, end)] hoac None."""
    if len(ws) < 2:
        return None
    cuts = range(1, len(ws))
    sims = None
    if embedder is not None and pair_texts and all(t.strip() for t in pair_texts):
        ref = embedder.encode(list(pair_texts))
        left = embedder.encode([_words_text(ws[:i]) for i in cuts])
        right = embedder.encode([_words_text(ws[i:]) for i in cuts])
        sims = [float(left[j] @ ref[0]) + float(right[j] @ ref[1]) for j in range(len(cuts))]
    best = None
    for j, i in enumerate(cuts):
        len1, len2 = ws[i - 1].end - ws[0].start, ws[-1].end - ws[i].start
        if len1 <= 0.2 or len2 <= 0.2:
            continue
        cost = (_len_cost(len1, originals[0], ratio) + _len_cost(len2, originals[1], ratio)
                - _boundary_bonus(ws[i - 1].text.strip(), ws[i].start - ws[i - 1].end))
        if sims:
            cost -= SIM_WEIGHT * sims[j]
        if best is None or cost < best[0]:
            best = (cost, i)
    if best is None:
        return None
    i = best[1]
    cut_end, cut_start = _cut_times(ws[i - 1], ws[i])
    return [(ws[:i], start, cut_end), (ws[i:], cut_start, end)]


def _chunks(ws):
    """Chia cac tu cua 1 cau moi thanh cac cau con theo dau ket cau."""
    out, cur = [], []
    for w in ws:
        cur.append(w)
        if SENTENCE_END_RE.search(w.text.strip()):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _refine_boundaries(present, sentences, sent_words, texts, expected, ratio, embedder) -> int:
    """Buoc so nghia chinh tinh diem ca doan dai voi ca doan dai -> kem nhay voi 1 cau
    con o sat ranh gioi (vd "Towering around the water are rows of date palms." thuoc
    DAU cau sau nhung bi xep vao CUOI cau truoc). Voi tung ranh gioi, thu dua cau con
    cuoi cua cau truoc sang cau sau (hoac nguoc lai), so cau con do voi DUOI phu de cau
    truoc va DAU phu de cau sau (cung so chu), cong them thay doi do dai; doi neu ro rang tot hon.
    Tra ve so ranh gioi da doi."""
    moved = 0
    for x, y in zip(present, present[1:]):
        xw, yw = sent_words[x], sent_words[y]
        sub_x, sub_y = texts[x].split(), texts[y].split()
        if not xw or not yw or not sub_x or not sub_y:
            continue
        cx, cy = _chunks(xw), _chunks(yw)
        dur_x, dur_y = xw[-1].end - xw[0].start, yw[-1].end - yw[0].start
        words_per_sec = (len(sub_x) + len(sub_y)) / max(0.5, dur_x + dur_y)
        base_len = _len_cost(dur_x, expected[x], ratio) + _len_cost(dur_y, expected[y], ratio)
        options = []
        if len(cx) >= 2:    # cau con cuoi cua x -> dau y
            options.append((cx[-1], -1, [w for c in cx[:-1] for w in c], cx[-1] + yw))
        if len(cy) >= 2:    # cau con dau cua y -> cuoi x
            options.append((cy[0], 1, xw + cy[0], [w for c in cy[1:] for w in c]))
        best = None
        for chunk, direction, new_x, new_y in options:
            m = max(3, round(words_per_sec * (chunk[-1].end - chunk[0].start)))
            vec = embedder.encode([_words_text(chunk), " ".join(sub_x[-m:]), " ".join(sub_y[:m])])
            tail, head = float(vec[0] @ vec[1]), float(vec[0] @ vec[2])
            sim_gain = (head - tail) if direction < 0 else (tail - head)
            new_len = (_len_cost(new_x[-1].end - new_x[0].start, expected[x], ratio)
                       + _len_cost(new_y[-1].end - new_y[0].start, expected[y], ratio))
            gain = REFINE_SIM_WEIGHT * sim_gain + (base_len - new_len)
            if gain > REFINE_MIN_GAIN and (best is None or gain > best[0]):
                best = (gain, new_x, new_y)
        if best is None:
            continue
        _, new_x, new_y = best
        sent_words[x], sent_words[y] = new_x, new_y
        cut_end, cut_start = _cut_times(new_x[-1], new_y[0])
        sentences[x].end, sentences[x].text = cut_end, _words_text(new_x)
        sentences[y].start, sentences[y].text = cut_start, _words_text(new_y)
        moved += 1
    return moved


def _inner_pause(a, b, node_start, node_end) -> float:
    """Cho ngung dai nhat ben trong doan a->b, chi xet khoang giua (20-80%) doan.
    Doan gop 2 cau goc ma nguoi doc ngung ro o giua = van doc thanh 2 cau.
    (Diem nghia cua doan gop luon cao hon 2 manh rieng -> khong co luat nay DP
    hay gop nham khi cau goc bi cat giua cau.)"""
    span = node_end[b] - node_start[a]
    best = 0.0
    for x in range(a + 1, b):
        if 0.2 <= (node_end[x] - node_start[a]) / span <= 0.8:
            best = max(best, node_start[x] - node_end[x])
    return best


def _correlation(a, b) -> float:
    n = len(a)
    if n < 3 or n != len(b):
        return 1.0
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else 1.0


def duration_correlation(sentences, original_durations) -> float:
    """He so tuong quan do dai cau moi vs cau goc - thap = co the lech cau."""
    a = [s.end - s.start for s in sentences]
    b = list(original_durations)
    n = len(a)
    if n < 3 or n != len(b):
        return 1.0
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else 1.0
