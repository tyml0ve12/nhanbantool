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

SENTENCE_END_RE = re.compile(r"[.!?ã€‚ï¼ï¼Ÿâ€¦]+[\"'â€â€™Â»)\]]*$")

MIN_PAUSE = 0.12          # cho ngat ngan hon muc nay khong lam ranh gioi (giay)
MAX_UNITS = 16            # 1 cau toi da trai qua bao nhieu doan giua 2 cho ngat
REFINE_RADIUS = 1         # buoc so nghia chinh xac: xe dich moi ranh gioi +-1 cho ngat
MERGE_PENALTY = 2.0       # phat khi gop 2 cau goc vao 1 doan (chi gop khi ro rang tot hon)
MERGE_GAP_WEIGHT = 1.5    # + phat moi giay 2 cau goc cach nhau tren timeline (gop -> cau sau phat som hon hinh)
MERGE_FAR_GAP = 10.0      # cau sau KHONG nam ngay sau tren timeline -> coi nhu cach xa 10s
LOG_RATIO_SIGMA = 0.35    # do lech do dai chap nhan (log ti le)
SIM_WEIGHT = 12.0         # trong so do giong nghia
PAUSE_WEIGHT = 1.0        # diem moi giay ngat o ranh gioi
PUNCT_BONUS = 0.4

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
    ma khoang ngat qua ngan de do duoc (vd "â€¦em silÃªncio total. Elas caÃ§amâ€¦") -
    thieu cho cat nay thi 2 cau bi gop nham, cau sau phat som hon hinh."""
    cands = []
    wi = 0
    for s, e in silences:
        if e - s < MIN_PAUSE or s <= t0 or e >= t1:
            continue
        while wi + 1 < len(words) and words[wi + 1].end <= s + 0.35:
            wi += 1
        punct = words[wi].end <= s + 0.35 and bool(SENTENCE_END_RE.search(words[wi].text.strip()))
        cands.append((s, e, PAUSE_WEIGHT * (e - s) + (PUNCT_BONUS if punct else 0.0)))

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
        raise SegmentError("Track gá»‘c khÃ´ng cÃ³ cÃ¢u nÃ o.")
    if not words:
        raise SegmentError("KhÃ´ng nháº­n dáº¡ng Ä‘Æ°á»£c giá»ng nÃ³i nÃ o trong file audio.")

    t0, t1 = _speech_bounds(words, silences, audio_duration)
    cands = _candidates(words, silences, t0, t1)
    if len(cands) < n - 1:
        raise SegmentError(f"Chá»‰ tÃ¬m Ä‘Æ°á»£c {len(cands)} chá»— ngáº¯t nghá»‰, cáº§n Ã­t nháº¥t {n - 1} Ä‘á»ƒ tÃ¡ch {n} cÃ¢u. "
                           "Audio nÃ y cÃ³ thá»ƒ thiáº¿u cÃ¢u hoáº·c Ä‘á»c quÃ¡ liá»n máº¡ch.")

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
        # (vd "Leones del Serengeti." + "Â¿ReinarÃ¡n o caerÃ¡n?") -> can 2:1
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
                on_log(f"So nghÄ©a chÃ­nh xÃ¡c {len(near)} cÃ¡ch cáº¯tâ€¦")
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
        raise SegmentError("KhÃ´ng cÄƒn Ä‘Æ°á»£c cÃ¢u má»›i vá»›i cÃ¢u gá»‘c (Ä‘á»™ dÃ i quÃ¡ chÃªnh lá»‡ch). "
                           "Kiá»ƒm tra audio cÃ³ Ä‘Ãºng ká»‹ch báº£n cá»§a project nÃ y khÃ´ng.")
    ratio = sum(node_end[b] - node_start[a] for a, b, _, _ in steps) / orig_total

    # sentences[k] = cau moi cua cau goc k; cau goc duoc GOP vao cau truoc -> None
    sentences = [None] * n
    expected = [0.0] * n            # do dai goc tuong ung (cong ca cau bi gop)
    merged = []
    for a, b, k, step in steps:
        s, e = node_start[a], node_end[b]
        text = "".join(w.text for w in words if s - 0.3 <= (w.start + w.end) / 2 <= e + 0.1).strip()
        table = sim if step == 1 else sim2
        similarity = float(table[(a, b)][k]) if table else 0.0
        sentences[k] = Sentence(max(0.0, s - PAD_BEFORE), min(audio_duration, e + PAD_AFTER), text, similarity)
        expected[k] = sum(original_durations[k:k + step])
        if step == 2:
            merged.append((labels[k], labels[k + 1]))
    present = [k for k in range(n) if sentences[k] is not None]
    for x, y in zip(present, present[1:]):
        if sentences[x].end > sentences[y].start:
            sentences[x].end = sentences[y].start = (sentences[x].end + sentences[y].start) / 2

    if use_meaning:
        low = sorted(labels[k] for k in present if sentences[k].similarity < LOW_SIMILARITY)
        if len(low) > MAX_LOW_SIMILARITY_SHARE * n:
            raise SegmentError(
                f"{len(low)}/{n} cÃ¢u cÃ³ ná»™i dung khÃ´ng khá»›p cÃ¢u gá»‘c (vd cÃ¢u {', '.join(map(str, low[:5]))}). "
                "Audio nÃ y cÃ³ thá»ƒ thiáº¿u/thá»«a cÃ¢u hoáº·c khÃ´ng cÃ¹ng ká»‹ch báº£n.")
    corr = _correlation([sentences[k].end - sentences[k].start for k in present], [expected[k] for k in present])
    if corr < MIN_CORRELATION:
        raise SegmentError(f"Äá»™ khá»›p Ä‘á»™ dÃ i cÃ¢u má»›i vá»›i cÃ¢u gá»‘c quÃ¡ tháº¥p ({corr:.2f}). "
                           "Audio nÃ y cÃ³ thá»ƒ khÃ´ng cÃ¹ng ká»‹ch báº£n, hoáº·c thiáº¿u/thá»«a cÃ¢u.")
    warnings = []
    if not use_meaning:
        warnings.append("KhÃ´ng cÃ³ ná»™i dung cÃ¢u gá»‘c â†’ chá»‰ cÄƒn theo Ä‘á»™ dÃ i, cÃ³ thá»ƒ lá»‡ch á»Ÿ Ä‘oáº¡n cáº¯t giá»¯a cÃ¢u. "
                        "NÃªn má»Ÿ CapCut nghe kiá»ƒm tra.")
    if corr < WARN_CORRELATION:
        warnings.append(f"Äá»™ khá»›p Ä‘á»™ dÃ i cÃ¢u chá»‰ {corr:.2f} â€” nÃªn má»Ÿ CapCut nghe kiá»ƒm tra.")
    for k in present:
        sent, od = sentences[k], expected[k]
        if use_meaning and sent.similarity < LOW_SIMILARITY:
            warnings.append(f"CÃ¢u {labels[k]}: ná»™i dung Ã­t giá»‘ng cÃ¢u gá»‘c ({sent.similarity:.2f}) â€” nÃªn nghe kiá»ƒm tra.")
        elif abs(math.log(max(0.05, sent.end - sent.start) / (ratio * od))) > OUTLIER_LOG_RATIO:
            warnings.append(f"CÃ¢u {labels[k]} dÃ i {sent.end - sent.start:.1f}s, lá»‡ch nhiá»u so vá»›i cÃ¢u gá»‘c "
                            f"({od:.1f}s) â€” nÃªn nghe kiá»ƒm tra.")
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
