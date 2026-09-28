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

MIN_PAUSE = 0.12          # cho ngat ngan hon muc nay khong lam ranh gioi (giay)
MAX_UNITS = 16            # 1 cau toi da trai qua bao nhieu doan giua 2 cho ngat
REFINE_RADIUS = 1         # buoc so nghia chinh xac: xe dich moi ranh gioi +-1 cho ngat
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
    sentences: list
    correlation: float
    speed_ratio: float
    used_meaning: bool
    warnings: list = field(default_factory=list)


def _speech_bounds(words, silences, duration):
    first, last = words[0].start, words[-1].end
    for s, e in silences:
        if s <= 0.05 and abs(e - first) < 1.0:
            first = e
        if e >= duration - 0.05 and abs(s - last) < 1.0:
            last = s
    return first, last


def _candidates(words, silences, t0, t1):
    """Cac cho ngat trong audio: (het tieng, bat dau tieng lai, diem thuong)."""
    cands = []
    wi = 0
    for s, e in silences:
        if e - s < MIN_PAUSE or s <= t0 or e >= t1:
            continue
        while wi + 1 < len(words) and words[wi + 1].end <= s + 0.35:
            wi += 1
        punct = words[wi].end <= s + 0.35 and bool(SENTENCE_END_RE.search(words[wi].text.strip()))
        cands.append((s, e, PAUSE_WEIGHT * (e - s) + (PUNCT_BONUS if punct else 0.0)))
    return cands


def split_sentences(words, original_durations, silences, audio_duration,
                    original_texts=None, embedder=None, on_log=None) -> SegmentResult:
    n = len(original_durations)
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
    sim = None
    use_meaning = bool(original_texts) and embedder is not None and len(original_texts) == n
    if use_meaning:
        mids = np.array([(w.start + w.end) / 2 for w in words])

        def span_text(a, b):
            lo, hi = np.searchsorted(mids, node_start[a] - 0.1), np.searchsorted(mids, node_end[b] + 0.1)
            return "".join(w.text for w in words[lo:hi]).strip()

        orig_vec = embedder.encode(list(original_texts))
        # Buoc 1 (nhanh): vector moi doan nho giua 2 cho ngat, vector doan dai =
        # ghep cac doan nho (tong token / so token) -> chi chay model ~vai tram lan.
        sums, counts = embedder.encode_token_sums([span_text(u, u + 1) for u in range(last_node)])
        csum = np.vstack([np.zeros((1, sums.shape[1]), np.float32), np.cumsum(sums, 0)])
        ccount = np.concatenate([[0.0], np.cumsum(counts)])
        a_idx = np.array([a for a, _ in spans])
        b_idx = np.array([b for _, b in spans])
        vec = (csum[b_idx] - csum[a_idx]) / np.maximum(ccount[b_idx] - ccount[a_idx], 1)[:, None]
        vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        coarse = vec @ orig_vec.T
        sim = {span: coarse[k] for k, span in enumerate(spans)}
        path = _align_iter(n, node_start, node_end, node_score, original_durations, ratio, sim, orig_total)

        # Buoc 2 (chinh xac): chay model tren nguyen doan cho cac cach cat lan
        # can phuong an buoc 1 (moi ranh gioi xe dich +-1 cho ngat), chon lai.
        if path is not None:
            near = []
            for a0, b0 in zip(path, path[1:]):
                for a in range(a0 - REFINE_RADIUS, a0 + REFINE_RADIUS + 1):
                    for b in range(b0 - REFINE_RADIUS, b0 + REFINE_RADIUS + 1):
                        if 0 <= a < b <= last_node and (a == 0) == (a0 == 0) and (b == last_node) == (b0 == last_node):
                            near.append((a, b))
            near = sorted(set(near))
            if on_log:
                on_log(f"So nghĩa chính xác {len(near)} cách cắt…")
            exact = embedder.encode([span_text(a, b) for a, b in near]) @ orig_vec.T
            sim = {span: exact[k] for k, span in enumerate(near)}
            refined = _align_iter(n, node_start, node_end, node_score, original_durations, ratio, sim, orig_total)
            if refined is not None:
                path = refined
            else:
                sim = {span: coarse[k] for k, span in enumerate(spans)}
    else:
        path = _align_iter(n, node_start, node_end, node_score, original_durations, ratio, None, orig_total)
    if path is None:
        raise SegmentError("Không căn được câu mới với câu gốc (độ dài quá chênh lệch). "
                           "Kiểm tra audio có đúng kịch bản của project này không.")
    ratio = sum(node_end[b] - node_start[a] for a, b in zip(path, path[1:])) / orig_total

    sentences = []
    for i, (a, b) in enumerate(zip(path, path[1:])):
        s, e = node_start[a], node_end[b]
        text = "".join(w.text for w in words if s - 0.3 <= (w.start + w.end) / 2 <= e + 0.1).strip()
        similarity = float(sim[(a, b)][i]) if sim else 0.0
        sentences.append(Sentence(max(0.0, s - PAD_BEFORE), min(audio_duration, e + PAD_AFTER), text, similarity))
    for x, y in zip(sentences, sentences[1:]):
        if x.end > y.start:
            x.end = y.start = (x.end + y.start) / 2

    if use_meaning:
        low = [i + 1 for i, s in enumerate(sentences) if s.similarity < LOW_SIMILARITY]
        if len(low) > MAX_LOW_SIMILARITY_SHARE * n:
            raise SegmentError(
                f"{len(low)}/{n} câu có nội dung không khớp câu gốc (vd câu {', '.join(map(str, low[:5]))}). "
                "Audio này có thể thiếu/thừa câu hoặc không cùng kịch bản.")
    corr = duration_correlation(sentences, original_durations)
    if corr < MIN_CORRELATION:
        raise SegmentError(f"Độ khớp độ dài câu mới với câu gốc quá thấp ({corr:.2f}). "
                           "Audio này có thể không cùng kịch bản, hoặc thiếu/thừa câu.")
    warnings = []
    if not use_meaning:
        warnings.append("Không có nội dung câu gốc → chỉ căn theo độ dài, có thể lệch ở đoạn cắt giữa câu. "
                        "Nên mở CapCut nghe kiểm tra.")
    if corr < WARN_CORRELATION:
        warnings.append(f"Độ khớp độ dài câu chỉ {corr:.2f} — nên mở CapCut nghe kiểm tra.")
    for i, (sent, od) in enumerate(zip(sentences, original_durations)):
        if use_meaning and sent.similarity < LOW_SIMILARITY:
            warnings.append(f"Câu {i + 1}: nội dung ít giống câu gốc ({sent.similarity:.2f}) — nên nghe kiểm tra.")
        elif abs(math.log(max(0.05, sent.end - sent.start) / (ratio * od))) > OUTLIER_LOG_RATIO:
            warnings.append(f"Câu {i + 1} dài {sent.end - sent.start:.1f}s, lệch nhiều so với câu gốc "
                            f"({od:.1f}s) — nên nghe kiểm tra.")
    return SegmentResult(sentences, corr, ratio, use_meaning, warnings)


def _align_iter(n, node_start, node_end, node_score, original_durations, ratio, sim, orig_total):
    """Chay DP 2 lan: lan 2 dung ti le toc do tinh lai tu ket qua lan 1."""
    path = None
    for _ in range(2):
        path = _align(n, node_start, node_end, node_score, original_durations, ratio, sim)
        if path is None:
            return None
        ratio = sum(node_end[b] - node_start[a] for a, b in zip(path, path[1:])) / orig_total
    return path


def _align(n, node_start, node_end, node_score, original_durations, ratio, sim):
    """Quy hoach dong: tra ve [0, b1, ..., b_{n-1}, node_cuoi] hoac None."""
    last_node = len(node_start) - 1
    INF = float("inf")
    prev = {0: (0.0, None)}
    back = []
    for i in range(n):
        expected = ratio * original_durations[i]
        cur = {}
        remaining = n - 1 - i
        targets = [last_node] if remaining == 0 else range(1, last_node - remaining + 1)
        for b in targets:
            best, arg = INF, None
            for a in range(max(0, b - MAX_UNITS), b):
                if a not in prev:
                    continue
                length = node_end[b] - node_start[a]
                if length <= 0.2 or (sim is not None and (a, b) not in sim):
                    continue
                dev = math.log(length / expected) / LOG_RATIO_SIGMA
                cost = prev[a][0] + 0.5 * dev * dev - node_score[b]
                if sim is not None:
                    cost -= SIM_WEIGHT * float(sim[(a, b)][i])
                if cost < best:
                    best, arg = cost, a
            if arg is not None:
                cur[b] = (best, arg)
        if not cur:
            return None
        back.append(cur)
        prev = cur
    if last_node not in prev:
        return None
    path = [last_node]
    for i in range(n - 1, -1, -1):
        path.append(back[i][path[-1]][1])
    return list(reversed(path))


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
