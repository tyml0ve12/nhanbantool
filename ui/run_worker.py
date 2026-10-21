"""Luong chay nen cho nut "Chay": nhan dang -> tach cau -> xep timeline ->
xuat mp4 (tung ngon ngu) -> ghi draft gop tat ca ngon ngu -> ap dung vao project."""
import logging
import os
import re
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal

from core import audio, draft_patch, logo, parts, pipeline, render, segmenter, speech
from core.capcut_project import ProjectError, US, is_capcut_running
from core.paths import long_path_message, path_too_long

log = logging.getLogger("run_worker")

MODE_DRAFT = "draft"
MODE_EXPORT = "export"
MODE_BOTH = "both"

MIN_SUBTITLE_COVER = 0.8   # phu de phu it hon muc nay -> nhan dang giong goc thay the
EXTRA_ROOM_SEC = 0.3       # cau thua dat tren track chinh khi khoang trong >= loi cau thua + 0.3s (chot 2026-10-20)
TAIL_EARLY_SEC = 30        # doan cuoi voice dat truoc clip cuoi hon 30s -> canh bao do "Duoi voice" (viec #6b)


def output_name(video_path: str, language: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0] or "video"
    safe = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in language)
    return f"{stem}_{safe}.mp4"


@dataclass
class LanguageJob:
    row: int
    name: str
    parts: list              # 1 file, hoac cac phan 1.mp3 ... 10.mp3 (da sap thu tu)
    logo: dict = None        # {"path", "placement"} - chen khi xuat mp4 (draft CapCut khong doi)
    audio_path: str = ""     # file audio thuc dung: file duy nhat, hoac file gop cac phan


@dataclass
class RunConfig:
    project_dir: str
    draft_files: list
    track_index: int
    sentence_count: int
    video_path: str
    output_dir: str
    mode: str
    default_language: str
    whisper_model: str
    device: str
    languages: list          # [LanguageJob]
    render_settings: object = None   # core.render.RenderSettings (hop thoai Xuat video)
    output_paths: dict = None        # {row: duong dan mp4} - ten file do nguoi dung dat


class RunWorker(QThread):
    log = Signal(str)
    language_status = Signal(int, str, str)   # row, text, kind (wait/run/ok/warn/error)
    progress = Signal(int)                     # 0-100 toan bo lan chay
    export_progress = Signal(int, int, str, float)   # row, %, toc do (vd "3.2x"), giay con lai
    finished_all = Signal(int, int, bool)      # so NN ok, so NN loi, draft con cho ap dung

    def __init__(self, config: RunConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._stop = False
        self._models = {}
        self._embedder = None
        self._warned_rows = set()
        self._lang_progress = {}
        self._progress_lock = threading.Lock()
        self._driver_warned = False
        self._order = []
        self._optional = set()      # clip file phu chen ngan (cau tuy chon)
        self._onsets = None
        self._onset_src = None      # nguon onset tung clip (ghi log chi tiet)
        self._text_src = None       # nguon noi dung tung cau goc (ghi log chi tiet)
        self._extras = []           # doan co tieng trong file giong goc khong thuoc clip nao (pipeline.ExtraSpeech)

    def stop(self):
        self._stop = True

    def _stopped(self):
        return self._stop

    # ------------------------------------------------------------------
    def run(self):
        ok = err = 0
        pending_apply = False
        try:
            ok, err, pending_apply = self._run()
        except InterruptedError:
            self.log.emit("Đã dừng theo yêu cầu.")
        except Exception as e:
            log.exception("Loi khong luong truoc")
            self.log.emit(f"LỖI: {e}")
            self.log.emit(traceback.format_exc(limit=3))
        self.finished_all.emit(ok, err, pending_apply)

    def _run(self):
        cfg = self.config
        uses_draft = cfg.mode in (MODE_DRAFT, MODE_BOTH)
        uses_export = cfg.mode in (MODE_EXPORT, MODE_BOTH)

        draft = draft_patch.load_draft(cfg.draft_files[0])
        clips = pipeline.read_original_clips(draft, cfg.track_index)
        timeline_end = int(draft.get("duration", 0)) or (clips[-1].target_start_us + clips[-1].target_duration_us)
        self.log.emit(f"Bắt đầu: {len(cfg.languages)} ngôn ngữ · {len(clips)} câu gốc")

        video_duration = 0.0
        if uses_export:
            video_duration = audio.probe_duration(cfg.video_path)
            timeline_end = min(timeline_end, int(video_duration * US)) if timeline_end else int(video_duration * US)

        self.progress.emit(2)
        source_texts = self._original_texts(draft, clips)
        self.progress.emit(8)
        # Can cau theo THU TU DOC trong file giong goc (co the khac thu tu timeline,
        # vd cau tieu de doc dau tien nhung keo ra giua video), roi dat lai theo timeline.
        # File phu chen ngan (can noi dung de so nghia): xep theo timeline, cau tuy chon.
        self._optional = pipeline.short_insert_clips(clips) if source_texts else set()
        self._order = pipeline.reading_order(clips, self._optional)
        for i in sorted(self._optional):
            self.log.emit(f"Câu {i + 1} thuộc file phụ chèn ngắn '{os.path.basename(clips[i].source_path)}' "
                          "→ xếp theo vị trí trên timeline, voice mới có thể không đọc (câu tùy chọn).")
        moved = sum(1 for k, i in enumerate(self._order) if k != i)
        if moved:
            self.log.emit(f"Track gốc có {moved} câu đặt khác thứ tự đọc (vd câu tiêu đề kéo ra giữa video) "
                          "→ căn theo thứ tự đọc trong file giọng gốc.")
        durations = [clips[i].target_duration_us / US for i in self._order]

        # Che do "Ca hai": ghi + ap dung DRAFT CapCut TRUOC (mo CapCut kiem tra duoc
        # ngay), render mp4 SAU. Che do chi xuat mp4: render chong voi nhan dang
        # ngon ngu tiep theo (GPU nhan dang, CPU/o dia render cung luc).
        draft_first = uses_draft and uses_export
        self._lang_progress = {job.row: 0.0 for job in cfg.languages}
        export_pool = ThreadPoolExecutor(max_workers=2) if uses_export else None
        done = []      # [(job, placements, audio_duration_us, future_export_hoac_None)]
        err = 0
        pending_apply = False
        try:
            for job in cfg.languages:
                if self._stop:
                    raise InterruptedError
                try:
                    placements, dur_us = self._process_language(job, clips, durations, timeline_end,
                                                                source_texts, 0.6 if uses_export else 1.0)
                except InterruptedError:
                    raise
                except Exception as e:
                    # Loi bat ky (vd GPU het bo nho) chi hong ngon ngu nay, cac ngon ngu khac van chay
                    if not isinstance(e, (segmenter.SegmentError, pipeline.PlacementError, audio.AudioError,
                                          ProjectError, OSError)):
                        log.exception("Loi khi xu ly %s", job.name)
                    err += 1
                    self._fail(job, e)
                    continue
                future = None
                if uses_export and not draft_first:
                    future = export_pool.submit(self._export, job, placements, video_duration)
                done.append((job, placements, dur_us, future))

            if draft_first and done:
                pending_apply = self._write_draft(draft, [(j, p, d) for j, p, d, _ in done])
                self.log.emit("Draft CapCut xong — bắt đầu render mp4…")
                for job, placements, dur_us, _ in done:
                    self.language_status.emit(job.row, "Đã ghi draft · chờ render", "run")
                done = [(j, p, d, export_pool.submit(self._export, j, p, video_duration)) for j, p, d, _ in done]

            results = []   # [(job, placements, audio_duration_us)]
            for job, placements, dur_us, future in done:
                if future is not None:
                    try:
                        future.result()
                    except InterruptedError:
                        raise
                    except Exception as e:
                        if not isinstance(e, (audio.AudioError, OSError)):
                            log.exception("Loi khi xuat %s", job.name)
                        err += 1
                        self._fail(job, e)
                        continue
                results.append((job, placements, dur_us))
        finally:
            if export_pool:
                export_pool.shutdown(wait=True, cancel_futures=True)
        ok = len(results)

        if uses_draft and not draft_first and results:
            pending_apply = self._write_draft(draft, results)
        self.progress.emit(100)
        return ok, err, pending_apply

    # ------------------------------------------------------------------
    def _original_texts(self, draft, clips):
        """Noi dung cau goc de so nghia: UU TIEN nhan dang giong goc (cat dung theo tung
        clip), phu de chi de bu cau trong / khi khong tim thay file giong goc.
        Phu de CapCut la ban DICH cua giong goc (them sai so dich), hay vat qua 2 clip,
        bi cat doi, hoac lay nham track (vd track phu de tao tu giong nhan ban). Do tren
        project 232 + 107 (12 ban long tieng): giong goc dung 97.5% cau so voi 94.6%."""
        voice = self._original_voice_texts(clips)
        if voice and sum(1 for t in voice if t) >= 0.8 * len(clips):
            subs, _ = pipeline.subtitle_texts(draft, clips)
            filled = [i + 1 for i, (v, s) in enumerate(zip(voice, subs)) if not v and s]
            texts = [v or s for v, s in zip(voice, subs)]
            self._text_src = ["giong goc" if v else ("phu de (bu)" if s else "") for v, s in zip(voice, subs)]
            empty = sum(1 for t in texts if not t)
            self.log.emit("Nội dung câu gốc: nhận dạng từ giọng gốc"
                          + (f", {len(filled)} câu không nghe được lấy từ phụ đề" if filled else "")
                          + (f", {empty} câu không có nội dung" if empty else "") + ".")
            return texts
        texts, cover = pipeline.subtitle_texts(draft, clips)
        self._text_src = ["phu de" if t else "" for t in texts]
        if cover >= 1.0:
            self.log.emit("Nội dung câu gốc: lấy từ track phụ đề (100% câu có phụ đề).")
            return texts
        missing = [i + 1 for i, t in enumerate(texts) if not t]
        if cover >= MIN_SUBTITLE_COVER:
            if voice:
                # Doan khong co phu de (vd cau tieu de nam o track chu khac) -> lay tu giong goc
                self._text_src = ["phu de" if t else ("giong goc (bu)" if v else "") for t, v in zip(texts, voice)]
                texts = [t or v for t, v in zip(texts, voice)]
                self.log.emit(f"Nội dung câu gốc: phụ đề ({cover:.0%}), {len(missing)} câu không có phụ đề "
                              f"(câu {', '.join(map(str, missing[:6]))}) lấy từ giọng gốc.")
            else:
                self.log.emit(f"Nội dung câu gốc: phụ đề ({cover:.0%}); {len(missing)} câu không có phụ đề "
                              "và không tìm thấy file giọng gốc để bù.")
            return texts
        if voice:
            self.log.emit(f"Phụ đề chỉ phủ {cover:.0%} câu → dùng nội dung nhận dạng từ giọng gốc.")
            self._text_src = ["giong goc" if v else "" for v in voice]
            return voice
        self.log.emit("Không có phụ đề đủ và không tìm thấy file giọng gốc → chỉ căn theo độ dài câu "
                      "(kém chính xác hơn).")
        self._text_src = None
        return None

    def _original_voice_texts(self, clips):
        """Noi dung tung cau goc nhan dang tu file giong goc (theo source_timerange). None neu khong co file.
        Dong thoi ghi self._onsets: luc giong goc bat dau noi trong tung clip (de can cau moi)."""
        self._onsets = None
        self._onset_src = None
        self._extras = []
        # Tim trong project truoc; thu muc cha (vd CapCut Drafts chua ca tram project) sau cung
        search_roots = [self.config.project_dir, os.path.dirname(self.config.project_dir)]
        sources = pipeline.resolve_original_sources(clips, search_roots)
        words_by_file = {}
        for path in sorted({p for p in sources.values() if p}):
            words, lang, _ = speech.transcribe_words(path, self.config.whisper_model, self.config.device,
                                                     lambda p: None, self.log.emit, self._stopped, self._models)
            words_by_file[path] = words
            self.log.emit(f"Đã nhận dạng giọng gốc {os.path.basename(path)} ({lang}).")
            log.info("Giong goc %s: %s, %d tu", path, lang, len(words))
        missing = sorted(os.path.basename(p.replace("\\", "/")) for p, real in sources.items() if not real)
        if missing:
            log.info("Khong tim thay file giong goc: %s", ", ".join(missing))
        if not words_by_file:
            return None
        # Luc giong goc bat dau noi: do bang nang luong (cung thuoc voi giong moi), whisper chi de du phong
        levels_by_file = {}
        for path in words_by_file:
            try:
                levels_by_file[path] = audio.levels_db(path)
            except audio.AudioError as e:
                log.warning("Khong do duoc am luong giong goc %s: %s -> dung moc whisper", path, e)
        self._onset_src = []
        self._onsets = pipeline.original_onsets(clips, sources, words_by_file, self._onset_src,
                                                levels_by_file, audio.LEVEL_FRAME_SEC)
        texts = pipeline.original_texts(clips, sources, words_by_file)
        # Cau goc da xoa khoi timeline nhung con trong file giong goc -> giong moi co the doc THUA
        self._extras = pipeline.find_extra_speech(clips, sources, words_by_file, levels_by_file, texts,
                                                  self._embed().encode, audio.LEVEL_FRAME_SEC)
        self._report_extras()
        return texts

    def _report_extras(self):
        """Ghi log + bao tren giao dien cac doan co tieng trong file giong goc khong thuoc clip nao."""
        edge = 0
        for e in self._extras:
            log.info("DOAN HO giong goc [%s] %s %.2f-%.2fs tieng %.2fs sau cau %d truoc cau %d giong clip %.2f: '%s'",
                     e.kind, os.path.basename(e.path), e.start, e.end, e.speech, e.after + 1, e.before + 1,
                     e.reread_sim, e.text)
            where = f"{os.path.basename(e.path)} {e.start:.1f}–{e.end:.1f}s"
            if e.kind == "thua":
                self.log.emit(f"Giọng gốc có đoạn không thuộc clip nào (câu gốc đã xóa khỏi timeline?): {where} "
                              f"'{e.text[:80]}' → giọng mới đọc câu này sẽ thành đoạn riêng.")
                if e.reread_sim >= pipeline.EXTRA_REREAD_WARN:
                    self.log.emit(f"Cảnh báo: đoạn {where} giống 1 clip gốc ({e.reread_sim:.2f}) — có thể là đoạn "
                                  "đọc lại, nên nghe kiểm tra.")
            elif e.kind == "manh":
                self.log.emit(f"Cảnh báo: giọng gốc {where} có đoạn ngắn không thuộc clip nào ('{e.text[:60]}') "
                              "— có thể clip bị cắt lẹm, nên kiểm tra.")
            elif e.kind in ("dau file", "cuoi file"):
                edge += 1
        if edge:
            self.log.emit(f"Giọng gốc có {edge} đoạn có tiếng trước clip đầu / sau clip cuối của file (không dùng) "
                          "— bỏ qua, chi tiết trong file log.")

    def _embed(self):
        if self._embedder is None:
            from core.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def _set_progress(self, job, fraction):
        """Tien do 1 ngon ngu (0..1) -> tien do chung. Goi duoc tu nhieu luong."""
        with self._progress_lock:
            self._lang_progress[job.row] = fraction
            overall = sum(self._lang_progress.values()) / max(1, len(self._lang_progress))
        self.progress.emit(int(8 + 90 * overall))

    def _fail(self, job, error):
        self.language_status.emit(job.row, "Lỗi", "error")
        self.log.emit(f"[{job.name}] LỖI: {error}")
        self._set_progress(job, 1.0)

    def _process_language(self, job, clips, durations, timeline_end, source_texts, share):
        """share: phan tien do cua ngon ngu nay danh cho buoc nay (phan con lai la xuat mp4)."""
        cfg = self.config
        if len(job.parts) > 1:
            self.language_status.emit(job.row, f"Ghép {len(job.parts)} phần…", "run")
            job.audio_path, reused = parts.merge_parts(job.parts, job.name, cfg.project_dir)
            self.log.emit(f"[{job.name}] {'Dùng lại' if reused else 'Đã ghép'} {len(job.parts)} phần → "
                          f"{os.path.basename(job.audio_path)}")
            if os.path.dirname(os.path.abspath(job.audio_path)) != os.path.dirname(os.path.abspath(job.parts[0])):
                self.log.emit(f"[{job.name}] Thư mục các phần quá sâu → file ghép đặt trong project: {job.audio_path}")
        else:
            job.audio_path = job.parts[0]
        self.language_status.emit(job.row, "Nhận dạng giọng nói…", "run")

        def on_progress(p):
            self.language_status.emit(job.row, f"Nhận dạng giọng nói {p}%", "run")
            self._set_progress(job, share * 0.8 * p / 100)

        cached = os.path.isfile(speech._cache_path(job.audio_path, cfg.whisper_model))   # chi de ghi log
        words, lang, dur = speech.transcribe_words(job.audio_path, cfg.whisper_model, cfg.device, on_progress,
                                                   lambda m: self.log.emit(f"[{job.name}] {m}"),
                                                   self._stopped, self._models)
        if "device" in self._models:
            self.log.emit(f"[{job.name}] Nhận dạng xong ({lang}, {len(words)} từ, "
                          f"{'GPU' if self._models['device'] == 'cuda' else 'CPU'})")
        if self._stop:
            raise InterruptedError

        self.language_status.emit(job.row, "Đang tách câu…", "run")
        silences = audio.detect_silences(job.audio_path)
        loose = False
        plan = self._reading_plan(clips, source_texts)
        durations_ext = [clips[x].target_duration_us / US if kind == "c" else x.end - x.start for kind, x in plan]
        texts_in_order = [source_texts[x] if kind == "c" else x.text for kind, x in plan] if source_texts else None
        split_args = dict(labels=[x + 1 if kind == "c" else f"thừa sau {x.after + 1}" for kind, x in plan],
                          pair_gaps=self._plan_gaps(clips, plan),
                          ghosts={k for k, (kind, _) in enumerate(plan) if kind == "g"},
                          optional={k for k, (kind, x) in enumerate(plan) if kind == "c" and x in self._optional})
        embedder = self._embed() if source_texts else None
        try:
            result = segmenter.split_sentences(words, durations_ext, silences, dur, texts_in_order, embedder,
                                               **split_args)
        except segmenter.TooFewPausesError as e:
            # Doc qua lien mach (vd tieng A Rap) -> do lai khoang lang nhay hon roi thu lai
            log.info("%s: %s -> do lai khoang lang nhay hon", job.name, e)
            self.log.emit(f"[{job.name}] Audio đọc liền mạch, ít chỗ ngắt → đo lại khoảng lặng nhạy hơn…")
            silences = audio.detect_silences(job.audio_path, noise_db=segmenter.LOOSE_NOISE_DB,
                                             min_len=segmenter.LOOSE_MIN_PAUSE)
            loose = True
            result = segmenter.split_sentences(words, durations_ext, silences, dur, texts_in_order, embedder,
                                               min_pause=segmenter.LOOSE_MIN_PAUSE, **split_args)
        self._set_progress(job, share)
        # Cau thu k (thu tu doc) thuoc ve clip self._order[k] -> xep lai theo timeline
        by_clip = [None] * len(clips)
        ghosts = []                     # [(ExtraSpeech, cau moi hoac None)]
        for (kind, x), sent in zip(plan, result.sentences):
            if kind == "c":
                by_clip[x] = sent
            else:
                ghosts.append((x, sent))
        # Cat o cho lang nhat giua 2 cau (giu tron chu dau/cuoi), keo dai 0.5s moi dau (phan lan sang
        # cau ben canh tat tieng), luc bat dau noi = luc giong goc bat dau noi (ca 2 do bang nang luong).
        span_dbg, place_dbg = {}, []
        spans = pipeline.voice_spans(result.sentences, audio.levels_db(job.audio_path), audio.LEVEL_FRAME_SEC, dur,
                                     span_dbg)
        onsets = getattr(self, "_onsets", None)
        placements, report = pipeline.compute_placements(by_clip, clips, timeline_end, spans, onsets, place_dbg)
        placements, place_dbg, job.extra_placements, extra_notes = self._place_extras(
            job, ghosts, clips, by_clip, spans, onsets, timeline_end, placements, place_dbg)
        self.log.emit(f"[{job.name}] Mỗi câu kéo dài {pipeline.EXTEND_SEC:g}s hai đầu (phần lấn câu bên cạnh tắt tiếng); "
                      "lúc bắt đầu nói căn theo " + ("giọng gốc (đo theo âm lượng)." if onsets
                                                     else "mép câu gốc (không có file giọng gốc)."))
        for first, second in result.merged:
            self.log.emit(f"[{job.name}] Câu {first} + {second} được đọc liền thành 1 câu → đặt ở vị trí câu "
                          f"{first}, câu {second} không có tiếng riêng.")
        self.log.emit(f"[{job.name}] Tách {sum(isinstance(d['clip'], int) for d in place_dbg)}/{len(clips)} câu"
                      + (f" ({len(result.merged)} cặp đọc liền)" if result.merged else "")
                      + (f" · voice THIẾU {len(result.missing)} câu" if result.missing else "")
                      + f" · độ khớp {result.correlation:.2f}"
                      f" · {report.pushed_later} câu đẩy ra sau · {report.pulled_earlier} câu kéo sớm lên"
                      f" · 0 chồng tiếng · không cắt câu nào")
        warnings = list(result.warnings)
        # Giong moi THIEU cau (chot 2026-10-21): de trong cho clip do, cau sau van bam clip cua no
        for label in reversed(result.missing):
            c = clips[label - 1]
            text = " ".join(str((source_texts or [""] * len(clips))[label - 1] or "").split())
            warnings.insert(0, f"Voice {job.name} THIẾU câu {label}: '{text[:80]}{'…' if len(text) > 80 else ''}' "
                               f"(timeline {c.target_start_us / US:.1f}–{(c.target_start_us + c.target_duration_us) / US:.1f}s)"
                               " → để trống chỗ câu này.")
        warnings[0:0] = extra_notes
        tail = self._early_tail(placements, clips)
        if tail:
            warnings.insert(0, tail)
        for i in sorted(self._optional):
            if by_clip[i] is None:
                warnings.insert(0, f"Voice {job.name} không có câu {i + 1} (file phụ "
                                   f"'{os.path.basename(clips[i].source_path)}' – có thể là lời kêu gọi riêng) → để trống.")
        for label, best in result.debug.get("unsure_missing", []):
            warnings.append(f"Câu {label} có thể bị thiếu trong voice (giống nhất {best:.2f}) — vẫn xếp như cũ, nên nghe kiểm tra.")
        if report.far:
            detail = ", ".join(f"câu {i} ({d:+.1f}s)" for i, d in report.far[:8])
            more = f" và {len(report.far) - 8} câu khác" if len(report.far) > 8 else ""
            warnings.append(f"{len(report.far)} câu lệch mốc gốc hơn 5 giây: {detail}{more} — nên nghe kiểm tra.")
        for w in warnings:
            self.log.emit(f"[{job.name}] Cảnh báo: {w}")
        result.warnings = warnings
        try:
            self._log_details(job, clips, by_clip, result, source_texts, spans, span_dbg, placements, place_dbg,
                              dict(lang=lang, dur=dur, words=len(words), cached=cached,
                                   silences=len(silences), loose=loose))
            self.log.emit(f"[{job.name}] Đã ghi chi tiết từng câu vào file log.")
        except Exception:
            log.exception("Loi khi ghi log chi tiet %s", job.name)     # ghi log hong khong lam hong ket qua
        if result.warnings:
            self._warned_rows.add(job.row)
        status_kind = "warn" if result.warnings else "ok"
        self.language_status.emit(job.row, "Đã xếp câu" + (" · có cảnh báo" if result.warnings else ""),
                                  status_kind)
        return placements, int(round(dur * US))

    def _log_details(self, job, clips, by_clip, result, source_texts, spans, span_dbg, placements, place_dbg, info):
        """Ghi vao FILE log (khong hien tren giao dien) chi tiet tung cau: clip goc, noi dung, cho cat
        trong file voice, vi tri dat tren timeline, do lech luc bat dau noi, canh bao -> chan doan lech
        cau / cat sai chi bang file log. Moi cau 1 dong, cac phan cach nhau bang " | ", thoi gian = giay."""
        name = job.name
        # Nhan dang lay tu cache va chua nap model lan nao -> chi biet thiet bi theo cau hinh
        device = self._models.get("device", "") or self.config.device
        dbg = result.debug
        log.info("[%s] CHI TIET | model=%s thiet_bi=%s compute=%s nhan_dang=%s | ngon_ngu=%s audio=%.3fs tu=%d"
                 " | khoang_lang_ffmpeg=%d che_do_nhay=%s cho_ngat=%d (lang ffmpeg %d, dau cau whisper %d)"
                 " tieng=%.3f-%.3f | floor=%.1fdB | ti_le_toc_do=%.3f do_khop=%.3f so_nghia=%s",
                 name, self.config.whisper_model, device or "?", speech.COMPUTE_TYPE.get(device, "?"),
                 "cache" if info["cached"] else "moi", info["lang"], info["dur"], info["words"],
                 info["silences"], "co" if info["loose"] else "khong", dbg.get("pauses", 0),
                 dbg.get("silence_pauses", 0), dbg.get("pauses", 0) - dbg.get("silence_pauses", 0),
                 dbg.get("speech_start", 0.0), dbg.get("speech_end", 0.0),
                 span_dbg.get("floor", float("nan")), result.speed_ratio, result.correlation, result.used_meaning)
        src = self._text_src or [""] * len(clips)
        counts = {}
        for s in src:
            counts[s or "khong co"] = counts.get(s or "khong co", 0) + 1
        filled = [i + 1 for i, s in enumerate(src) if "(bu)" in s]
        log.info("[%s] CHI TIET | noi dung cau goc: %s%s | onset: %s", name,
                 ", ".join(f"{k} {v}" for k, v in counts.items()),
                 f" (cau bu: {', '.join(map(str, filled))})" if filled else "",
                 "giong goc (nang luong, whisper du phong)" if self._onsets else "mep clip (khong co file giong goc)")
        # Canh bao theo tung cau (so thu tu tren timeline): so cau o DAU canh bao ("Câu x" / "Câu x + y"), canh bao
        # lech > 5s liet ke "câu x (+d s)"; canh bao khong ghi so cau -> ghi rieng
        per_sentence = {}
        for w in result.warnings:
            head = re.match(r"Câu (\d+)(?: \+ (\d+))?", w) or re.match(r"Voice .* THIẾU câu (\d+)()", w)
            nums = {int(x) for x in head.groups() if x} if head else set()
            if "lệch mốc gốc" in w:
                nums |= {int(x) for x in re.findall(r"câu (\d+) \(", w)}
            if not nums:
                log.info("[%s] CHI TIET | canh bao chung: %s", name, w)
            for k in nums:
                per_sentence.setdefault(k, []).append(w)
        for label, best, near, decision in dbg.get("missing_candidates", []):
            log.info("[%s] CHI TIET | ung vien THIEU cau %s: giong nhat cuc bo %.3f, cau ben canh %.3f (chenh %.3f)"
                     " -> %s", name, label, best, near, near - best, decision)
        for m in dbg.get("refine", []):
            log.info("[%s] CHI TIET | doi ranh gioi cau %s|%s (cau lech do dai): '%s' | '%s' -> '%s' | '%s' | cum '%s':"
                     " nghia duoi cau truoc %.3f / dau cau sau %.3f (diem %+.2f) | ngung %.2fs -> %.2fs | cat %.3f -> %.3fs"
                     " | do dai %.2f/%.2fs -> %.2f/%.2fs (mong %.2f/%.2fs)", name, m["labels"][0], m["labels"][1],
                     m["old"][0], m["old"][1], m["new"][0], m["new"][1], m["chunk"], m["sim_tail"], m["sim_head"], m["gain"],
                     m["pause_old"], m["pause_new"], m["cut_old"], m["cut_new"], m["len_old"][0], m["len_old"][1],
                     m["len_new"][0], m["len_new"][1], m["expected"][0], m["expected"][1])
        merged_into = {second: first for first, second in result.merged}
        by_place = {d["clip"]: (p, d) for p, d in zip(placements, place_dbg)}

        def sec(us):
            return us / US

        def short(text, n=60):
            text = " ".join(str(text or "").replace("|", "/").split())
            return text if len(text) <= n else text[:n - 1] + "…"

        for i, c in enumerate(clips):
            label = i + 1
            row = [f"[{name}] cau {label}"]
            onset = (f"{self._onsets[i]:.3f} ({self._onset_src[i]})" if self._onsets and self._onset_src
                     else "0 (mep clip, khong co giong goc)")
            onset_kind = ("mep_clip" if not (self._onsets and self._onset_src) else
                          "nang_luong" if self._onset_src[i].startswith("nang luong") else
                          "whisper" if self._onset_src[i].startswith("du phong") else "mep_clip")
            row.append(f"goc tl={sec(c.target_start_us):.3f}-{sec(c.target_start_us + c.target_duration_us):.3f}"
                       f" src={sec(c.source_start_us):.3f}-{sec(c.source_start_us + c.source_duration_us):.3f}"
                       f" file={os.path.basename(c.source_path.replace(chr(92), '/'))} onset={onset}")
            row.append(f'noi_dung_goc="{short(source_texts[i] if source_texts else "")}" nguon={src[i] or "khong co"}')
            s = by_clip[i]
            if s is None and label in result.missing:
                best = dict(dbg.get("missing_sim", [])).get(label)
                row.append("THIEU: giong moi khong doc cau nay (giong nhat cuc bo "
                           f"{best:.3f}) -> de trong" if best is not None else "THIEU -> de trong")
            elif s is None and i in self._optional:
                row.append("KHONG CO trong giong moi (file phu chen ngan, cau tuy chon) -> de trong")
            elif s is None:
                row.append(f"GOP vao cau {merged_into.get(label, '?')} (khong co doan rieng)")
            else:
                d = s.debug
                row.append(f'moi="{short(s.text)}" giong_nghia={s.similarity:.3f}')
                row.append(f"cat_cu={s.start:.3f}-{s.end:.3f} dau={d.get('start_kind', '?')}"
                           f"({d.get('start_pause', 0.0):.2f}s) cuoi={d.get('end_kind', '?')}"
                           f"({d.get('end_pause', 0.0):.2f}s)")
                v, sd = spans.get(id(s)), span_dbg.get(id(s), {})
                if v is not None:
                    row.append(f"nghe_duoc={v.audible_start:.3f}-{v.audible_end:.3f}"
                               f" cho_cat_dau={sd.get('cut_start', '?')} {sd.get('db_start', float('nan')):+.1f}dB(so floor)"
                               f" cho_cat_cuoi={sd.get('cut_end', '?')} {sd.get('db_end', float('nan')):+.1f}dB(so floor)"
                               f" nguong={sd.get('threshold', float('nan')):+.1f}dB(so floor)"
                               f" noi={v.speech_start:.3f}({'nang_luong' if sd.get('onset_by_energy') else '=cat_cu'})"
                               f"-{v.speech_end:.3f}({'nang_luong' if sd.get('offset_by_energy') else '=cat_cu'})"
                               f" keo_dai={v.ext_start:.3f}-{v.ext_end:.3f}"
                               + "".join(f" keo_dai_{side}_rut={full:.3f}->{now:.3f}(am cau ben canh)"
                                         for side, full, now in zip(("dau", "cuoi"), sd.get("ext_full", ()),
                                                                    (v.ext_start, v.ext_end))
                                         if abs(full - now) > 1e-6))
                if i in by_place:
                    (target, source, duration, mute_head, mute_tail), pd = by_place[i]
                    row.append(f"dat tl={sec(target):.3f} src={sec(source):.3f} dur={sec(duration):.3f}"
                               f" tat_dau={sec(mute_head):.3f} tat_cuoi={sec(mute_tail):.3f}"
                               f" noi_tl={sec(pd['speech_us']):.3f} goc_noi_tl={sec(pd['onset_us']):.3f}"
                               f" delta_vs_{onset_kind}={sec(pd['speech_us'] - pd['onset_us']):+.3f}"
                               f" TH1={sec(pd['th1_us']):+.3f} TH2={sec(pd['th2_us']):+.3f}"
                               f" rut_nghe_duoc={sec(pd['aud_head_cut_us']):.3f}/{sec(pd['aud_tail_cut_us']):.3f}"
                               f" rut_keo_dai={sec(pd['head_cut_us']):.3f}/{sec(pd['tail_cut_us']):.3f}"
                               + (f" bo_lang_dau={sec(pd['trim_start_us']):.3f}" if pd["trim_start_us"] else ""))
            for w in per_sentence.get(label, []):
                row.append(f"CANH BAO: {short(w, 200)}")
            log.info(" | ".join(row))

    def _export(self, job, placements, video_duration):
        """Chay o luong phu (song song voi nhan dang ngon ngu tiep theo)."""
        cfg = self.config
        out_path = (cfg.output_paths or {}).get(job.row) or \
            os.path.join(cfg.output_dir, output_name(cfg.video_path, job.name))
        settings = cfg.render_settings or render.RECOMMENDED
        if path_too_long(out_path, len(".part.mp4")):      # file tam khi render: <ten>.mp4.part.mp4
            raise audio.AudioError(long_path_message(out_path))
        self.language_status.emit(job.row, "Đang dựng audio…", "run")
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            audio.build_dub_wav(job.audio_path, [tuple(v / US for v in p) for p in placements],
                                video_duration, wav)

            def on_progress(p, speed, eta):
                self.language_status.emit(job.row, f"Đang render {p}%", "run")
                self.export_progress.emit(job.row, p, speed, float(eta))
                self._set_progress(job, 0.6 + 0.4 * p / 100)

            logo_png, geo_for = self._logo_for(job)
            if logo_png:
                self._warn_old_nvidia_driver()
            how = render.render(cfg.video_path, wav, out_path, settings, logo_png, geo_for, on_progress,
                                self._stopped, on_log=lambda m: self.log.emit(f"[{job.name}] {m}"))
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass
        warned = job.row in self._warned_rows
        self.language_status.emit(job.row, "Xong · có cảnh báo" if warned else "Xong · đã xuất mp4",
                                  "warn" if warned else "ok")
        self.log.emit(f"[{job.name}] Xuất {out_path} ({how})")

    def _reading_plan(self, clips, source_texts) -> list:
        """Thu tu doc co chen cau MA: [("c", chi so clip) | ("g", ExtraSpeech)]. Cau ma = doan "thua" nam giua 2 clip
        lien nhau theo thu tu doc (can noi dung cau goc de so nghia)."""
        extras = [e for e in self._extras if e.kind == "thua"] if source_texts else []
        plan = []
        for k, i in enumerate(self._order):
            plan.append(("c", i))
            nxt = self._order[k + 1] if k + 1 < len(self._order) else None
            plan += [("g", e) for e in extras if e.after == i and e.before == nxt]
        return plan

    @staticmethod
    def _early_tail(placements, clips) -> str:
        """Luoi an toan: doan CUOI file voice dat cach clip cuoi tren timeline > TAIL_EARLY_SEC (vd bai 120: duoi cau
        keu goi bi tach sang clip chen giua video) -> canh bao do."""
        if not placements or not clips:
            return ""
        t, src, d, head, tail = max(placements, key=lambda p: p[1] + p[2])
        if t >= clips[-1].target_start_us - TAIL_EARLY_SEC * US:
            return ""
        return (f"Đuôi voice ({(src + head) / US:.1f}–{(src + d - tail) / US:.1f}s) đặt ở timeline "
                f"{(t + head) / US:.1f}s, trước câu cuối ({clips[-1].target_start_us / US:.1f}s) hơn "
                f"{TAIL_EARLY_SEC}s — kiểm tra thứ tự câu.")

    def _plan_gaps(self, clips, plan) -> list:
        """pair_gaps theo thu tu doc co cau ma (cap co cau ma: khong gop duoc, gia tri khong dung)."""
        gaps = []
        for (ka, a), (kb, b) in zip(plan, plan[1:]):
            if ka != "c" or kb != "c":
                gaps.append(0.0)
            elif b != a + 1:
                gaps.append(segmenter.MERGE_FAR_GAP)
            else:
                end_a = clips[a].target_start_us + clips[a].target_duration_us
                gaps.append(max(0.0, (clips[b].target_start_us - end_a) / US))
        return gaps

    def _place_extras(self, job, ghosts, clips, by_clip, spans, onsets, timeline_end, placements, place_dbg):
        """Cau THUA giong moi doc (cau ma da nhan doan) -> DOAN RIENG ngay sau cau truoc (cau cua clip e.after):
        du cho (tu het loi cau truoc toi luc noi cau sau >= loi cau thua + EXTRA_ROOM_SEC) -> track chinh, co tieng;
        khong du -> track phu "<ten> – câu thừa", tat tieng. Cac cau khac KHONG xe dich (kiem lai sau khi chen).
        Tra ve (placements track chinh, place_dbg, placements track phu, canh bao)."""
        notes, aux, main = [], [], []          # main: [(ExtraSpeech, cau, luc noi tren timeline)]
        base_speech = {d["clip"]: d["speech_us"] for d in place_dbg}
        for e, sent in ghosts:
            if sent is None:
                log.info("[%s] Cau thua %s %.2f-%.2fs: giong moi khong doc -> bo qua", job.name,
                         os.path.basename(e.path), e.start, e.end)
                continue
            probe, far = [], 10 ** 12          # do loi + le truoc chu dau cua cau thua (dat o xa: khong TH2 / cat)
            pipeline.compute_placements([sent], [pipeline.OriginalClip(far, 1, "", 0, 0)], 2 * far, spans, None, probe)
            core_len, lead = probe[0]["core_us"][1] - probe[0]["core_us"][0], probe[0]["lead_us"]
            prev = next((d for d in place_dbg if d["clip"] == e.after), None)
            if prev is not None:
                prev_end = prev["core_us"][1]
            else:                              # cau truoc gop / thieu: loi cuoi cung bat dau truoc het clip do
                end_ref = clips[e.after].target_start_us + clips[e.after].target_duration_us
                prev_end = max((d["core_us"][1] for d in place_dbg if d["core_us"][0] <= end_ref), default=0)
            next_speech = min((d["speech_us"] for d in place_dbg if d["core_us"][0] >= prev_end), default=timeline_end)
            speech_tl = prev_end + lead
            room = next_speech - prev_end
            fits = room >= core_len + int(EXTRA_ROOM_SEC * US) and prev_end + core_len <= timeline_end
            if fits:
                trial = main + [(e, sent, speech_tl)]
                new_pl, new_dbg = self._placements_with(trial, clips, by_clip, spans, onsets, timeline_end)
                moved = [i for i, v in base_speech.items()
                         if next((d["speech_us"] for d in new_dbg if d["clip"] == i), v) != v]
                if not moved:
                    main, placements, place_dbg = trial, new_pl, new_dbg
            if fits and main and main[-1][0] is e:
                where = f"đặt trên track chính sau câu {e.after + 1} (timeline {speech_tl / US:.1f}s)"
            else:
                one = []
                pl, _ = pipeline.compute_placements([sent], [pipeline.OriginalClip(speech_tl, core_len, "", 0, 0)],
                                                    max(timeline_end, speech_tl + core_len), spans, None, one)
                aux += pl
                where = (f"không đủ chỗ (trống {room / US:.1f}s, cần {(core_len + int(EXTRA_ROOM_SEC * US)) / US:.1f}s) → "
                         f"track phụ '{draft_patch.extra_track_name(job.name)}' (tắt tiếng), timeline {speech_tl / US:.1f}s"
                         " (lùi ra sau nếu chạm đoạn thừa khác)")
            reread = (f" Đoạn gốc giống 1 clip ({e.reread_sim:.2f}) — có thể là đoạn đọc lại, nên nghe."
                      if e.reread_sim >= pipeline.EXTRA_REREAD_WARN else "")
            notes.append(f"Voice {job.name} có câu THỪA (câu gốc đã xóa khỏi timeline): '{sent.text[:80]}' — gốc "
                         f"{os.path.basename(e.path)} {e.start:.1f}–{e.end:.1f}s → {where}.{reread}")
            log.info("[%s] CHI TIET | cau thua: goc %s %.2f-%.2fs '%s' | moi %.3f-%.3fs '%s' giong %.3f | loi %.3fs,"
                     " trong %.3fs -> %s", job.name, os.path.basename(e.path), e.start, e.end, e.text, sent.start,
                     sent.end, sent.text, sent.similarity, core_len / US, room / US, where)
        return placements, place_dbg, pipeline.stack_placements(aux), notes

    def _placements_with(self, main, clips, by_clip, spans, onsets, timeline_end):
        """Xep lai track chinh co them cac cau thua (clip ao dat dung luc noi tren timeline). place_dbg tra ve chi so clip
        THAT (cau thua: ("thua", thu tu))."""
        items = [(c.target_start_us, ("c", i)) for i, c in enumerate(clips)]
        items += [(tl, ("g", n)) for n, (_, _, tl) in enumerate(main)]
        items.sort(key=lambda x: x[0])
        clips2, sents2, onsets2, keys = [], [], [], []
        for tl, (kind, x) in items:
            if kind == "c":
                clips2.append(clips[x])
                sents2.append(by_clip[x])
                onsets2.append(onsets[x] if onsets else 0.0)
            else:
                clips2.append(pipeline.OriginalClip(tl, 1, "", 0, 0))
                sents2.append(main[x][1])
                onsets2.append(0.0)
            keys.append(x if kind == "c" else ("thua", x))
        dbg = []
        pl, _ = pipeline.compute_placements(sents2, clips2, timeline_end, spans, onsets2, dbg)
        for d in dbg:
            d["clip"] = keys[d["clip"]]
        return pl, dbg

    def _pair_gaps(self, clips):
        """Khoang cach tren timeline (giay) giua 2 cau goc lien nhau theo thu tu doc.
        Cau sau khong nam ngay sau tren timeline -> coi nhu rat xa (khong nen gop)."""
        gaps = []
        for i, j in zip(self._order, self._order[1:]):
            if j != i + 1:
                gaps.append(segmenter.MERGE_FAR_GAP)
            else:
                end_i = clips[i].target_start_us + clips[i].target_duration_us
                gaps.append(max(0.0, (clips[j].target_start_us - end_i) / US))
        return gaps

    def _warn_old_nvidia_driver(self):
        """Xuat video co logo ma card roi NVIDIA khong dung duoc do driver cu -> nhac 1 lan/luot chay."""
        with self._progress_lock:
            if self._driver_warned:
                return
            self._driver_warned = True
        from core.encoder import nvidia_status
        status = nvidia_status()
        if status.needs_update:
            self.log.emit("⚠ " + status.message())

    def _logo_for(self, job):
        """(png da xu ly, ham (w, h video DICH) -> (x, y, w, h) pixel) hoac ('', None)."""
        if not job.logo or not job.logo.get("path"):
            return "", None
        from PySide6.QtGui import QImage
        placement = logo.LogoPlacement.from_dict(job.logo.get("placement"))
        png = logo.prepared_logo_png(job.logo["path"], placement.remove_bg)
        img = QImage(png)
        # Vi tri luu theo TI LE khung hinh -> tinh theo do phan giai xuat (co the khac video goc)
        return png, lambda vw, vh: logo.overlay_geometry(placement, img.width(), img.height(), vw, vh)

    def _write_draft(self, draft, results) -> bool:
        """Tra ve True neu da ghi ket qua nhung CHUA ap dung (CapCut dang mo)."""
        cfg = self.config
        names = [job.name for job, _, _ in results]
        audible = cfg.default_language if cfg.default_language in names else names[0]
        tracks = [draft_patch.LanguageTrack(job.name, job.audio_path, dur_us, placements, job.name == audible,
                                            list(getattr(job, "extra_placements", None) or []))
                  for job, placements, dur_us in results]
        dubbed = draft_patch.build_dubbed_draft(draft, cfg.track_index, tracks)
        draft_patch.validate_dubbed(dubbed, draft["tracks"][cfg.track_index]["id"], tracks)
        draft_patch.write_dubbed(cfg.draft_files, dubbed)
        self.log.emit(f"Draft: thêm {len(tracks)} track ({', '.join(names)}), bật tiếng \"{audible}\", "
                      f"tắt tiếng track gốc.")
        if is_capcut_running():
            self.log.emit("CapCut đang mở → chưa áp dụng vào project. Tắt CapCut rồi bấm \"Áp dụng vào project\".")
            return True
        stamp = draft_patch.apply_to_project(cfg.draft_files)
        self.log.emit(f"Đã áp dụng vào project ({len(cfg.draft_files)} file draft, backup {stamp}). "
                      "Mở CapCut để kiểm tra.")
        if cfg.mode == MODE_DRAFT:
            for job, _, _ in results:
                warned = job.row in self._warned_rows
                self.language_status.emit(job.row, "Xong · có cảnh báo" if warned else "Xong · đã ghi draft",
                                          "warn" if warned else "ok")
        return False
