"""Luong chay nen cho nut "Chay": nhan dang -> tach cau -> xep timeline ->
xuat mp4 (tung ngon ngu) -> ghi draft gop tat ca ngon ngu -> ap dung vao project."""
import logging
import os
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal

from core import audio, draft_patch, pipeline, segmenter, speech
from core.capcut_project import ProjectError, US, is_capcut_running

log = logging.getLogger("run_worker")

MODE_DRAFT = "draft"
MODE_EXPORT = "export"
MODE_BOTH = "both"

MIN_SUBTITLE_COVER = 0.8   # phu de phu it hon muc nay -> nhan dang giong goc thay the


def output_name(video_path: str, language: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0] or "video"
    safe = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in language)
    return f"{stem}_{safe}.mp4"


@dataclass
class LanguageJob:
    row: int
    name: str
    audio_path: str


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


class RunWorker(QThread):
    log = Signal(str)
    language_status = Signal(int, str, str)   # row, text, kind (wait/run/ok/warn/error)
    progress = Signal(int)                     # 0-100 toan bo lan chay
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
        durations = [c.target_duration_us / US for c in clips]
        timeline_end = int(draft.get("duration", 0)) or (clips[-1].target_start_us + clips[-1].target_duration_us)
        self.log.emit(f"Bắt đầu: {len(cfg.languages)} ngôn ngữ · {len(clips)} câu gốc")

        video_duration = 0.0
        if uses_export:
            video_duration = audio.probe_duration(cfg.video_path)
            timeline_end = min(timeline_end, int(video_duration * US)) if timeline_end else int(video_duration * US)

        self.progress.emit(2)
        source_texts = self._original_texts(draft, clips)
        self.progress.emit(8)

        # Chay chong: GPU nhan dang ngon ngu k+1 trong luc CPU/o dia xuat mp4
        # cua ngon ngu k (luong phu) -> tan dung ca GPU lan CPU cung luc.
        self._lang_progress = {job.row: 0.0 for job in cfg.languages}
        export_pool = ThreadPoolExecutor(max_workers=2) if uses_export else None
        done = []      # [(job, placements, audio_duration_us, future_export_hoac_None)]
        err = 0
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
                    if not isinstance(e, (segmenter.SegmentError, audio.AudioError, ProjectError, OSError)):
                        log.exception("Loi khi xu ly %s", job.name)
                    err += 1
                    self._fail(job, e)
                    continue
                future = export_pool.submit(self._export, job, placements, video_duration) if uses_export else None
                done.append((job, placements, dur_us, future))

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

        pending_apply = False
        if uses_draft and results:
            pending_apply = self._write_draft(draft, results)
        self.progress.emit(100)
        return ok, err, pending_apply

    # ------------------------------------------------------------------
    def _original_texts(self, draft, clips):
        """Noi dung cau goc de so nghia: uu tien phu de, roi toi nhan dang giong goc."""
        texts, cover = pipeline.subtitle_texts(draft, clips)
        if cover >= MIN_SUBTITLE_COVER:
            self.log.emit(f"Nội dung câu gốc: lấy từ track phụ đề ({cover:.0%} câu có phụ đề).")
            return texts
        self.log.emit(f"Phụ đề chỉ phủ {cover:.0%} câu → nhận dạng giọng gốc để lấy nội dung câu.")
        search_roots = [os.path.dirname(self.config.project_dir), self.config.project_dir]
        sources = pipeline.resolve_original_sources(clips, search_roots)
        words_by_file = {}
        for path in sorted({p for p in sources.values() if p}):
            words, lang, _ = speech.transcribe_words(path, self.config.whisper_model, self.config.device,
                                                     lambda p: None, self.log.emit, self._stopped, self._models)
            words_by_file[path] = words
            self.log.emit(f"Đã nhận dạng giọng gốc {os.path.basename(path)} ({lang}).")
        if not words_by_file:
            self.log.emit("Không tìm thấy file giọng gốc → chỉ căn theo độ dài câu (kém chính xác hơn).")
            return None
        return pipeline.original_texts(clips, sources, words_by_file)

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
        self.language_status.emit(job.row, "Nhận dạng giọng nói…", "run")

        def on_progress(p):
            self.language_status.emit(job.row, f"Nhận dạng giọng nói {p}%", "run")
            self._set_progress(job, share * 0.8 * p / 100)

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
        result = segmenter.split_sentences(words, durations, silences, dur, source_texts,
                                           self._embed() if source_texts else None)
        self._set_progress(job, share)
        placements, truncated = pipeline.compute_placements(result.sentences, clips, timeline_end)
        self.log.emit(f"[{job.name}] Tách {len(result.sentences)}/{len(clips)} câu · độ khớp {result.correlation:.2f}"
                      f" · {truncated} câu cắt đuôi · 0 chồng tiếng")
        for w in result.warnings:
            self.log.emit(f"[{job.name}] Cảnh báo: {w}")
        if result.warnings:
            self._warned_rows.add(job.row)
        status_kind = "warn" if result.warnings else "ok"
        self.language_status.emit(job.row, "Đã xếp câu" + (" · có cảnh báo" if result.warnings else ""),
                                  status_kind)
        return placements, int(round(dur * US))

    def _export(self, job, placements, video_duration):
        """Chay o luong phu (song song voi nhan dang ngon ngu tiep theo)."""
        cfg = self.config
        out_path = os.path.join(cfg.output_dir, output_name(cfg.video_path, job.name))
        self.language_status.emit(job.row, "Đang dựng audio…", "run")
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            audio.build_dub_wav(job.audio_path, [(t / US, s / US, d / US) for t, s, d in placements],
                                video_duration, wav)

            def on_progress(p):
                self.language_status.emit(job.row, f"Đang export {p}%", "run")
                self._set_progress(job, 0.6 + 0.4 * p / 100)

            audio.mux_video(cfg.video_path, wav, out_path, on_progress, self._stopped)
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass
        warned = job.row in self._warned_rows
        self.language_status.emit(job.row, "Xong · có cảnh báo" if warned else "Xong · đã xuất mp4",
                                  "warn" if warned else "ok")
        self.log.emit(f"[{job.name}] Xuất {out_path}")

    def _write_draft(self, draft, results) -> bool:
        """Tra ve True neu da ghi ket qua nhung CHUA ap dung (CapCut dang mo)."""
        cfg = self.config
        names = [job.name for job, _, _ in results]
        audible = cfg.default_language if cfg.default_language in names else names[0]
        tracks = [draft_patch.LanguageTrack(job.name, job.audio_path, dur_us, placements, job.name == audible)
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
