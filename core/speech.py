"""Nhan dang giong noi bang faster-whisper de lay moc thoi gian TUNG TU cua
audio ngon ngu dich (thay cho file SRT).

- Model tai ve 1 lan vao thu muc models/ canh app, cac lan sau dung lai.
- Ket qua nhan dang duoc cache theo (file audio + model) trong cache/, chay
  lai cung 1 file (vd sau khi sua loi khac) khong phai nhan dang lai.
"""
import glob
import hashlib
import json
import logging
import os
import site
import sys
from dataclasses import dataclass, asdict
from typing import Callable, Optional

from core.paths import get_app_dir

log = logging.getLogger("speech")

CACHE_VERSION = 2
GPU_BATCH_SIZE = 16
CPU_BATCH_SIZE = 8


@dataclass
class Word:
    start: float   # giay
    end: float
    text: str


def get_models_dir() -> str:
    path = os.path.join(get_app_dir(), "models")
    os.makedirs(path, exist_ok=True)
    return path


def get_cache_dir() -> str:
    path = os.path.join(get_app_dir(), "cache")
    os.makedirs(path, exist_ok=True)
    return path


_dll_dirs_added = False


def add_nvidia_dll_dirs():
    """Thu vien CUDA (cuBLAS, cuDNN) cai qua pip nam o site-packages/nvidia/*/bin
    - Windows khong tu tim thay, phai them vao duong dan DLL truoc khi nap model."""
    global _dll_dirs_added
    if _dll_dirs_added or os.name != "nt":
        return
    _dll_dirs_added = True
    # Uu tien ban rut gon di kem app (bin/cuda: chi 3 DLL whisper that su nap,
    # ~740 MB thay vi ~2 GB cua goi pip day du), sau do toi goi pip nvidia-*.
    dirs = [os.path.join(get_app_dir(), "bin", "cuda")]
    for root in list(site.getsitepackages()) + [site.getusersitepackages()]:
        dirs += glob.glob(os.path.join(root, "nvidia", "*", "bin"))
    for bin_dir in dirs:
        if not os.path.isdir(bin_dir):
            continue
        try:
            os.add_dll_directory(bin_dir)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        except OSError:
            pass


def detect_device() -> str:
    """'cuda' neu co GPU NVIDIA, nguoc lai 'cpu'. (Thieu thu vien CUDA thi luc
    nap model se tu chuyen sang CPU - xem load_model.)"""
    try:
        import ctranslate2
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def is_model_downloaded(model: str) -> bool:
    # Moi model nam o repo khac nhau (Systran/..., mobiuslabsgmbh/... cho turbo)
    pattern = os.path.join(get_models_dir(), f"models--*--faster-whisper-{model}", "snapshots", "*", "model.bin")
    return bool(glob.glob(pattern))


def load_model(model: str, device: str, on_log: Callable[[str], None]):
    """Tra ve (WhisperModel, device_thuc_te)."""
    add_nvidia_dll_dirs()
    _patch_av_metadata_errors()
    from faster_whisper import WhisperModel

    # Model da co tren may -> khong ket noi HuggingFace (nhanh hon, chay duoc khi mat mang)
    local_only = is_model_downloaded(model)
    if not local_only:
        on_log(f"Đang tải model whisper '{model}' (chỉ tải lần đầu, có thể mất vài phút)…")
    common = dict(download_root=get_models_dir(), local_files_only=local_only)
    if device == "cuda":
        try:
            m = WhisperModel(model, device="cuda", compute_type="float16", **common)
            _warm_up(m)
            return m, "cuda"
        except Exception as e:
            log.warning("Khong chay duoc whisper tren GPU: %s", e)
            on_log(f"Không chạy được trên GPU ({_short(e)}) → chuyển sang CPU.")
    m = WhisperModel(model, device="cpu", compute_type="int8", **common)
    return m, "cpu"


def _patch_av_metadata_errors():
    """faster-whisper <= 1.2.1 goi av.open(..., metadata_errors="ignore") nhung PyAV 19
    da bo tham so nay -> loi "unexpected keyword argument 'metadata_errors'" tren may
    cai thu vien moi. Bo tham so do khi av >= 19 (giong cach faster-whisper master sua)."""
    import av
    if int(av.__version__.split(".")[0]) < 19 or getattr(av.open, "_nhanban_patched", False):
        return
    original_open = av.open

    def open_without_metadata_errors(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return original_open(*args, **kwargs)

    open_without_metadata_errors._nhanban_patched = True
    av.open = open_without_metadata_errors
    log.info("PyAV %s: bo tham so metadata_errors khi faster-whisper goi av.open", av.__version__)


def _warm_up(model):
    """Loi thieu cuBLAS/cuDNN chi lo ra khi chay that, khong phai luc nap model."""
    import numpy as np
    segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language="en")
    list(segments)


def _short(e) -> str:
    text = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return text[:120]


# Model nap 1 LAN cho ca phien app (dung chung moi lan bam Chay) - neu moi lan
# chay nap 1 ban moi thi ban cu van chiem bo nho GPU toi khi bi don -> het VRAM.
_loaded = {}


def _get_model(model_name: str, device: str, on_log, force_reload: bool = False) -> dict:
    global _loaded
    key = (model_name, device)
    if _loaded.get("key") == key and not force_reload:
        return _loaded
    if _loaded:
        _loaded.clear()
        import gc
        gc.collect()
    from faster_whisper import BatchedInferencePipeline
    model, real_device = load_model(model_name, device, on_log)
    _loaded = {
        "key": (model_name, real_device), "model": model, "device": real_device,
        "pipeline": BatchedInferencePipeline(model=model),
        "batch": GPU_BATCH_SIZE if real_device == "cuda" else CPU_BATCH_SIZE,
    }
    if real_device != device:
        _loaded["key"] = key   # GPU loi -> nho la da chuyen CPU, khong thu GPU lai moi lan
    return _loaded


def _run_batched(loaded, audio_path, on_progress, should_stop):
    """Chay theo LO: cat audio thanh cac doan co tieng (VAD) roi giai ma nhieu
    doan cung luc tren GPU -> nhanh ~2 lan so voi giai ma tuan tu va GPU duoc
    dung het (tuan tu chi dung 15-20%)."""
    segments, info = loaded["pipeline"].transcribe(
        audio_path, word_timestamps=True, batch_size=loaded["batch"], beam_size=5)
    words = []
    duration = float(info.duration or 0)
    for seg in segments:
        if should_stop():
            raise InterruptedError("Đã dừng")
        for w in seg.words or []:
            words.append(Word(float(w.start), float(w.end), w.word))
        if duration:
            on_progress(min(99, int(seg.end * 100 / duration)))
    return words, info


def _cache_path(audio_path: str, model: str) -> str:
    st = os.stat(audio_path)
    key = f"{os.path.abspath(audio_path)}|{st.st_size}|{st.st_mtime_ns}|{model}|{CACHE_VERSION}"
    return os.path.join(get_cache_dir(), hashlib.sha1(key.encode("utf-8")).hexdigest() + ".json")


def transcribe_words(audio_path: str, model_name: str, device: str,
                     on_progress: Callable[[int], None], on_log: Callable[[str], None],
                     should_stop: Callable[[], bool], model_holder: dict) -> tuple:
    """Nhan dang toan bo file audio. Tra ve (danh_sach_Word, ma_ngon_ngu, thoi_luong_giay).
    model_holder: dict dung chung giua cac ngon ngu de chi nap model 1 lan."""
    cache = _cache_path(audio_path, model_name)
    if os.path.isfile(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                data = json.load(f)
            on_log("Dùng lại kết quả nhận dạng đã lưu (cache).")
            on_progress(100)
            return [Word(**w) for w in data["words"]], data["language"], data["duration"]
        except (OSError, ValueError, KeyError, TypeError):
            pass

    loaded = _get_model(model_name, device, on_log)
    model_holder["device"] = loaded["device"]

    while True:
        try:
            words, info = _run_batched(loaded, audio_path, on_progress, should_stop)
            break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            # Card dung chung voi trinh duyet/app khac -> bo nho trong it hon du
            # kien. Giam so doan chay song song; het cach thi chuyen sang CPU.
            if loaded["batch"] > 1:
                loaded["batch"] //= 2
                on_log(f"GPU thiếu bộ nhớ → giảm xuống {loaded['batch']} đoạn song song và chạy lại.")
            elif loaded["device"] == "cuda":
                on_log("GPU vẫn thiếu bộ nhớ → chuyển sang CPU (chậm hơn). "
                       "Nên đóng bớt trình duyệt/app dùng card đồ hoạ.")
                loaded = _get_model(model_name, "cpu", on_log, force_reload=True)
                model_holder["device"] = "cpu"
            else:
                raise
    duration = float(info.duration or 0)
    on_progress(100)

    try:
        with open(cache, "w", encoding="utf-8") as f:
            json.dump({"language": info.language, "duration": duration,
                       "words": [asdict(w) for w in words]}, f, ensure_ascii=False)
    except OSError:
        pass
    return words, info.language, duration
