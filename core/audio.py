"""Cac thao tac audio/video bang ffmpeg: do khoang lang, do thoi luong, dung
track long tieng dai bang video, ghep vao video (khong encode lai hinh)."""
import os
import re
import subprocess
import tempfile
from typing import Callable

from core.paths import find_ffmpeg, find_ffprobe

SAMPLE_RATE = 48000
CHANNELS = 2
BYTES_PER_FRAME = CHANNELS * 2   # s16le
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class AudioError(Exception):
    pass


def _run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", creationflags=NO_WINDOW, **kw)


def probe_duration(path: str) -> float:
    result = _run([find_ffprobe(), "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=nw=1:nk=1", path])
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise AudioError(f"Không đọc được thời lượng file: {os.path.basename(path)}\n{result.stderr[-300:]}")


def detect_silences(path: str, noise_db: int = -35, min_len: float = 0.12) -> list:
    """Tra ve [(bat_dau, ket_thuc)] cac khoang lang (giay)."""
    result = _run([find_ffmpeg(), "-hide_banner", "-nostats", "-i", path,
                   "-af", f"silencedetect=noise={noise_db}dB:d={min_len}", "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", result.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", result.stderr)]
    return [(max(0.0, s), e) for s, e in zip(starts, ends)]


def decode_to_pcm(path: str, out_path: str):
    result = _run([find_ffmpeg(), "-hide_banner", "-y", "-i", path, "-vn",
                   "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE), "-f", "s16le", out_path])
    if result.returncode != 0:
        raise AudioError(f"Không giải mã được audio {os.path.basename(path)}:\n{result.stderr[-400:]}")


def build_dub_wav(source_audio: str, placements: list, total_duration: float, out_wav: str):
    """placements: [(vi_tri_tren_timeline, cat_tu, thoi_luong)] (giay), da sap
    xep, khong chong nhau. Ghi ra file WAV dai dung total_duration: cho nao co
    cau thi co tieng, con lai im lang."""
    import wave

    fd, pcm_path = tempfile.mkstemp(suffix=".pcm")
    os.close(fd)
    try:
        decode_to_pcm(source_audio, pcm_path)
        src_frames = os.path.getsize(pcm_path) // BYTES_PER_FRAME
        total_frames = int(round(total_duration * SAMPLE_RATE))
        with open(pcm_path, "rb") as src, wave.open(out_wav, "wb") as out:
            out.setnchannels(CHANNELS)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            written = 0
            for target, cut_from, length in placements:
                t0 = int(round(target * SAMPLE_RATE))
                if t0 < written:
                    raise AudioError("Các câu bị chồng lên nhau khi dựng audio (lỗi nội bộ).")
                if t0 >= total_frames:
                    break
                _write_silence(out, t0 - written)
                s0 = int(round(cut_from * SAMPLE_RATE))
                n = min(int(round(length * SAMPLE_RATE)), total_frames - t0, max(0, src_frames - s0))
                src.seek(s0 * BYTES_PER_FRAME)
                data = src.read(n * BYTES_PER_FRAME)
                out.writeframes(_fade(data))
                written = t0 + len(data) // BYTES_PER_FRAME
            _write_silence(out, total_frames - written)
    finally:
        try:
            os.remove(pcm_path)
        except OSError:
            pass


def _write_silence(out, frames):
    chunk = b"\x00" * (SAMPLE_RATE * BYTES_PER_FRAME)
    while frames > 0:
        n = min(frames, SAMPLE_RATE)
        out.writeframes(chunk[:n * BYTES_PER_FRAME])
        frames -= n


def _fade(data: bytes, ms: int = 8) -> bytes:
    """Fade rat ngan o 2 dau doan cat de khong nghe tieng 'tach' khi cat dut."""
    import array
    samples = array.array("h", data)
    n = min(len(samples) // 2, int(SAMPLE_RATE * ms / 1000) * CHANNELS)
    for i in range(n):
        k = (i // CHANNELS) / (n / CHANNELS)
        samples[i] = int(samples[i] * k)
        samples[-1 - i] = int(samples[-1 - i] * k)
    return samples.tobytes()


# Bo nen AAC theo thu tu uu tien. Bo mac dinh cua ffmpeg ("aac" twoloop) chi
# chay 1 luong va mat ~17s cho 20 phut audio - la phan cham nhat khi xuat.
_AAC_CHOICES = (
    ("aac_mf", ["-c:a", "aac_mf"]),                          # Windows Media Foundation ~3.5x nhanh hon
    ("aac fast", ["-c:a", "aac", "-aac_coder", "fast"]),     # ~3x nhanh hon, co san moi ban ffmpeg
)
_aac_args = None


def aac_encoder_args() -> list:
    """Chon bo nen AAC nhanh nhat THUC SU chay duoc tren may nay (thu nen 0.2s)."""
    global _aac_args
    if _aac_args is None:
        _aac_args = _AAC_CHOICES[-1][1]
        for _, args in _AAC_CHOICES:
            r = _run([find_ffmpeg(), "-hide_banner", "-v", "error", "-f", "lavfi",
                      "-i", "sine=f=440:d=0.2:sample_rate=48000", "-ac", "2"] + args + ["-b:a", "192k", "-f", "null", "-"])
            if r.returncode == 0:
                _aac_args = args
                break
    return list(_aac_args)


def mux_video(video_path: str, audio_wav: str, out_path: str,
              on_progress: Callable[[int], None], should_stop: Callable[[], bool]):
    """Giu nguyen hinh (copy, khong encode lai, giu nguyen do phan giai), thay
    toan bo tieng bang audio long tieng."""
    duration = probe_duration(video_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_out = out_path + ".part.mp4"
    args = [find_ffmpeg(), "-hide_banner", "-y", "-i", video_path, "-i", audio_wav,
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"] + aac_encoder_args() + [
            "-b:a", "192k", "-t", f"{duration:.3f}", "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats", tmp_out]
    err_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=err_file, text=True,
                            encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    try:
        for line in proc.stdout:
            if should_stop():
                proc.kill()
                raise InterruptedError("Đã dừng")
            if line.startswith("out_time_us=") and duration:
                try:
                    us = int(line.split("=", 1)[1])
                    on_progress(min(99, int(us / 1e6 * 100 / duration)))
                except ValueError:
                    pass
        proc.wait()
        if proc.returncode != 0:
            err_file.seek(0)
            raise AudioError(f"Xuất video lỗi:\n{err_file.read()[-500:]}")
        os.replace(tmp_out, out_path)
        on_progress(100)
    finally:
        if proc.poll() is None:
            proc.kill()
        err_file.close()
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass
