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


LEVEL_FRAME_SEC = 0.01


def levels_db(path: str):
    """Muc am luong (dB, RMS) tung khung 10ms cua ca file - de tim cho LANG NHAT giua 2 cau.
    Tra ve numpy array (khung i = giay i * 0.01)."""
    import numpy as np
    rate = 16000
    result = subprocess.run([find_ffmpeg(), "-hide_banner", "-v", "error", "-i", path, "-vn",
                             "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
                            capture_output=True, creationflags=NO_WINDOW)
    if result.returncode != 0 or not result.stdout:
        raise AudioError(f"Không giải mã được audio {os.path.basename(path)}:\n"
                         f"{result.stderr.decode('utf-8', 'replace')[-400:]}")
    x = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(rate * LEVEL_FRAME_SEC)
    n = len(x) // hop
    return 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12))


def decode_to_pcm(path: str, out_path: str):
    result = _run([find_ffmpeg(), "-hide_banner", "-y", "-i", path, "-vn",
                   "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE), "-f", "s16le", out_path])
    if result.returncode != 0:
        raise AudioError(f"Không giải mã được audio {os.path.basename(path)}:\n{result.stderr[-400:]}")


class PcmReader:
    """Doc doan mau (mono, float -1..1) tu file audio da giai ma ra PCM tam (khong nap ca file vao RAM)."""

    def __init__(self, path: str):
        import numpy as np
        fd, self._tmp = tempfile.mkstemp(suffix=".pcm")
        os.close(fd)
        decode_to_pcm(path, self._tmp)
        self._x = np.memmap(self._tmp, dtype=np.int16, mode="r").reshape(-1, CHANNELS)

    def mono(self, a: int, b: int):
        import numpy as np
        a, b = max(0, a), min(len(self._x), b)
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        return self._x[a:b].astype(np.float32).mean(axis=1) / 32768

    def close(self):
        self._x._mmap.close()
        del self._x
        try:
            os.remove(self._tmp)
        except OSError:
            pass


FEAT_WIN_SEC = 0.03


def pitch_hf_frames(path: str, n_frames: int, frame_sec: float = LEVEL_FRAME_SEC) -> tuple:
    """(R, H) moi khung 10ms (cua so 30ms tam o giua khung) cua ca file, tinh gop bang FFT - giong het
    thu_nghiem_test_v124/t027/loud2.py cua Tester (T-027 vong 2): R = tu tuong quan chuan hoa lon nhat 70-400 Hz sau loc
    thap 1 kHz (co cao do khi >= 0.6), H = ti le nang luong > 2.5 kHz tren > 80 Hz. Doc PCM tam theo tung khuc."""
    import numpy as np
    win = int(FEAT_WIN_SEC * SAMPLE_RATE)
    fr = np.fft.rfftfreq(win, 1 / SAMPLE_RATE)
    lp, hf, al = fr <= 1000, fr >= 2500, fr >= 80
    l0, l1 = int(SAMPLE_RATE / 400), int(SAMPLE_RATE / 70)
    hann = np.hanning(win).astype(np.float32)
    lag_norm = (1 - np.arange(l1 + 1) / win)[None, :]
    R = np.zeros(n_frames, np.float32)
    H = np.zeros(n_frames, np.float32)
    reader = PcmReader(path)
    try:
        total = len(reader._x)
        batch = 4000
        for s in range(0, n_frames, batch):
            idx = np.arange(s, min(n_frames, s + batch))
            c = ((idx + 0.5) * frame_sec * SAMPLE_RATE).astype(int) - win // 2
            c = np.clip(c, 0, max(0, total - win))
            lo, hi = int(c.min()), int(c.max()) + win
            chunk = reader.mono(lo, hi)
            if len(chunk) < hi - lo:
                chunk = np.pad(chunk, (0, hi - lo - len(chunk)))
            M = chunk[(c - lo)[:, None] + np.arange(win)[None, :]]
            M = M - M.mean(axis=1, keepdims=True)
            Sx = np.fft.rfft(M * hann, axis=1)
            P = np.abs(Sx) ** 2 + 1e-18
            H[idx] = P[:, hf].sum(1) / P[:, al].sum(1)
            y = np.fft.irfft(np.where(lp[None, :], Sx, 0), n=win, axis=1)
            Y = np.fft.rfft(y, n=2 * win, axis=1)
            ac = np.fft.irfft(np.abs(Y) ** 2, axis=1)[:, :l1 + 1]
            norm = ac[:, :1] * lag_norm
            R[idx] = (ac[:, l0:l1 + 1] / np.maximum(norm[:, l0:l1 + 1], 1e-12)).max(1)
    finally:
        reader.close()
    return R, H


def build_dub_wav(source_audio: str, placements: list, total_duration: float, out_wav: str, ramps: list = None,
                  ducks: list = None):
    """placements: [(vi_tri_tren_timeline, cat_tu, thoi_luong[, tat_dau, tat_cuoi])] (giay), da sap
    xep, khong chong nhau. tat_dau/tat_cuoi: phan keo dai o 2 dau doan -> am luong 0 (giong keyframe
    trong draft). ramps: [(vuot_dau, vuot_cuoi)] (giay) song song placements - giong LanguageTrack.ramps.
    ducks: [[(dau, cuoi)] vung ha tieng lay hoi dau / duoi cau (giay, vi tri trong doan)] - giong LanguageTrack.ducks (T-027).
    Ghi ra file WAV dai dung total_duration: cho nao co cau thi co tieng, con lai im lang."""
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
            for k, placement in enumerate(placements):
                target, cut_from, length = placement[:3]
                mute_head, mute_tail = (placement[3], placement[4]) if len(placement) >= 5 else (0.0, 0.0)
                t0 = int(round(target * SAMPLE_RATE))
                s0 = int(round(cut_from * SAMPLE_RATE))
                want = int(round(length * SAMPLE_RATE))
                head = int(round(mute_head * SAMPLE_RATE))
                tail = int(round(mute_tail * SAMPLE_RATE))
                if t0 < written:
                    # 2 cau NOI SAT nhau: lam tron micro-giay -> mau co the lech 1-2 mau.
                    # Lech nho (<10ms) thi bo phan du o dau cau sau; lech lon moi la loi that.
                    overlap = written - t0
                    if overlap > SAMPLE_RATE // 100:
                        raise AudioError("Các câu bị chồng lên nhau khi dựng audio (lỗi nội bộ).")
                    t0, s0, want, head = written, s0 + overlap, want - overlap, max(0, head - overlap)
                    duck_shift = overlap
                else:
                    duck_shift = 0
                if t0 >= total_frames:
                    break
                _write_silence(out, t0 - written)
                n = min(want, total_frames - t0, max(0, src_frames - s0))
                src.seek(s0 * BYTES_PER_FRAME)
                data = src.read(n * BYTES_PER_FRAME)
                ramp_head, ramp_tail = (r * 1000 for r in ramps[k]) if ramps else (None, None)
                duck = [tuple(int(round(x * SAMPLE_RATE)) - duck_shift for x in r) for r in ducks[k]] if ducks else None
                data = _mute_edges(data, head, max(0, tail - (want - n)), ramp_head_ms=ramp_head, ramp_tail_ms=ramp_tail,
                                   ducks=duck)
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


DUCK_GAIN = 0.1      # ha tieng lay hoi dau cau -20 dB (giong draft_patch.DUCK_GAIN, T-027)
DUCK_RAMP_MS = 10


def _mute_edges(data: bytes, head_frames: int, tail_frames: int, ramp_ms: int = 10,
                ramp_head_ms: float = None, ramp_tail_ms: float = None, ducks=None) -> bytes:
    """Tat tieng (0) phan keo dai dau/cuoi doan - giong keyframe am luong ghi vao draft CapCut
    (0 -> vuot 10ms -> 100% ... 100% -> vuot 10ms -> 0). ramp_head_ms / ramp_tail_ms > ramp_ms: mep nghe duoc giua
    tieng to -> vuot dai hon, ke ca khi khong co phan keo dai (giong mute_keyframes, T-001). ducks [(dau, cuoi)] (mau):
    vung am luong DUCK_GAIN, vuot DUCK_RAMP_MS trong vung; cham dau / cuoi phan nghe duoc thi noi lien phan tat tieng
    (ha tieng lay hoi dau / duoi cau, giong draft - T-027). Am luong = min(duong tat tieng, duong ha hoi)."""
    ramp_head_ms = ramp_ms if ramp_head_ms is None else ramp_head_ms
    ramp_tail_ms = ramp_ms if ramp_tail_ms is None else ramp_tail_ms
    long_head, long_tail = ramp_head_ms > ramp_ms, ramp_tail_ms > ramp_ms
    if head_frames <= 0 and tail_frames <= 0 and not long_head and not long_tail and not ducks:
        return data
    import numpy as np
    x = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS).astype(np.float32)
    n = len(x)
    env = np.ones(n, dtype=np.float32)
    # vung nghe duoc ngan hon 2 lan vuot -> vuot ngan lai (giong mute_keyframes trong draft)
    half = max(0, (n - max(0, head_frames) - max(0, tail_frames)) // 2)
    ramp_head = min(int(SAMPLE_RATE * ramp_head_ms / 1000), half)
    ramp_tail = min(int(SAMPLE_RATE * ramp_tail_ms / 1000), half)
    if head_frames > 0 or long_head:
        h = min(n, max(0, head_frames))
        env[:h] = 0
        r = env[h:h + ramp_head]
        env[h:h + ramp_head] = np.linspace(0, 1, len(r), endpoint=False)
    if tail_frames > 0 or long_tail:
        t = max(0, n - max(0, tail_frames))
        env[t:] = 0
        lo = max(0, t - ramp_tail)
        env[lo:t] = np.minimum(env[lo:t], np.linspace(1, 0, t - lo, endpoint=False))
    if ducks:
        dr = int(SAMPLE_RATE * DUCK_RAMP_MS / 1000)
        aud0, aud1 = max(0, head_frames), n - max(0, tail_frames)
        duck = np.ones(n, dtype=np.float32)
        for ds, de in sorted(ducks):
            ds, de = max(int(ds), aud0), min(int(de), aud1)
            if de - ds < 2 * dr:
                continue
            a = 0 if ds <= aud0 else ds
            b = n if de >= aud1 else de
            duck[a:b] = DUCK_GAIN
            if ds > aud0:
                duck[ds:ds + dr] = np.linspace(1, DUCK_GAIN, dr, endpoint=False)
            if de < aud1:
                duck[de - dr:de] = np.linspace(DUCK_GAIN, 1, dr, endpoint=False)
        env = np.minimum(env, duck)
    return (x * env[:, None]).round().astype(np.int16).tobytes()


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


def probe_video_size(path: str) -> tuple:
    result = _run([find_ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                   "stream=width,height", "-of", "csv=p=0:s=x", path])
    try:
        w, h = result.stdout.strip().split("x")[:2]
        return int(w), int(h)
    except ValueError:
        raise AudioError(f"Không đọc được kích thước video: {os.path.basename(path)}")


def mux_video(video_path: str, audio_wav: str, out_path: str,
              on_progress: Callable[[int], None], should_stop: Callable[[], bool],
              logo_png: str = "", logo_geometry: tuple = None, on_log: Callable[[str], None] = None):
    """Thay toan bo tieng bang audio long tieng.
    - Khong co logo: COPY hinh (khong encode lai, giu nguyen do phan giai) - vai giay.
    - Co logo: ve logo len hinh (suot video, ro 100%) -> phai encode lai; thu lan
      luot bo nen nhanh nhat (NVENC -> QSV -> AMF -> CPU), bo nao loi thi thu bo sau."""
    if not logo_png:
        _mux_once(video_path, audio_wav, out_path, on_progress, should_stop, ["-c:v", "copy"], [])
        return
    from core.encoder import usable_encoders
    x, y, w, h = logo_geometry
    graph = f"[2:v]scale={w}:{h}[lg];[0:v][lg]overlay={x}:{y}:format=auto,format=yuv420p[v]"
    last_error = None
    for name, label, params in usable_encoders():
        if on_log:
            on_log(f"Chèn logo + nén hình bằng {label}…")
        try:
            _mux_once(video_path, audio_wav, out_path, on_progress, should_stop,
                      ["-c:v", name] + params, ["-i", logo_png, "-filter_complex", graph], video_map="[v]")
            return
        except AudioError as e:
            last_error = e
            if on_log:
                on_log(f"{label} lỗi, thử bộ nén khác…")
    raise last_error or AudioError("Không nén được video có logo.")


def _mux_once(video_path, audio_wav, out_path, on_progress, should_stop, video_args, extra_inputs,
              video_map="0:v:0"):
    duration = probe_duration(video_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_out = out_path + ".part.mp4"
    args = [find_ffmpeg(), "-hide_banner", "-y", "-i", video_path, "-i", audio_wav] + extra_inputs + [
            "-map", video_map, "-map", "1:a:0"] + video_args + aac_encoder_args() + [
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
