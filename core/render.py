"""Render mp4: video goc + audio long tieng (+ logo), theo cau hinh xuat.

Nhanh nhat co the (da do tren may that, video 4K H.265 60fps):
- Khong logo + giu nguyen thong so goc -> COPY hinh (khong nen lai): ~50x thoi gian thuc.
- Can nen lai -> TOAN BO tren card roi: giai ma NVDEC -> doi kich thuoc + chen logo
  tren GPU (scale_cuda/overlay_cuda) -> nen NVENC preset p1. 4K: ~3.2x thoi gian thuc
  (gap doi preset p4, chat luong SSIM 0.996 vs 0.997 - mat thuong khong phan biet).
- Loi (vd video goc codec NVDEC khong ho tro) -> tu lui ve giai ma + loc tren CPU,
  roi toi cac bo nen khac (QSV/AMF/CPU).
"""
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from typing import Callable, Optional

from core.audio import AudioError, NO_WINDOW, _run, aac_encoder_args, probe_duration
from core.encoder import usable_encoders
from core.paths import find_ffmpeg, find_ffprobe

SPEED_PRESETS = {                 # (NVENC, QSV, AMF, x264/x265)
    "fast": ("p1", "veryfast", "speed", "veryfast"),
    "balanced": ("p4", "medium", "balanced", "fast"),
    "quality": ("p7", "slower", "quality", "slow"),
}


@dataclass
class RenderSettings:
    """Cau hinh xuat, dung chung cho moi ngon ngu."""
    width: int = 0                 # 0 = giu nhu video goc
    height: int = 0
    codec: str = "h265"            # h265 / h264
    encoder: str = "auto"          # auto = nhanh nhat chay duoc (uu tien NVENC)
    bitrate_kbps: int = 60000
    fps: float = 60.0              # 0 = giu nhu video goc
    audio_kbps: int = 384
    speed: str = "fast"            # fast / balanced / quality
    # Khong logo + dung do phan giai/fps -> COPY hinh goc du codec goc khac (vd H.264):
    # nhanh gap nhieu lan, chat luong giu nguyen 100%, YouTube nhan ca H.264.
    copy_if_no_logo: bool = True

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        base = RenderSettings()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                if isinstance(getattr(base, k), bool):
                    v = v if isinstance(v, bool) else str(v).lower() in ("true", "1")   # QSettings luu "false"
                setattr(base, k, type(getattr(base, k))(v))
        return base


# Chuan YouTube cua doi (giong Tool Join Video): 4K 3840x2160, H.265, 60 Mbps VBR,
# 60fps, MP4, AAC 384 kbps 48 kHz. Video goc da 4K + khong logo -> copy hinh (nhanh);
# video goc nho hon -> phong len 4K.
RECOMMENDED = RenderSettings(width=3840, height=2160)


@dataclass
class VideoInfo:
    codec: str
    width: int
    height: int
    fps: float
    bitrate_kbps: int
    duration: float


def probe_video(path: str) -> VideoInfo:
    r = _run([find_ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
              "stream=codec_name,width,height,r_frame_rate,bit_rate:format=duration,bit_rate",
              "-of", "json", path])
    try:
        data = json.loads(r.stdout)
        s = data["streams"][0]
        num, _, den = s.get("r_frame_rate", "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        bitrate = int(s.get("bit_rate") or data.get("format", {}).get("bit_rate") or 0) // 1000
        return VideoInfo(s.get("codec_name", ""), int(s["width"]), int(s["height"]), fps, bitrate,
                         float(data.get("format", {}).get("duration") or 0))
    except (ValueError, KeyError, IndexError):
        raise AudioError(f"Không đọc được thông số video: {os.path.basename(path)}")


CODEC_NAMES = {"h265": "hevc", "h264": "h264"}


def target_size(settings: RenderSettings, info: VideoInfo) -> tuple:
    if not settings.width or not settings.height:
        return info.width, info.height
    return settings.width, settings.height


def needs_encode(settings: RenderSettings, info: VideoInfo, has_logo: bool) -> bool:
    """Copy hinh (nhanh nhat) khi khong logo va video goc da dung thong so dich."""
    if has_logo:
        return True
    if target_size(settings, info) != (info.width, info.height):
        return True
    if settings.fps and abs(settings.fps - info.fps) > 0.5:
        return True
    if CODEC_NAMES.get(settings.codec) != info.codec and not settings.copy_if_no_logo:
        return True
    return False


def encoder_chain(settings: RenderSettings) -> list:
    """[(ten, mo ta)] theo thu tu thu - bo nen nguoi dung chon (neu co) dung dau."""
    wanted = "hevc" if settings.codec == "h265" else "h264"
    chain = []
    for name, label, _ in usable_encoders(settings.codec):
        chain.append((name, label))
    if settings.encoder != "auto":
        chain.sort(key=lambda c: c[0] != settings.encoder)
    return chain or [("libx265" if wanted == "hevc" else "libx264", "CPU")]


def _encoder_args(name: str, settings: RenderSettings) -> list:
    b = settings.bitrate_kbps
    rate = ["-b:v", f"{b}k", "-maxrate", f"{int(b * 1.5)}k", "-bufsize", f"{b * 2}k"]
    nv, qsv, amf, cpu = SPEED_PRESETS.get(settings.speed, SPEED_PRESETS["fast"])
    if "nvenc" in name:
        # Bo nen NVENC la nut that (do: 92-98% tai). Tat B-frame: +10-15% toc do, o
        # bitrate cao (60 Mbps) chat luong khac biet khong dang ke.
        args = ["-c:v", name, "-preset", nv, "-rc", "vbr"] + rate + (["-bf", "0"] if settings.speed == "fast" else [])
    elif "qsv" in name:
        args = ["-c:v", name, "-preset", qsv] + rate
    elif "amf" in name:
        args = ["-c:v", name, "-quality", amf, "-rc", "vbr_peak"] + rate
    else:
        args = ["-c:v", name, "-preset", cpu] + rate
    if settings.codec == "h265":
        args += ["-tag:v", "hvc1"]         # H.265 trong mp4 phat duoc tren moi trinh phat/YouTube
    return args


def _audio_args(settings: RenderSettings) -> list:
    # aac_mf (nhanh) chi toi ~320 kbps -> muc cao hon dung aac chuan che do nhanh
    if settings.audio_kbps > 320:
        return ["-c:a", "aac", "-aac_coder", "fast", "-b:a", f"{settings.audio_kbps}k", "-ar", "48000"]
    return aac_encoder_args() + ["-b:a", f"{settings.audio_kbps}k", "-ar", "48000"]


def _video_plan(settings, info, encoder, logo_png, logo_geo, gpu):
    """(tham so dau vao truoc -i video, filter/map, tham so nen hinh)."""
    w, h = target_size(settings, info)
    fps_args = ["-r", f"{settings.fps:g}"] if settings.fps and abs(settings.fps - info.fps) > 0.5 else []
    scale = (w, h) != (info.width, info.height)
    logo_chain = None
    if logo_png and logo_geo:
        from core.logo import visible_part
        x, y, lw, lh = logo_geo
        vis = visible_part(x, y, lw, lh, w, h)
        if vis:
            # Logo lan ra ngoai mep (CapCut cho phep): cat phan thua, giu dung vi tri
            cx0, cy0, cw, ch, px, py = vis
            crop = f",crop={cw}:{ch}:{cx0}:{cy0}" if (cw, ch) != (lw, lh) else ""
            logo_chain = (f"scale={lw}:{lh}{crop}", px, py)
    if gpu:
        pre = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        chain = f"scale_cuda={w}:{h}:format=yuv420p" if scale else "scale_cuda=format=yuv420p"
        if logo_chain:
            f, px, py = logo_chain
            graph = (f"[0:v]{chain}[m];[2:v]{f},format=yuva420p,hwupload_cuda[l];"
                     f"[m][l]overlay_cuda={px}:{py}[v]")
        else:
            graph = f"[0:v]{chain}[v]"
    else:
        pre = []
        parts = [f"scale={w}:{h}:flags=lanczos"] if scale else []
        if logo_chain:
            f, px, py = logo_chain
            base = ",".join(parts) or "null"
            graph = f"[0:v]{base}[m];[2:v]{f}[l];[m][l]overlay={px}:{py},format=yuv420p[v]"
        else:
            graph = f"[0:v]{','.join(parts + ['format=yuv420p'])}[v]"
    venc = _encoder_args(encoder, settings) + fps_args
    if gpu and not scale:
        # overlay_cuda giu nguyen khung dem cua bo giai ma (lam tron chieu cao len boi so 32:
        # 2160 -> 2176, 1080 -> 1088) -> day khung co vet XANH LA. Ghi thong tin cat vao
        # luong video (khong nen lai, khong ton them thoi gian) de hien thi dung kich thuoc.
        pad = -h % 32
        if pad:
            venc += ["-bsf:v", f"{_metadata_bsf(settings.codec)}=crop_bottom={pad}"]
    return pre, ["-filter_complex", graph, "-map", "[v]"], venc


def _metadata_bsf(codec: str) -> str:
    return "hevc_metadata" if codec == "h265" else "h264_metadata"


def _fix_padded_size(out_path: str, settings: RenderSettings, want: tuple):
    """Kiem tra lai file ra: kich thuoc lech (con dong/cot dem) -> cat bang thong tin
    trong luong video (copy, khong nen lai)."""
    got = probe_video(out_path)
    dw, dh = got.width - want[0], got.height - want[1]
    if (dw, dh) == (0, 0):
        return
    if dw < 0 or dh < 0 or dw > 64 or dh > 64:
        raise AudioError(f"Video xuất ra sai kích thước ({got.width}×{got.height}, cần {want[0]}×{want[1]}).")
    tmp = out_path + ".fix.mp4"
    r = _run([find_ffmpeg(), "-hide_banner", "-v", "error", "-y", "-i", out_path, "-map", "0", "-c", "copy",
              "-bsf:v", f"{_metadata_bsf(settings.codec)}=crop_right={dw}:crop_bottom={dh}", tmp])
    if r.returncode != 0:
        raise AudioError(f"Không sửa được kích thước video:\n{r.stderr[-300:]}")
    os.replace(tmp, out_path)


def render(video: str, audio_wav: str, out_path: str, settings: RenderSettings,
           logo_png: str = "", logo_geo_for: Callable = None,
           on_progress: Callable[[int, str, float], None] = lambda p, s, e: None,
           should_stop: Callable[[], bool] = lambda: False,
           on_log: Callable[[str], None] = lambda m: None) -> str:
    """Tra ve mo ta cach da render (vd 'copy hình', 'NVENC (GPU)').
    logo_geo_for(w, h) -> (x, y, lw, lh) theo kich thuoc video DICH."""
    info = probe_video(video)
    duration = info.duration or probe_duration(video)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    audio = _audio_args(settings)
    copy = not needs_encode(settings, info, bool(logo_png))
    check_free_space(out_path, estimate_size_bytes(settings, info, copy))

    if copy:
        _run_ffmpeg(["-i", video, "-i", audio_wav, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"] + audio,
                    out_path, duration, on_progress, should_stop)
        return "copy hình (không nén lại)"

    w, h = target_size(settings, info)
    logo_geo = logo_geo_for(w, h) if logo_png and logo_geo_for else None
    last_error = None
    for encoder, label in encoder_chain(settings):
        # NVENC: thu duong TOAN BO GPU truoc, loi thi giai ma/loc tren CPU
        for gpu in ((True, False) if "nvenc" in encoder else (False,)):
            pre, vmap, venc = _video_plan(settings, info, encoder, logo_png, logo_geo, gpu)
            inputs = pre + ["-i", video, "-i", audio_wav] + (["-i", logo_png] if logo_png else [])
            try:
                _run_ffmpeg(inputs + vmap + ["-map", "1:a:0"] + venc + audio, out_path, duration,
                            on_progress, should_stop)
                _fix_padded_size(out_path, settings, (w, h))
                return f"{label}{' · toàn bộ trên GPU' if gpu else ''}"
            except DiskFullError:
                raise                      # o day -> doi bo nen cung vo ich
            except AudioError as e:
                last_error = e
                on_log(f"{label}{' (toàn bộ GPU)' if gpu else ''} lỗi → thử cách khác. Chi tiết: "
                       f"{str(e).strip().splitlines()[-1][:200]}")
    raise last_error or AudioError("Không render được video.")


class DiskFullError(AudioError):
    pass


DISK_FULL_MARKERS = ("No space left on device", "There is not enough space on the disk", "Disk full",
                     "Error writing trailer", "No space")


def estimate_size_bytes(settings: RenderSettings, info: VideoInfo, copy: bool) -> int:
    video_kbps = info.bitrate_kbps if copy else settings.bitrate_kbps * 1.3   # VBR co the vuot bitrate dat
    return int((video_kbps + settings.audio_kbps) * 1000 / 8 * info.duration * 1.05)


def check_free_space(out_path: str, need_bytes: int):
    """Khong du cho trong -> bao ngay (truoc day render chay 10 phut roi moi loi o day,
    tool lai hieu nham la loi bo nen, doi sang bo nen cham hon)."""
    import shutil
    folder = os.path.dirname(os.path.abspath(out_path))
    free = shutil.disk_usage(folder).free
    if free < need_bytes:
        drive = os.path.splitdrive(folder)[0] or folder
        raise DiskFullError(f"Ổ {drive} chỉ còn trống {free / 1e9:.1f} GB, video này cần khoảng "
                            f"{need_bytes / 1e9:.1f} GB. Hãy chọn thư mục lưu ở ổ khác (vd ổ có nhiều chỗ trống hơn).")


def _run_ffmpeg(args, out_path, duration, on_progress, should_stop):
    tmp_out = out_path + ".part.mp4"
    # KHONG dung -movflags +faststart: buoc do ghi lai TOAN BO file sau khi render
    # (file 4K 30 phut ~14 GB -> them vai phut o 99%); YouTube khong can.
    cmd = [find_ffmpeg(), "-hide_banner", "-y"] + args + [
        "-t", f"{duration:.3f}", "-progress", "pipe:1", "-nostats", tmp_out]
    err_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file, text=True,
                            encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    speed = ""
    try:
        for line in proc.stdout:
            if should_stop():
                proc.kill()
                raise InterruptedError("Đã dừng")
            key, _, value = line.strip().partition("=")
            if key == "speed":
                speed = value
            elif key == "out_time_us" and duration:
                try:
                    done = int(value) / 1e6
                except ValueError:
                    continue
                pct = min(99, int(done * 100 / duration))
                try:
                    x = float(speed.rstrip("x"))
                    eta = (duration - done) / x if x > 0 else 0
                except ValueError:
                    eta = 0
                on_progress(pct, speed, eta)
        proc.wait()
        if proc.returncode != 0:
            err_file.seek(0)
            detail = err_file.read()[-1500:]
            import logging
            logging.getLogger("render").error("ffmpeg loi (exit %s):\n%s", proc.returncode, detail)
            try:
                full = shutil.disk_usage(os.path.dirname(os.path.abspath(out_path))).free < 200e6
            except OSError:
                full = False
            if full or any(m.lower() in detail.lower() for m in DISK_FULL_MARKERS):
                raise DiskFullError("Ổ đĩa lưu video đã đầy nên không render tiếp được. "
                                    "Hãy chọn thư mục lưu ở ổ khác rồi xuất lại.")
            raise AudioError(f"Render lỗi:\n{detail[-600:]}")
        os.replace(tmp_out, out_path)
        on_progress(100, speed, 0)
    finally:
        if proc.poll() is None:
            proc.kill()
        err_file.close()
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass
