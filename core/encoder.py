"""Chon bo nen video H.264 nhanh nhat THUC SU chay duoc tren may (chi can khi
phai encode lai hinh - vd chen logo). Uu tien card roi NVIDIA."""
import logging
import re
from dataclasses import dataclass

from core.audio import _run
from core.paths import find_ffmpeg

log = logging.getLogger("encoder")

# (ten encoder, mo ta, tham so chat luong cao - giu net nhu video goc) - theo codec,
# thu tu uu tien: card roi NVIDIA -> Intel -> AMD -> CPU
CANDIDATES = {
    "h264": (
        ("h264_nvenc", "GPU NVIDIA (NVENC)", ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0"]),
        ("h264_qsv", "GPU Intel (Quick Sync)", ["-preset", "medium", "-global_quality", "20"]),
        ("h264_amf", "GPU AMD (AMF)", ["-quality", "quality", "-rc", "cqp", "-qp_i", "19", "-qp_p", "21"]),
        ("libx264", "CPU (x264)", ["-preset", "veryfast", "-crf", "18"]),
    ),
    "h265": (
        ("hevc_nvenc", "GPU NVIDIA (NVENC)", []),
        ("hevc_qsv", "GPU Intel (Quick Sync)", []),
        ("hevc_amf", "GPU AMD (AMF)", []),
        ("libx265", "CPU (x265)", []),
    ),
}

_usable = {}


def _works(name: str) -> bool:
    """Nen thu 4 khung 256x256 - bat duoc loi driver qua cu / thieu phan cung,
    thu ma chi doc ten encoder thi khong biet (vd NVENC can driver >= 610)."""
    r = _run([find_ffmpeg(), "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
              "-frames:v", "4", "-c:v", name, "-f", "null", "-"])
    if r.returncode != 0:
        log.info("Encoder %s khong dung duoc: %s", name, r.stderr.strip()[-300:])
    return r.returncode == 0


def usable_encoders(codec: str = "h264") -> list:
    """[(ten, mo ta, tham so)] theo thu tu uu tien, chi gom encoder chay duoc. Do 1 lan/codec."""
    if codec not in _usable:
        cands = CANDIDATES[codec]
        found = [c for c in cands if _works(c[0])]
        if not any(c[0] == cands[-1][0] for c in found):
            found.append(cands[-1])          # CPU luon la lua chon cuoi
        _usable[codec] = found
    return list(_usable[codec])


NVIDIA_DRIVER_URL = "https://www.nvidia.com/en-us/drivers/results/"


@dataclass
class NvidiaStatus:
    has_nvidia: bool = False
    gpu_name: str = ""
    driver: str = ""             # driver dang cai, vd "591.86"
    required: str = ""           # driver toi thieu ffmpeg can cho NVENC, vd "610.00"
    nvenc_ok: bool = False

    @property
    def needs_update(self) -> bool:
        return self.has_nvidia and not self.nvenc_ok and bool(self.required)

    def message(self) -> str:
        return (f"Card rời {self.gpu_name or 'NVIDIA'} chưa dùng được để xuất video vì driver cũ "
                f"(đang dùng {self.driver or '?'}, cần từ {self.required} trở lên). Tool tạm dùng bộ nén "
                f"khác (chậm hơn). Cập nhật driver NVIDIA tại: {NVIDIA_DRIVER_URL}")


def nvidia_status() -> NvidiaStatus:
    """May co card NVIDIA khong, NVENC chay duoc khong, neu khong thi co phai do driver cu."""
    st = NvidiaStatus()
    try:
        r = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if r.returncode == 0 and r.stdout.strip():
            name, _, driver = r.stdout.strip().splitlines()[0].partition(",")
            st.has_nvidia, st.gpu_name, st.driver = True, name.strip(), driver.strip()
    except OSError:
        pass                                  # khong co nvidia-smi = khong co card/driver NVIDIA
    if not st.has_nvidia:
        return st
    r = _run([find_ffmpeg(), "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
              "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"])
    st.nvenc_ok = r.returncode == 0
    m = re.search(r"minimum required Nvidia driver for nvenc is ([\d.]+)", r.stderr)
    if m:
        st.required = m.group(1)
    elif not st.nvenc_ok and "nvenc API version" in r.stderr:
        st.required = "mới nhất"
    return st
