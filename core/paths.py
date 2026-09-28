"""Duong dan thu muc app va cac cong cu di kem (ffmpeg...)."""
import os
import shutil


def get_app_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_tool(name: str) -> str:
    """Uu tien ban di kem trong bin/, sau do tim trong PATH he thong."""
    bundled = os.path.join(get_app_dir(), "bin", f"{name}.exe")
    if os.path.isfile(bundled):
        return bundled
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"Khong tim thay {name}.exe. Hay dat {name}.exe vao thu muc 'bin/' "
        "cua ung dung, hoac cai FFmpeg va them vao PATH he thong."
    )


def find_ffmpeg() -> str:
    return _find_tool("ffmpeg")


def find_ffprobe() -> str:
    return _find_tool("ffprobe")
