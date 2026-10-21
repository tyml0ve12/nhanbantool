"""Duong dan thu muc app va cac cong cu di kem (ffmpeg...)."""
import os
import shutil

# Windows (chua bat "long paths"): duong dan file toi da 259 ky tu - dai hon thi mo/ghi file loi
# "[Errno 2] No such file or directory" (bai 44: ten ngon ngu = ten thu muc dai, file gop 268 ky tu).
MAX_PATH = 260


def path_too_long(path: str, extra: int = 0) -> bool:
    """True neu duong dan (+ extra ky tu hau to cua file tam, vd ".part.mp4") vuot gioi han Windows."""
    return os.name == "nt" and len(os.path.abspath(path)) + extra >= MAX_PATH


def long_path_message(path: str) -> str:
    return (f"Đường dẫn quá dài ({len(os.path.abspath(path))} ký tự, Windows giới hạn {MAX_PATH - 1}):\n{path}\n"
            "Hãy đổi tên thư mục / tên ngôn ngữ / tên file ngắn hơn, hoặc chuyển project tới thư mục ngắn hơn.")


def is_network_path(path: str) -> bool:
    """Duong dan mang UNC (\\\\server\\share\\... hoac //server/...) lay tu draft (vd //xuat-video/...): kiem tra ton tai
    (isfile / exists) co the treo cho mang rat lau khi may chu khong con -> khong kiem, tim theo ten (T-023)."""
    p = str(path or "").replace("\\", "/")
    return p.startswith("//")


def short_local_copy(path: str) -> str:
    """File NGUON (chi doc) co duong dan qua dai (vd giong goc tren Google Drive nhieu cap, bai 121) -> chep ra
    <app>\\cache\\nguon\\<ma bam><duoi> (giu mtime) va tra ve ban chep; ffmpeg / whisper doc duoc. Ban chep
    duoc dung lai khi file goc khong doi (cung kich thuoc + mtime)."""
    if not path_too_long(path):
        return path
    import hashlib
    src = "\\\\?\\" + os.path.abspath(path) if os.name == "nt" else path
    st = os.stat(src)
    key = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    folder = os.path.join(get_app_dir(), "cache", "nguon")
    os.makedirs(folder, exist_ok=True)
    dst = os.path.join(folder, hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + os.path.splitext(path)[1])
    if not os.path.isfile(dst):
        shutil.copy2(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    return dst


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
