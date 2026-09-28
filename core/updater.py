"""Tu cap nhat qua GitHub Releases (theo HUONG_DAN_BUILD_VA_UPDATE.md).

Moi ban phat hanh tren GitHub co file dinh kem:
  - latest.json     : {"version", "notes", "url", "sha256", "mandatory"}
  - app_update.zip  : CHI code moi (main.py, cli.py, requirements.txt, core/, ui/),
                      KHONG kem models/ (~1.7GB), bin/ (ffmpeg + CUDA) - giu nguyen tren may.
"""
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Callable, Optional

from core.paths import get_app_dir
from core.version import APP_VERSION, GITHUB_REPO

log = logging.getLogger("updater")

APP_FOLDER = "NhanBanLongTieng"   # %LOCALAPPDATA%\NhanBanLongTieng (install.ps1 tao)
# Nhung thu duoc thay khi cap nhat - models/, bin/, cache/, logs/ khong bao gio bi dong vao.
CODE_ITEMS = ("main.py", "cli.py", "requirements.txt", "core", "ui")
INSTALLED_MARKER = ".installed"
USER_AGENT = f"NhanBanLongTieng/{APP_VERSION}"


@dataclass
class UpdateInfo:
    version: str
    notes: str
    url: str
    sha256: str
    mandatory: bool


def parse_version(text: str) -> tuple:
    parts = []
    for p in text.strip().lstrip("vV").split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def is_update_enabled() -> bool:
    """Chi bat khi app chay tu ban da CAI DAT (co file danh dau do install.ps1
    tao) - tranh ghi de len thu muc ma nguon luc dang phat trien."""
    return bool(GITHUB_REPO) and os.path.isfile(os.path.join(get_app_dir(), INSTALLED_MARKER))


def _open_url(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def check_for_update(timeout: float = 10) -> Optional[UpdateInfo]:
    """Tra ve UpdateInfo neu tren GitHub co ban moi hon ban dang chay, nguoc lai None.
    Loi mang se nem exception - noi goi tu quyet dinh bo qua hay bao loi."""
    manifest_url = f"https://github.com/{GITHUB_REPO}/releases/latest/download/latest.json"
    with _open_url(manifest_url, timeout) as resp:
        data = json.loads(resp.read().decode("utf-8-sig"))
    info = UpdateInfo(
        version=str(data["version"]),
        notes=str(data.get("notes", "")),
        url=str(data["url"]),
        sha256=str(data["sha256"]).lower(),
        mandatory=bool(data.get("mandatory", True)),
    )
    if not info.url.startswith("https://"):
        raise ValueError(f"Link tai ban cap nhat khong hop le: {info.url}")
    if parse_version(info.version) <= parse_version(APP_VERSION):
        return None
    return info


def download_update(info: UpdateInfo, on_progress: Callable[[int], None]) -> str:
    """Tai goi cap nhat ve thu muc tam, kiem tra SHA256. Tra ve duong dan file zip."""
    fd, zip_path = tempfile.mkstemp(prefix="nhanban_update_", suffix=".zip")
    hasher = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as out, _open_url(info.url, 60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                hasher.update(chunk)
                done += len(chunk)
                if total:
                    on_progress(int(done * 100 / total))
        if hasher.hexdigest() != info.sha256:
            raise ValueError("Goi cap nhat tai ve bi loi (sai ma kiem tra SHA256), vui long thu lai.")
    except Exception:
        os.remove(zip_path)
        raise
    return zip_path


def _safe_extract(zip_path: str, dest: str):
    dest_real = os.path.realpath(dest)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = os.path.realpath(os.path.join(dest, member.replace("\\", "/")))
            if target != dest_real and not target.startswith(dest_real + os.sep):
                raise ValueError(f"Goi cap nhat chua duong dan khong hop le: {member}")
        zf.extractall(dest)


def _remove(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def _copy_items(src_dir: str, dst_dir: str):
    for name in CODE_ITEMS:
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(dst_dir, name)
        _remove(dst)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, dst)


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _pip_install_requirements(requirements_path: str):
    python_exe = os.path.join(os.path.dirname(sys.executable), "python.exe")
    if not os.path.isfile(python_exe):
        python_exe = sys.executable
    args = [python_exe, "-m", "pip", "install", "--disable-pip-version-check", "-r", requirements_path]
    # Python rieng cua tool (do install.ps1 cai) -> cai thang vao do; Python co
    # san cua may -> --user, giong cach install.ps1 da cai thu vien.
    own_python = os.path.join(os.environ.get("LOCALAPPDATA", ""), APP_FOLDER, "python")
    if not os.path.normcase(sys.prefix).startswith(os.path.normcase(own_python)):
        args.append("--user")
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    log.info("pip install -r requirements: exit %s\n%s\n%s", result.returncode, result.stdout, result.stderr)
    if result.returncode != 0:
        raise RuntimeError("Khong cai duoc thu vien moi can cho ban cap nhat (kiem tra ket noi internet).")


def apply_update(zip_path: str):
    """Thay code dang cai bang code trong goi cap nhat. Neu loi giua chung se
    khoi phuc lai ban cu, khong de app o trang thai hong."""
    app_dir = get_app_dir()
    work_root = os.path.dirname(app_dir)
    staging = os.path.join(work_root, "update_staging")
    backup = os.path.join(work_root, "app_backup")

    _remove(staging)
    os.makedirs(staging)
    _safe_extract(zip_path, staging)
    if not os.path.isfile(os.path.join(staging, "main.py")):
        raise ValueError("Goi cap nhat khong hop le (thieu main.py).")

    old_requirements = _read_text(os.path.join(app_dir, "requirements.txt"))
    new_requirements = _read_text(os.path.join(staging, "requirements.txt"))

    _remove(backup)
    os.makedirs(backup)
    _copy_items(app_dir, backup)

    try:
        if new_requirements and new_requirements != old_requirements:
            _pip_install_requirements(os.path.join(staging, "requirements.txt"))
        _copy_items(staging, app_dir)
    except Exception:
        log.exception("Cap nhat that bai - khoi phuc ban cu")
        _copy_items(backup, app_dir)
        raise
    finally:
        _remove(staging)
        try:
            os.remove(zip_path)
        except OSError:
            pass
    log.info("Da cap nhat code thanh cong.")


def restart_app():
    app_dir = get_app_dir()
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen([sys.executable, os.path.join(app_dir, "main.py")], cwd=app_dir,
                     creationflags=flags, close_fds=True)
