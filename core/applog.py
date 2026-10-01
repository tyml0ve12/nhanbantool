"""Ghi log toan bo hoat dong cua app ra file, de tra cuu khi co loi (app chay
bang pythonw.exe nen khong co cua so dong lenh de xem loi truc tiep)."""
import logging
import os
from datetime import datetime

from core.paths import get_app_dir

LOG_PREFIX = "nhanban_"


def get_log_dir() -> str:
    log_dir = os.path.join(get_app_dir(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setup_logging() -> str:
    """Goi 1 lan luc khoi dong app. Tra ve duong dan file log dang dung."""
    log_dir = get_log_dir()
    log_path = os.path.join(log_dir, f"{LOG_PREFIX}{datetime.now():%Y%m%d_%H%M%S}.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
    ))
    root.addHandler(file_handler)

    # Don cac file log cu, chi giu 20 file gan nhat
    try:
        files = sorted(
            (f for f in os.listdir(log_dir) if f.startswith(LOG_PREFIX) and f.endswith(".log")),
            reverse=True,
        )
        for old_file in files[20:]:
            os.remove(os.path.join(log_dir, old_file))
    except OSError:
        pass

    logging.info("=== Nhan Ban Long Tieng khoi dong - log tai: %s ===", log_path)
    _log_environment()
    return log_path


# Thu vien hay gay loi khi lech phien ban giua cac may (vd av 19 bo metadata_errors)
_PACKAGES = ("faster-whisper", "av", "ctranslate2", "onnxruntime", "tokenizers", "numpy", "PySide6")


def _log_environment():
    """Ghi phien ban app / Python / Windows / thu vien -> doc log la biet may khach khac gi."""
    import platform
    import sys
    from importlib import metadata
    from core.version import APP_VERSION

    versions = []
    for name in _PACKAGES:
        try:
            versions.append(f"{name}={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            versions.append(f"{name}=(chua cai)")
    logging.info("App v%s | Python %s (%s) | %s", APP_VERSION, platform.python_version(),
                 sys.executable, platform.platform())
    logging.info("Thu vien: %s", ", ".join(versions))


def make_support_zip(max_files: int = 5) -> str:
    """Nen cac file log moi nhat thanh 1 file zip de nguoi dung gui ve. Tra ve duong dan zip."""
    import zipfile

    log_dir = get_log_dir()
    files = sorted(
        (f for f in os.listdir(log_dir) if f.startswith(LOG_PREFIX) and f.endswith(".log")),
        reverse=True,
    )[:max_files]
    zip_path = os.path.join(log_dir, f"log_gui_ho_tro_{datetime.now():%Y%m%d_%H%M%S}.zip")
    for handler in logging.getLogger().handlers:
        handler.flush()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(os.path.join(log_dir, f), f)
    return zip_path
