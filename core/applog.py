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
    return log_path
