"""Nhan Ban Long Tieng - BrightStar - Hoang Duc
Diem vao chinh cua ung dung.
"""
import logging
import sys
import traceback

from PySide6.QtWidgets import QApplication

from core.applog import setup_logging
from ui.main_window import MainWindow, APP_TITLE


def main():
    setup_logging()
    log = logging.getLogger("main")

    def log_uncaught_exception(exc_type, exc_value, exc_tb):
        log.critical("Loi khong bat duoc:\n%s",
                     "".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
    sys.excepthook = log_uncaught_exception

    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setOrganizationName("Hoang Duc")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
