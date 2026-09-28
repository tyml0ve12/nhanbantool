"""Giao dien tu cap nhat: kiem tra ngam khi mo app, hoi nguoi dung, tai + cai, khoi dong lai."""
import logging

from PySide6.QtCore import QObject, QThread, Qt, Signal, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from core import updater
from core.version import APP_VERSION

log = logging.getLogger("update_ui")


class UpdateCheckWorker(QThread):
    found = Signal(object)   # UpdateInfo
    up_to_date = Signal()
    failed = Signal(str)

    def run(self):
        try:
            info = updater.check_for_update()
        except Exception as e:
            log.warning("Kiem tra cap nhat that bai: %s", e)
            self.failed.emit(str(e))
            return
        if info:
            self.found.emit(info)
        else:
            self.up_to_date.emit()


class UpdateInstallWorker(QThread):
    progress = Signal(int)
    stage = Signal(str)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.info = info

    def run(self):
        try:
            self.stage.emit(f"Đang tải bản v{self.info.version}...")
            zip_path = updater.download_update(self.info, self.progress.emit)
            self.stage.emit("Đang cài bản cập nhật (có thể mất vài phút nếu cần tải thêm thư viện)...")
            self.progress.emit(0)
            updater.apply_update(zip_path)
        except Exception as e:
            log.exception("Cap nhat that bai")
            self.failed.emit(str(e))
            return
        self.finished_ok.emit()


class UpdateController(QObject):
    """Gan vao MainWindow. has_unsaved_work: ham tra ve True neu nguoi dung
    dang co viec se mat khi app khoi dong lai (vd dang chay long tieng)."""

    def __init__(self, window, has_unsaved_work):
        super().__init__(window)
        self.window = window
        self.has_unsaved_work = has_unsaved_work
        self.check_worker = None
        self.install_worker = None
        self.progress_dialog = None
        self.manual = False

    def check_on_startup(self):
        if updater.is_update_enabled():
            QTimer.singleShot(2000, lambda: self._start_check(manual=False))

    def check_manually(self):
        if not updater.is_update_enabled():
            QMessageBox.information(
                self.window, "Cập nhật",
                f"Bạn đang dùng Nhân Bản Lồng Tiếng v{APP_VERSION}.\n"
                "Tự cập nhật chỉ hoạt động ở bản đã cài bằng CAI DAT.bat."
            )
            return
        self._start_check(manual=True)

    def _start_check(self, manual):
        if self.check_worker and self.check_worker.isRunning():
            return
        if self.install_worker and self.install_worker.isRunning():
            return
        self.manual = manual
        self.check_worker = UpdateCheckWorker(self)
        self.check_worker.found.connect(self._on_found)
        self.check_worker.up_to_date.connect(self._on_up_to_date)
        self.check_worker.failed.connect(self._on_check_failed)
        self.check_worker.start()

    def _on_up_to_date(self):
        if self.manual:
            QMessageBox.information(self.window, "Cập nhật",
                                    f"Bạn đang dùng bản mới nhất (v{APP_VERSION}).")

    def _on_check_failed(self, reason):
        # Mo app luc khong co mang -> im lang bo qua, chi bao khi nguoi dung tu bam kiem tra.
        if self.manual:
            QMessageBox.warning(self.window, "Cập nhật",
                                f"Không kiểm tra được bản cập nhật.\nKiểm tra kết nối internet rồi thử lại.\n\n{reason}")

    def _on_found(self, info):
        notes = info.notes.strip() or "(không có ghi chú)"
        text = (f"Đã có bản mới <b>v{info.version}</b> (bạn đang dùng v{APP_VERSION}).<br><br>"
                f"<b>Có gì mới:</b><br>{_escape(notes)}")
        if self.has_unsaved_work():
            text += "<br><br><i>App sẽ khởi động lại sau khi cập nhật — lần chạy đang dở sẽ bị dừng.</i>"

        box = QMessageBox(self.window)
        box.setWindowTitle("Có bản cập nhật mới")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(text)
        update_btn = box.addButton("Cập nhật ngay", QMessageBox.ButtonRole.AcceptRole)
        if info.mandatory:
            box.setInformativeText("Đây là bản cập nhật bắt buộc.")
            later_btn = box.addButton("Thoát app", QMessageBox.ButtonRole.RejectRole)
        else:
            later_btn = box.addButton("Để sau", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(update_btn)
        box.exec()

        if box.clickedButton() is update_btn:
            self._install(info)
        elif info.mandatory:
            QApplication.quit()

    def _install(self, info):
        self.progress_dialog = QProgressDialog("Đang chuẩn bị...", None, 0, 100, self.window)
        self.progress_dialog.setWindowTitle("Đang cập nhật")
        self.progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setMinimumWidth(420)
        self.progress_dialog.show()

        self.install_worker = UpdateInstallWorker(info, self)
        self.install_worker.progress.connect(self.progress_dialog.setValue)
        self.install_worker.stage.connect(self.progress_dialog.setLabelText)
        self.install_worker.finished_ok.connect(lambda: self._on_installed(info))
        self.install_worker.failed.connect(lambda reason: self._on_install_failed(info, reason))
        self.install_worker.start()

    def _on_installed(self, info):
        self.progress_dialog.close()
        QMessageBox.information(self.window, "Cập nhật xong",
                                f"Đã cập nhật lên v{info.version}. App sẽ khởi động lại.")
        try:
            updater.restart_app()
        except Exception as e:
            log.exception("Khong tu khoi dong lai duoc")
            QMessageBox.information(self.window, "Cập nhật xong",
                                    f"Vui lòng tự mở lại app.\n\n{e}")
        QApplication.quit()

    def _on_install_failed(self, info, reason):
        self.progress_dialog.close()
        QMessageBox.warning(self.window, "Cập nhật thất bại",
                            f"Không cập nhật được lên v{info.version}. App vẫn giữ nguyên bản cũ.\n\n{reason}")
        if info.mandatory:
            QApplication.quit()


def _escape(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("\n", "<br>"))
