"""TAO KICH BAN NGAT (T-033): chon cong cu doc -> ky hieu ngat, dan kich ban, xem truoc / tai .txt."""
import os
import re

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QLineEdit, QPlainTextEdit,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)

from core import script_break


class ToolSettingsDialog(QDialog):
    """Them / sua / xoa ten cong cu + ky hieu; luu vao file cau hinh rieng."""

    def __init__(self, tools: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt công cụ ngắt")
        self.resize(560, 360)
        v = QVBoxLayout(self)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Tên công cụ", "Ký hiệu ngắt"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        for t in tools:
            self._add_row(t["name"], t["marker"])
        v.addWidget(self.table)
        v.addWidget(QLabel("Bấm đúp vào ô để sửa."))

        row = QHBoxLayout()
        add_btn = QPushButton("Thêm")
        add_btn.clicked.connect(self._add_new)
        del_btn = QPushButton("Xóa")
        del_btn.clicked.connect(self._delete)
        save_btn = QPushButton("Lưu")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Hủy")
        cancel_btn.clicked.connect(self.reject)
        for b in (add_btn, del_btn):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(save_btn)
        row.addWidget(cancel_btn)
        v.addLayout(row)

    def _add_row(self, name: str, marker: str):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(name))
        self.table.setItem(r, 1, QTableWidgetItem(marker))
        return r

    def _add_new(self):
        r = self._add_row("Công cụ mới", "")
        self.table.setCurrentCell(r, 0)
        self.table.editItem(self.table.item(r, 0))

    def _delete(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    def tools(self) -> list:
        out = []
        for r in range(self.table.rowCount()):
            name = (self.table.item(r, 0).text() if self.table.item(r, 0) else "").strip()
            marker = self.table.item(r, 1).text() if self.table.item(r, 1) else ""
            if name:
                out.append({"name": name, "marker": marker})
        return out

    def _save(self):
        tools = self.tools()
        if not tools:
            QMessageBox.warning(self, "Cài đặt", "Cần ít nhất 1 công cụ.")
            return
        names = [t["name"] for t in tools]
        if len(set(names)) != len(names):
            QMessageBox.warning(self, "Cài đặt", "Tên công cụ bị trùng.")
            return
        try:
            script_break.save_tools(tools)
        except OSError as e:
            QMessageBox.critical(self, "Cài đặt", f"Không lưu được file cấu hình:\n{e}")
            return
        self.accept()


class ScriptBreakDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tạo kịch bản ngắt")
        self.resize(820, 680)
        self.settings = QSettings()
        v = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Công cụ:"))
        self.tool_combo = QComboBox()
        self.tool_combo.setMinimumWidth(180)
        self.tool_combo.currentIndexChanged.connect(self._fill_marker)
        row.addWidget(self.tool_combo)
        gear = QPushButton("⚙ Cài đặt")
        gear.setToolTip("Thêm / sửa / xóa tên công cụ và ký hiệu ngắt")
        gear.clicked.connect(self._open_settings)
        row.addWidget(gear)
        row.addSpacing(16)
        row.addWidget(QLabel("Ký hiệu:"))
        self.marker_edit = QLineEdit()
        self.marker_edit.setToolTip("Tự điền theo công cụ; sửa ở đây chỉ dùng cho lần này (không lưu)")
        row.addWidget(self.marker_edit, 1)
        v.addLayout(row)

        v.addWidget(QLabel("Dán kịch bản:"))
        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText("Dán kịch bản vào đây…")
        v.addWidget(self.input_edit, 1)

        v.addWidget(QLabel("Kết quả (chèn ký hiệu sau . ! ? 。 ！ ？ ؟ và cuối dòng; bỏ qua số, viết tắt, dấu ba chấm):"))
        self.output_edit = QPlainTextEdit()
        self.output_edit.setReadOnly(True)
        v.addWidget(self.output_edit, 1)

        btns = QHBoxLayout()
        self.count_label = QLabel()
        self.count_label.setObjectName("hint")
        btns.addWidget(self.count_label)
        btns.addStretch(1)
        preview_btn = QPushButton("Xem trước")
        preview_btn.clicked.connect(self._preview)
        save_btn = QPushButton("Tải .txt")
        save_btn.clicked.connect(self._save_txt)
        close_btn = QPushButton("Đóng")
        close_btn.clicked.connect(self.close)
        for b in (preview_btn, save_btn, close_btn):
            btns.addWidget(b)
        v.addLayout(btns)

        self._load_tools(self.settings.value("script_break/tool", ""))

    def _load_tools(self, select: str = ""):
        self.tools = script_break.load_tools()
        self.tool_combo.blockSignals(True)
        self.tool_combo.clear()
        for t in self.tools:
            self.tool_combo.addItem(t["name"])
        idx = self.tool_combo.findText(select) if select else -1
        self.tool_combo.setCurrentIndex(max(idx, 0))
        self.tool_combo.blockSignals(False)
        self._fill_marker()

    def _fill_marker(self):
        i = self.tool_combo.currentIndex()
        if 0 <= i < len(self.tools):
            self.marker_edit.setText(self.tools[i]["marker"])
            self.settings.setValue("script_break/tool", self.tools[i]["name"])

    def _open_settings(self):
        dlg = ToolSettingsDialog(self.tools, self)
        if dlg.exec():
            self._load_tools(self.tool_combo.currentText())

    def _result(self):
        text = self.input_edit.toPlainText()
        marker = self.marker_edit.text()
        if not text.strip():
            QMessageBox.information(self, "Tạo kịch bản ngắt", "Chưa dán kịch bản.")
            return None
        if not marker:
            QMessageBox.information(self, "Tạo kịch bản ngắt", "Chưa có ký hiệu ngắt.")
            return None
        out = script_break.insert_breaks(text, marker)
        self.output_edit.setPlainText(out)
        self.count_label.setText(f"Đã chèn {out.count(marker) - text.count(marker)} ký hiệu")
        return out

    def _preview(self):
        self._result()

    def _save_txt(self):
        out = self._result()
        if out is None:
            return
        folder = self.settings.value("script_break/folder", "")
        name = "kich_ban_ngat_" + re.sub(r'[\\/:*?"<>|]', "_", self.tool_combo.currentText()) + ".txt"
        path, _ = QFileDialog.getSaveFileName(self, "Lưu kịch bản ngắt", os.path.join(folder, name), "Text (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(out)
        except OSError as e:
            QMessageBox.critical(self, "Tạo kịch bản ngắt", f"Không lưu được file:\n{e}")
            return
        self.settings.setValue("script_break/folder", os.path.dirname(path))
        QMessageBox.information(self, "Tạo kịch bản ngắt", f"Đã lưu:\n{path}")
