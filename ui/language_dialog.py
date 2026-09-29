"""Chon audio cho ngon ngu: 1 ngon ngu co the gom nhieu phan (1.mp3 ... 10.mp3).

- pick_parts(): hop thoai chon 1 hoac nhieu file -> danh sach phan da sap 1 -> 10
- summarize_parts(): "10 phần · 21:40 · ✓ 1→10" / "✗ thiếu phần 4"
- MultiLanguageDialog: popup them NHIEU ngon ngu 1 lan (Ngon ngu | File audio | Logo kenh)
"""
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QHeaderView, QPushButton,
    QComboBox, QLabel, QFileDialog, QMessageBox, QAbstractItemView,
)

from core import audio, parts
from core.capcut_project import AUDIO_EXTS

COMMON_LANGUAGES = [
    "Tiếng Anh", "Tiếng Tây Ban Nha", "Tiếng Bồ Đào Nha", "Tiếng Pháp", "Tiếng Đức", "Tiếng Ý",
    "Tiếng Nga", "Tiếng Nhật", "Tiếng Hàn", "Tiếng Trung", "Tiếng Thái", "Tiếng Indonesia",
    "Tiếng Ả Rập", "Tiếng Hindi", "Tiếng Thổ Nhĩ Kỳ", "Tiếng Ba Lan", "Tiếng Hà Lan",
]
AUDIO_FILTER = "Audio (" + " ".join(f"*{e}" for e in AUDIO_EXTS) + ")"
IMAGE_FILTER = "Ảnh (*.png *.jpg *.jpeg *.webp *.bmp)"

_durations = {}


def duration_of(path: str) -> float:
    key = (path, os.path.getmtime(path))
    if key not in _durations:
        try:
            _durations[key] = audio.probe_duration(path)
        except Exception:
            _durations[key] = 0.0
    return _durations[key]


def fmt_seconds(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def summarize_parts(paths: list) -> tuple:
    """(chu hien thi, kind ok/warn/error) - de nguoi dung nhin la biet da du phan chua."""
    if not paths:
        return "Chưa chọn file", "error"
    total = sum(duration_of(p) for p in paths)
    if len(paths) == 1:
        return f"{os.path.basename(paths[0])} · {fmt_seconds(total)}", "ok"
    problem = parts.numbering_problem(paths)
    head = f"{len(paths)} phần · {fmt_seconds(total)}"
    if problem.startswith(("thiếu", "trùng")):
        return f"{head} · ✗ {problem}", "error"
    if problem:
        return f"{head} · ⚠ {problem}", "warn"
    first, last = parts.part_number(paths[0]), parts.part_number(paths[-1])
    return f"{head} · ✓ phần {first}→{last} liên tục", "ok"


def parts_tooltip(paths: list) -> str:
    return "\n".join(f"{i + 1}. {os.path.basename(p)} ({fmt_seconds(duration_of(p))})"
                     for i, p in enumerate(paths))


def pick_parts(parent, start_dir: str) -> list:
    files, _ = QFileDialog.getOpenFileNames(parent, "Chọn file audio (chọn nhiều file nếu ngôn ngữ có nhiều phần)",
                                            start_dir, AUDIO_FILTER)
    return parts.sort_parts([os.path.normpath(f) for f in files if not parts.is_merged_file(f)])


def pick_logo(parent, start_dir: str) -> str:
    path, _ = QFileDialog.getOpenFileName(parent, "Chọn logo kênh", start_dir, IMAGE_FILTER)
    return os.path.normpath(path) if path else ""


def language_combo(text: str = "") -> QComboBox:
    combo = QComboBox()
    combo.setEditable(True)
    combo.addItems(COMMON_LANGUAGES)
    combo.setCurrentText(text)
    combo.lineEdit().setPlaceholderText("Gõ hoặc chọn tên ngôn ngữ")
    return combo


STATUS_STYLE = {"ok": "color: #4cc38a;", "warn": "color: #e0b04a;", "error": "color: #ff6b6b;"}


class MultiLanguageDialog(QDialog):
    """Bang: Ngon ngu | File audio (nhieu phan) | Logo kenh | ✕"""

    COL_NAME, COL_AUDIO, COL_LOGO, COL_DEL = range(4)

    def __init__(self, parent, start_dir: str, rows: int = 3, video_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Thêm nhiều ngôn ngữ")
        self.resize(900, 420)
        self.start_dir = start_dir
        self.video_path = video_path
        self._parts = {}   # id(combo) -> [phan]
        self._logos = {}

        v = QVBoxLayout(self)
        hint = QLabel("Mỗi dòng là 1 ngôn ngữ. Ngôn ngữ có nhiều phần (1.mp3 … 10.mp3) thì chọn tất cả các phần "
                      "trong 1 lần — app tự xếp theo số và báo nếu thiếu phần.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        v.addWidget(hint)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Ngôn ngữ (tên track)", "File audio", "Logo kênh", ""])
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(self.COL_NAME, QHeaderView.ResizeMode.Interactive)
        h.setSectionResizeMode(self.COL_AUDIO, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(self.COL_LOGO, QHeaderView.ResizeMode.Interactive)
        h.setSectionResizeMode(self.COL_DEL, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(self.COL_NAME, 200)
        self.table.setColumnWidth(self.COL_LOGO, 190)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        v.addWidget(self.table)

        row = QHBoxLayout()
        add_row = QPushButton("＋ Thêm dòng")
        add_row.clicked.connect(lambda: self._add_row())
        row.addWidget(add_row)
        row.addStretch(1)
        cancel = QPushButton("Huỷ")
        cancel.clicked.connect(self.reject)
        self.ok_btn = QPushButton("Thêm")
        self.ok_btn.setStyleSheet("QPushButton { background: #2f8fff; color: white; border: none; "
                                  "font-weight: bold; padding: 6px 18px; }")
        self.ok_btn.clicked.connect(self._accept)
        row.addWidget(cancel)
        row.addWidget(self.ok_btn)
        v.addLayout(row)

        for _ in range(rows):
            self._add_row()

    def _add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        combo = language_combo("")
        self.table.setCellWidget(r, self.COL_NAME, combo)
        key = id(combo)
        self._parts[key] = []
        self._logos[key] = None

        audio_btn = QPushButton("Chọn file…")
        audio_btn.setStyleSheet("text-align: left; padding-left: 8px;")
        audio_btn.clicked.connect(lambda: self._choose_audio(combo, audio_btn))
        self.table.setCellWidget(r, self.COL_AUDIO, audio_btn)

        logo_btn = QPushButton("＋ Logo…")
        logo_btn.setStyleSheet("text-align: left; padding-left: 8px;")
        logo_btn.clicked.connect(lambda: self._choose_logo(combo, logo_btn))
        self.table.setCellWidget(r, self.COL_LOGO, logo_btn)

        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(32)
        del_btn.clicked.connect(lambda: self._remove_row(combo))
        self.table.setCellWidget(r, self.COL_DEL, del_btn)
        self._update_ok()

    def _row_of(self, combo):
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, self.COL_NAME) is combo:
                return r
        return -1

    def _remove_row(self, combo):
        r = self._row_of(combo)
        if r >= 0:
            self.table.removeRow(r)
            self._parts.pop(id(combo), None)
            self._logos.pop(id(combo), None)
        self._update_ok()

    def _choose_audio(self, combo, button):
        chosen = pick_parts(self, self.start_dir)
        if not chosen:
            return
        self.start_dir = os.path.dirname(chosen[0])
        self._parts[id(combo)] = chosen
        text, kind = summarize_parts(chosen)
        button.setText(text)
        button.setToolTip(parts_tooltip(chosen))
        button.setStyleSheet(f"text-align: left; padding-left: 8px; {STATUS_STYLE[kind]}")
        if not combo.currentText().strip():
            # 1 file: goi y ten file; nhieu phan (1.mp3...): goi y ten thu muc chua phan
            suggest = (os.path.splitext(os.path.basename(chosen[0]))[0] if len(chosen) == 1
                       else os.path.basename(os.path.dirname(chosen[0])))
            combo.setCurrentText(suggest.strip())
        self._update_ok()

    def _choose_logo(self, combo, button):
        # import trong ham: logo_dialog cung import tu module nay
        from ui.logo_dialog import edit_logo, logo_label
        changed, logo = edit_logo(self, combo.currentText().strip() or "ngôn ngữ", self.video_path,
                                  self._logos.get(id(combo)), self.start_dir)
        if changed or not logo:
            self._logos[id(combo)] = logo
            button.setText(logo_label(logo).replace("＋ Thêm logo", "＋ Logo…"))
            button.setToolTip((logo or {}).get("path", ""))

    def _update_ok(self):
        n = sum(1 for r in range(self.table.rowCount()) if self._parts.get(id(self.table.cellWidget(r, 0))))
        self.ok_btn.setText(f"Thêm {n} ngôn ngữ" if n else "Thêm")

    def entries(self) -> list:
        """[(ten, [phan], logo)] cac dong da chon audio."""
        out = []
        for r in range(self.table.rowCount()):
            combo = self.table.cellWidget(r, self.COL_NAME)
            chosen = self._parts.get(id(combo))
            if chosen:
                out.append((combo.currentText().strip(), chosen, self._logos.get(id(combo))))
        return out

    def _accept(self):
        entries = self.entries()
        if not entries:
            QMessageBox.warning(self, "Thêm nhiều ngôn ngữ", "Chưa chọn file audio cho dòng nào.")
            return
        if any(not name for name, _, _ in entries):
            QMessageBox.warning(self, "Thêm nhiều ngôn ngữ", "Có dòng đã chọn audio nhưng chưa có tên ngôn ngữ.")
            return
        names = [n.lower() for n, _, _ in entries]
        if len(set(names)) != len(names):
            QMessageBox.warning(self, "Thêm nhiều ngôn ngữ", "Có 2 dòng trùng tên ngôn ngữ.")
            return
        broken = [n for n, p, _ in entries if summarize_parts(p)[1] == "error"]
        if broken and QMessageBox.question(
                self, "Thiếu phần", f"{', '.join(broken)}: số thứ tự các phần bị thiếu/trùng.\nVẫn thêm?") \
                != QMessageBox.StandardButton.Yes:
            return
        self.accept()
