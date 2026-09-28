"""Cua so chinh: Project goc -> Danh sach ngon ngu -> Chay -> Log."""
import logging
import os
from datetime import datetime

from PySide6.QtCore import Qt, QSettings, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QRadioButton, QButtonGroup,
    QProgressBar, QPlainTextEdit, QStatusBar, QMessageBox, QInputDialog,
)

from core import draft_patch
from core.capcut_project import (
    ProjectError, load_project, scan_audio_folder, language_name_from_file,
    list_capcut_projects, is_registered_in_capcut,
    is_capcut_running, format_us, AUDIO_EXTS,
)
from core.speech import detect_device, is_model_downloaded
from core.version import APP_VERSION
from ui.update_ui import UpdateController
from ui.run_worker import RunWorker, RunConfig, LanguageJob, MODE_DRAFT, MODE_EXPORT, MODE_BOTH

log = logging.getLogger("main_window")

APP_SIGNATURE = "BrightStar - Hoàng Đức"
APP_TITLE = f"Nhân Bản Lồng Tiếng — {APP_SIGNATURE}"

COL_CHECK, COL_NAME, COL_AUDIO, COL_STATUS = range(4)

# (nhan hien thi, ten model faster-whisper). Model tai ve 1 lan o lan chay dau.
WHISPER_MODELS = (
    ("Khuyên dùng — large-v3-turbo (~1.6 GB)", "large-v3-turbo"),
    ("Máy không có card NVIDIA — small (~0.5 GB)", "small"),
)

STATUS_COLORS = {
    "wait": "#9a9a9a",
    "run": "#5aa9ff",
    "ok": "#4cc38a",
    "warn": "#e0b04a",
    "error": "#ff6b6b",
}

STYLE = """
QMainWindow, QWidget { background: #1e1e1e; color: #e8e8e8; font-size: 13px; }
QGroupBox { border: 1px solid #3a3a3a; border-radius: 8px; margin-top: 14px; padding: 10px 10px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #bdbdbd; font-weight: bold; }
QLineEdit, QComboBox, QPlainTextEdit, QTableWidget {
    background: #262626; border: 1px solid #3a3a3a; border-radius: 5px; padding: 4px 6px; }
QLineEdit:focus, QComboBox:focus { border-color: #2f8fff; }
QPushButton { background: #2d2d2d; border: 1px solid #454545; border-radius: 5px; padding: 5px 12px; }
QPushButton:hover { background: #363636; }
QPushButton:disabled { color: #6a6a6a; }
QHeaderView::section { background: #2a2a2a; color: #bdbdbd; border: none; border-bottom: 1px solid #3a3a3a; padding: 5px; }
QTableWidget { gridline-color: #303030; }
QTableWidget::item:selected { background: #24466e; }
QProgressBar { background: #262626; border: 1px solid #3a3a3a; border-radius: 5px; text-align: center; height: 18px; }
QProgressBar::chunk { background: #2f8fff; border-radius: 4px; }
QRadioButton::indicator { width: 12px; height: 12px; border-radius: 7px; border: 1px solid #777; background: #262626; }
QRadioButton::indicator:checked { background: #2f8fff; border: 1px solid #2f8fff; }
QLabel#hint { color: #9a9a9a; }
QLabel#projectInfo { color: #4cc38a; }
QLabel#projectError { color: #ff6b6b; }
"""

RUN_BUTTON_STYLE = (
    "QPushButton { background: #2f8fff; color: white; border: none; font-weight: bold; padding: 8px 22px; }"
    "QPushButton:hover { background: #4a9dff; }"
    "QPushButton:disabled { background: #2a4a70; color: #9ab; }"
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} · v{APP_VERSION}")
        self.resize(1000, 860)
        self.setStyleSheet(STYLE)
        self.settings = QSettings()
        self.project = None
        self.project_registered = False
        self._loaded_path = ""
        self.worker = None
        self.updates = UpdateController(self, has_unsaved_work=lambda: bool(self.worker and self.worker.isRunning()))
        self._build_ui()
        self._restore_last_paths()
        self.updates.check_on_startup()

        self.capcut_timer = QTimer(self)
        self.capcut_timer.timeout.connect(self._refresh_capcut_state)
        self.capcut_timer.start(3000)
        self._refresh_capcut_state()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 6, 14, 10)
        layout.addWidget(self._build_project_box())
        layout.addWidget(self._build_language_box(), stretch=3)
        layout.addWidget(self._build_run_box())
        layout.addWidget(self._build_log_box(), stretch=2)
        self.setCentralWidget(root)

        self.setStatusBar(QStatusBar())
        self.capcut_label = QLabel()
        self.statusBar().addWidget(self.capcut_label)
        signature = QLabel(f"© {APP_SIGNATURE}")
        signature.setObjectName("hint")
        self.statusBar().addPermanentWidget(signature)
        update_btn = QPushButton(f"v{APP_VERSION} · Kiểm tra cập nhật")
        update_btn.setFlat(True)
        update_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        update_btn.clicked.connect(self.updates.check_manually)
        self.statusBar().addPermanentWidget(update_btn)

    def _path_row(self, grid, row, label, placeholder, on_browse):
        grid.addWidget(QLabel(label), row, 0)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        grid.addWidget(edit, row, 1)
        btn = QPushButton("Chọn…")
        btn.clicked.connect(on_browse)
        grid.addWidget(btn, row, 2)
        return edit

    def _build_project_box(self):
        box = QGroupBox("1. Project gốc")
        grid = QGridLayout(box)
        grid.setColumnStretch(1, 1)

        # Chon theo DANH SACH PROJECT CUA CAPCUT (dung thu muc CapCut that su mo) -
        # tranh sua nham 1 ban copy cung ten ma CapCut khong doc.
        grid.addWidget(QLabel("Project CapCut"), 0, 0)
        self.capcut_combo = QComboBox()
        self.capcut_combo.setEditable(True)
        self.capcut_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.capcut_combo.lineEdit().setPlaceholderText("Gõ tên project để tìm…")
        self.capcut_combo.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        self.capcut_combo.completer().setCompletionMode(self.capcut_combo.completer().CompletionMode.PopupCompletion)
        self.capcut_combo.activated.connect(self._on_capcut_project_chosen)
        grid.addWidget(self.capcut_combo, 0, 1)
        refresh_btn = QPushButton("Làm mới")
        refresh_btn.clicked.connect(self._refresh_capcut_projects)
        grid.addWidget(refresh_btn, 0, 2)

        self.project_edit = self._path_row(grid, 1, "Thư mục project",
                                           "Tự điền khi chọn project ở trên", self._browse_project)
        self.project_edit.editingFinished.connect(lambda: self._load_project(self.project_edit.text()))

        self.project_info = QLabel("Chưa chọn project.")
        self.project_info.setObjectName("hint")
        self.project_info.setWordWrap(True)
        grid.addWidget(self.project_info, 2, 1, 1, 2)

        grid.addWidget(QLabel("Track thoại gốc"), 3, 0)
        self.track_combo = QComboBox()
        grid.addWidget(self.track_combo, 3, 1, 1, 2)

        self.video_edit = self._path_row(grid, 4, "Video gốc (.mp4)",
                                         "Chỉ cần khi xuất mp4", self._browse_video)
        self.output_edit = self._path_row(grid, 5, "Thư mục xuất mp4",
                                          "Mặc định: thư mục 'Xuất lồng tiếng' cạnh project", self._browse_output)
        self._refresh_capcut_projects()
        return box

    def _refresh_capcut_projects(self):
        current = self.capcut_combo.currentData()
        self.capcut_combo.clear()
        for p in list_capcut_projects():
            when = datetime.fromtimestamp(p.modified_us / 1e6).strftime("%d/%m/%Y %H:%M") if p.modified_us else ""
            self.capcut_combo.addItem(f"{p.name}   ·   sửa {when}", p.folder)
        if not self.capcut_combo.count():
            self.capcut_combo.lineEdit().setPlaceholderText("Không đọc được danh sách project của CapCut — "
                                                            "hãy chọn thư mục bên dưới")
        idx = self.capcut_combo.findData(current) if current else -1
        self.capcut_combo.setCurrentIndex(idx)

    def _on_capcut_project_chosen(self, index):
        folder = self.capcut_combo.itemData(index)
        if folder:
            self.project_edit.setText(folder)
            self._load_project(folder)

    def _build_language_box(self):
        box = QGroupBox("2. Ngôn ngữ cần nhân bản")
        v = QVBoxLayout(box)

        toolbar = QHBoxLayout()
        add_btn = QPushButton("＋ Thêm file audio")
        add_btn.clicked.connect(self._add_audio_files)
        scan_btn = QPushButton("Quét thư mục audio…")
        scan_btn.clicked.connect(self._scan_audio_folder)
        remove_btn = QPushButton("Xoá dòng")
        remove_btn.clicked.connect(self._remove_selected)
        for b in (add_btn, scan_btn):
            toolbar.addWidget(b)
        toolbar.addStretch(1)
        toolbar.addWidget(remove_btn)
        v.addLayout(toolbar)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "Tên ngôn ngữ (tên track)", "File audio", "Trạng thái"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_CHECK, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_AUDIO, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(COL_NAME, 210)
        self.table.setColumnWidth(COL_STATUS, 220)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(160)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.itemChanged.connect(self._on_table_changed)
        self.table.setAcceptDrops(True)
        self.table.viewport().setAcceptDrops(True)
        self.table.dragEnterEvent = self._drag_enter
        self.table.dragMoveEvent = self._drag_enter
        self.table.dropEvent = self._drop_files
        v.addWidget(self.table)

        hint = QLabel("Kéo thả file audio vào bảng. Bấm đúp vào tên để sửa — tên này sẽ là tên track trong CapCut "
                      "và tên file mp4. Ranh giới từng câu được tự nhận dạng bằng giọng nói (faster-whisper).")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return box

    def _build_run_box(self):
        box = QGroupBox("3. Chạy")
        v = QVBoxLayout(box)
        h = QHBoxLayout()
        v.addLayout(h)

        h.addWidget(QLabel("Chế độ:"))
        self.mode_group = QButtonGroup(self)
        for text, mode in (("Draft CapCut", MODE_DRAFT), ("Xuất mp4", MODE_EXPORT), ("Cả hai", MODE_BOTH)):
            rb = QRadioButton(text)
            rb.setProperty("mode", mode)
            self.mode_group.addButton(rb)
            h.addWidget(rb)
            if mode == MODE_BOTH:
                rb.setChecked(True)
        self.mode_group.buttonToggled.connect(lambda *_: self._update_mode_widgets())

        h.addSpacing(18)
        self.default_lang_label = QLabel("Bật tiếng trong CapCut:")
        h.addWidget(self.default_lang_label)
        self.default_lang_combo = QComboBox()
        self.default_lang_combo.setMinimumWidth(150)
        h.addWidget(self.default_lang_combo)
        h.addStretch(1)

        h = QHBoxLayout()
        v.addLayout(h)
        h.addWidget(QLabel("Nhận dạng câu:"))
        self.model_combo = QComboBox()
        for text, model in WHISPER_MODELS:
            self.model_combo.addItem(text, model)
        # Khoa luu moi: ban cu da tu luu "large-v3" (nang, de het VRAM) khi turbo chua la mac dinh
        saved = self.settings.value("whisper_model_v2", "large-v3-turbo")
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(saved)))
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        h.addWidget(self.model_combo)
        self.device = detect_device()
        self.device_label = QLabel()
        self.device_label.setObjectName("hint")
        h.addWidget(self.device_label)
        self._on_model_changed()

        h.addStretch(1)
        self.stop_btn = QPushButton("Dừng")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_run)
        h.addWidget(self.stop_btn)
        self.run_btn = QPushButton("Chạy")
        self.run_btn.setStyleSheet(RUN_BUTTON_STYLE)
        self.run_btn.clicked.connect(self._start_run)
        h.addWidget(self.run_btn)
        return box

    def _build_log_box(self):
        box = QGroupBox("4. Tiến trình")
        v = QVBoxLayout(box)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        v.addWidget(self.progress_bar)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        self.log_view.setMinimumHeight(150)
        v.addWidget(self.log_view)

        row = QHBoxLayout()
        row.addStretch(1)
        self.apply_btn = QPushButton("Áp dụng vào project")
        self.apply_btn.setToolTip("Dùng khi lúc chạy CapCut đang mở nên chưa áp dụng được")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_pending)
        self.restore_btn = QPushButton("Khôi phục bản gốc…")
        self.restore_btn.clicked.connect(self._restore_backup)
        self.open_output_btn = QPushButton("Mở thư mục xuất")
        self.open_output_btn.clicked.connect(self._open_output)
        for b in (self.apply_btn, self.restore_btn, self.open_output_btn):
            row.addWidget(b)
        v.addLayout(row)
        return box

    # ------------------------------------------------------------- Project
    def _browse_project(self):
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục project CapCut", self.project_edit.text())
        if path:
            self.project_edit.setText(os.path.normpath(path))
            self._load_project(path)

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn video gốc", self.video_edit.text(),
                                              "Video (*.mp4 *.mov *.mkv)")
        if path:
            self.video_edit.setText(os.path.normpath(path))

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục xuất mp4", self.output_edit.text())
        if path:
            self.output_edit.setText(os.path.normpath(path))

    def _load_project(self, path):
        path = path.strip()
        if not path or (self.project and path == self._loaded_path):
            return
        self.track_combo.clear()
        try:
            info = load_project(path)
        except (ProjectError, OSError) as e:
            self.project = None
            self.project_info.setObjectName("projectError")
            self.project_info.setText(str(e))
            self.project_info.setStyleSheet("")
            self._repolish(self.project_info)
            return

        self.project = info
        self._loaded_path = path
        self.settings.setValue("project_dir", path)
        main = info.main_track
        for t in info.audio_tracks:
            label = f"Track {t.index} · {t.segment_count} câu" + (f" · {t.name}" if t.name else "")
            if t is main:
                label += "  (tự nhận — nhiều câu nhất)"
            self.track_combo.addItem(label, t.index)
        self.track_combo.setCurrentIndex(info.audio_tracks.index(main))

        summary = (f"CapCut {info.app_version} · dài {format_us(info.duration_us)} · "
                   f"{main.segment_count} câu thoại · sẽ cập nhật {len(info.draft_files)} file draft")
        self.project_registered = is_registered_in_capcut(info.drafts_dir)
        if self.project_registered:
            self.project_info.setObjectName("projectInfo")
            self.project_info.setText(summary)
        else:
            self.project_info.setObjectName("projectError")
            self.project_info.setText(
                summary + "\n⚠ Thư mục này KHÔNG phải project CapCut đang quản lý (có thể là bản copy) — "
                "CapCut sẽ không thấy thay đổi. Hãy chọn project ở ô 'Project CapCut' phía trên.")
        self._repolish(self.project_info)
        idx = self.capcut_combo.findData(os.path.normpath(info.drafts_dir))
        self.capcut_combo.setCurrentIndex(idx)

        # Doi project -> dien lai video + thu muc xuat theo project moi
        video = os.path.normpath(info.video_path_found) if info.video_path_found else ""
        self.video_edit.setText(video)
        base = os.path.dirname(video) if video else os.path.dirname(info.drafts_dir)
        self.output_edit.setText(os.path.join(base, "Xuất lồng tiếng"))
        self._log(f"Đã mở project: {info.drafts_dir}")

    @staticmethod
    def _repolish(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    # ------------------------------------------------------------ Languages
    def _add_language(self, name, audio):
        for r in range(self.table.rowCount()):
            if self.table.item(r, COL_AUDIO).data(Qt.ItemDataRole.UserRole) == audio:
                return
        self.table.blockSignals(True)
        r = self.table.rowCount()
        self.table.insertRow(r)

        check = QTableWidgetItem()
        check.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        check.setCheckState(Qt.CheckState.Checked)
        self.table.setItem(r, COL_CHECK, check)

        self.table.setItem(r, COL_NAME, QTableWidgetItem(name))
        self._set_readonly(r, COL_AUDIO, os.path.basename(audio), audio)
        self._set_readonly(r, COL_STATUS, "Chờ", None)
        self.table.item(r, COL_STATUS).setForeground(QColor(STATUS_COLORS["wait"]))
        self.table.blockSignals(False)
        self._refresh_default_langs()

    def _set_readonly(self, row, col, text, data):
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        if data is not None:
            item.setData(Qt.ItemDataRole.UserRole, data)
            item.setToolTip(data)
        self.table.setItem(row, col, item)

    def _add_audio_files(self):
        exts = " ".join(f"*{e}" for e in AUDIO_EXTS)
        files, _ = QFileDialog.getOpenFileNames(self, "Chọn file audio ngôn ngữ mới",
                                                self.settings.value("audio_dir", ""), f"Audio ({exts})")
        for f in files:
            self._add_audio_path(f)
        if files:
            self.settings.setValue("audio_dir", os.path.dirname(files[0]))

    def _add_audio_path(self, path):
        path = os.path.normpath(path)
        self._add_language(language_name_from_file(path), path)

    def _scan_audio_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa audio các ngôn ngữ",
                                                  self.settings.value("audio_dir", ""))
        if not folder:
            return
        self.settings.setValue("audio_dir", folder)
        found = scan_audio_folder(folder)
        if not found:
            QMessageBox.information(self, "Quét thư mục", "Không tìm thấy file audio nào trong thư mục này.")
            return
        for name, audio in found:
            self._add_language(name, os.path.normpath(audio))
        self._log(f"Quét thư mục: thêm {len(found)} ngôn ngữ từ {folder}")

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self._refresh_default_langs()

    def _on_table_changed(self, item):
        if item.column() in (COL_NAME, COL_CHECK):
            self._refresh_default_langs()

    def _checked_rows(self):
        return [r for r in range(self.table.rowCount())
                if self.table.item(r, COL_CHECK).checkState() == Qt.CheckState.Checked]

    def _refresh_default_langs(self):
        current = self.default_lang_combo.currentText()
        names = [self.table.item(r, COL_NAME).text().strip() for r in self._checked_rows()]
        self.default_lang_combo.blockSignals(True)
        self.default_lang_combo.clear()
        self.default_lang_combo.addItems(names)
        if current in names:
            self.default_lang_combo.setCurrentText(current)
        self.default_lang_combo.blockSignals(False)
        self.run_btn.setText(f"Chạy {len(names)} ngôn ngữ" if names else "Chạy")

    def _drag_enter(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def _drop_files(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                for name, audio in scan_audio_folder(path):
                    self._add_language(name, os.path.normpath(audio))
            elif path.lower().endswith(AUDIO_EXTS):
                self._add_audio_path(path)
        event.acceptProposedAction()

    # ------------------------------------------------------------------ Run
    def _on_model_changed(self):
        model = self.model_combo.currentData()
        self.settings.setValue("whisper_model_v2", model)
        where = "GPU NVIDIA" if self.device == "cuda" else "CPU (chậm hơn, nên chọn model nhỏ)"
        note = "" if is_model_downloaded(model) else " · lần chạy đầu sẽ tải model"
        self.device_label.setText(f"Chạy trên {where}{note}")

    def _mode(self):
        return self.mode_group.checkedButton().property("mode")

    def _update_mode_widgets(self):
        uses_draft = self._mode() in (MODE_DRAFT, MODE_BOTH)
        self.default_lang_label.setEnabled(uses_draft)
        self.default_lang_combo.setEnabled(uses_draft)

    def _validate(self):
        if not self.project:
            return "Hãy chọn thư mục project CapCut trước."
        rows = self._checked_rows()
        if not rows:
            return "Hãy thêm và tích chọn ít nhất 1 ngôn ngữ."
        names = [self.table.item(r, COL_NAME).text().strip() for r in rows]
        if any(not n for n in names):
            return "Có ngôn ngữ chưa có tên."
        if len(set(n.lower() for n in names)) != len(names):
            return "Có 2 ngôn ngữ trùng tên — hãy đổi tên để phân biệt track và file mp4."
        for r in rows:
            if not os.path.isfile(self.table.item(r, COL_AUDIO).data(Qt.ItemDataRole.UserRole)):
                return f"Không tìm thấy file audio của '{self.table.item(r, COL_NAME).text()}'."
        if self._mode() in (MODE_EXPORT, MODE_BOTH):
            if not os.path.isfile(self.video_edit.text().strip()):
                return "Chế độ xuất mp4 cần chọn video gốc (.mp4)."
            if not self.output_edit.text().strip():
                return "Hãy chọn thư mục xuất mp4."
        return ""

    def _start_run(self):
        error = self._validate()
        if error:
            QMessageBox.warning(self, "Chưa chạy được", error)
            return
        if self._mode() in (MODE_DRAFT, MODE_BOTH) and not self.project_registered:
            answer = QMessageBox.question(
                self, "Project không có trong CapCut",
                "Thư mục đang chọn KHÔNG phải project CapCut đang quản lý, nên mở CapCut sẽ không thấy "
                "track lồng tiếng mới.\n\nNên chọn lại ở ô 'Project CapCut'. Vẫn chạy tiếp?")
            if answer != QMessageBox.StandardButton.Yes:
                return

        track_index = self.track_combo.currentData()
        track = next(t for t in self.project.audio_tracks if t.index == track_index)
        jobs = []
        for r in self._checked_rows():
            jobs.append(LanguageJob(
                row=r,
                name=self.table.item(r, COL_NAME).text().strip(),
                audio_path=self.table.item(r, COL_AUDIO).data(Qt.ItemDataRole.UserRole),
            ))
            self._set_status(r, "Chờ", "wait")

        config = RunConfig(
            project_dir=self.project.drafts_dir,
            draft_files=self.project.draft_files,
            track_index=track_index,
            sentence_count=track.segment_count,
            video_path=self.video_edit.text().strip(),
            output_dir=self.output_edit.text().strip(),
            mode=self._mode(),
            default_language=self.default_lang_combo.currentText(),
            whisper_model=self.model_combo.currentData(),
            device=self.device,
            languages=jobs,
        )
        self.worker = RunWorker(config, self)
        self.worker.log.connect(self._log)
        self.worker.language_status.connect(self._set_status)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished_all.connect(self._on_finished)
        self._set_running(True)
        self.progress_bar.setValue(0)
        self.worker.start()

    def _stop_run(self):
        if self.worker:
            self.worker.stop()
            self.stop_btn.setEnabled(False)

    def _on_finished(self, ok_count, err_count, pending_apply):
        self._set_running(False)
        self.apply_btn.setEnabled(pending_apply)
        self._log(f"Hoàn tất: {ok_count} ngôn ngữ xong, {err_count} lỗi.")
        if err_count and not ok_count:
            QMessageBox.warning(self, "Chưa xong", "Tất cả ngôn ngữ đều lỗi — xem chi tiết trong phần Tiến trình.")

    def _set_running(self, running):
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.table.setEnabled(not running)

    def _set_status(self, row, text, kind):
        item = self.table.item(row, COL_STATUS)
        if item:
            item.setText(text)
            item.setForeground(QColor(STATUS_COLORS.get(kind, STATUS_COLORS["wait"])))

    # ---------------------------------------------------------------- Misc
    def _ensure_capcut_closed(self, action):
        if is_capcut_running():
            QMessageBox.warning(self, action, "CapCut đang mở. Hãy tắt hẳn CapCut rồi thử lại "
                                              "(nếu không CapCut có thể ghi đè ngược lên project).")
            return False
        return True

    def _apply_pending(self):
        if not self.project or not self._ensure_capcut_closed("Áp dụng vào project"):
            return
        try:
            stamp = draft_patch.apply_to_project(self.project.draft_files)
        except (ProjectError, OSError) as e:
            QMessageBox.warning(self, "Áp dụng vào project", str(e))
            return
        self.apply_btn.setEnabled(False)
        self._log(f"Đã áp dụng vào project (backup {stamp}). Mở CapCut để kiểm tra.")

    def _restore_backup(self):
        if not self.project:
            QMessageBox.information(self, "Khôi phục bản gốc", "Hãy chọn project trước.")
            return
        stamps = draft_patch.list_backups(self.project.draft_files)
        if not stamps:
            QMessageBox.information(self, "Khôi phục bản gốc", "Project này chưa có bản backup nào do tool tạo.")
            return
        labels = [f"{s[6:8]}/{s[4:6]}/{s[:4]} {s[9:11]}:{s[11:13]}:{s[13:15]}" for s in stamps]
        choice, ok = QInputDialog.getItem(self, "Khôi phục bản gốc",
                                          "Chọn bản backup (trạng thái project TRƯỚC lần áp dụng đó):",
                                          labels, 0, False)
        if not ok or not self._ensure_capcut_closed("Khôi phục bản gốc"):
            return
        try:
            draft_patch.restore_backup(self.project.draft_files, stamps[labels.index(choice)])
        except (ProjectError, OSError) as e:
            QMessageBox.warning(self, "Khôi phục bản gốc", str(e))
            return
        self._log(f"Đã khôi phục project về bản backup {choice}.")

    def _open_output(self):
        path = self.output_edit.text().strip()
        if path and os.path.isdir(path):
            os.startfile(path)
        else:
            QMessageBox.information(self, "Mở thư mục xuất", "Thư mục xuất chưa tồn tại (sẽ được tạo khi xuất mp4).")

    def _refresh_capcut_state(self):
        if is_capcut_running():
            self.capcut_label.setText("● CapCut đang mở — sẽ chưa áp dụng vào project được")
            self.capcut_label.setStyleSheet("color: #e0b04a;")
        else:
            self.capcut_label.setText("● CapCut đang tắt — sẵn sàng áp dụng")
            self.capcut_label.setStyleSheet("color: #4cc38a;")

    def _log(self, text):
        log.info(text)
        self.log_view.appendPlainText(text)

    def _restore_last_paths(self):
        last = self.settings.value("project_dir", "")
        if last and os.path.isdir(last):
            self.project_edit.setText(last)
            self._load_project(last)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, "Đang chạy", "Đang xử lý, bạn chắc chắn muốn thoát?") \
                    != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()
