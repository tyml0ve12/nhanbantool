"""Hop thoai Xuat video (tham khao Tool Join Video). 2 giao dien rieng:
- 1 video : gon - cau hinh + ten video + 1 thanh tien do lon.
- Nhieu video: cau hinh DUNG CHUNG o tren + danh sach tung video (ten file
  "Ngon ngu - Ten video goc", nhan co logo/copy, thanh % rieng) + tien do tong.
Nhan cau hinh -> phat tin hieu start_requested; hop thoai o lai hien tien do."""
import os

from PySide6.QtCore import Qt, QSettings, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QLabel, QComboBox, QPushButton,
    QTabWidget, QWidget, QSlider, QLineEdit, QFileDialog, QProgressBar, QMessageBox, QScrollArea, QFrame,
)

from core.render import RECOMMENDED, RenderSettings

EXPORT_BUTTON_STYLE = """
    QPushButton { background-color: #1E6FFF; color: white; font-weight: bold; padding: 8px 20px;
                  border-radius: 6px; border: none; }
    QPushButton:hover { background-color: #3D82FF; }
    QPushButton:disabled { background-color: #4a4a4a; color: #999; }
"""
TAG_STYLE = {
    "logo": "background:#3a2f12; color:#e0b04a; border-radius:4px; padding:1px 6px;",
    "copy": "background:#123a26; color:#4cc38a; border-radius:4px; padding:1px 6px;",
    "encode": "background:#12304a; color:#5aa9ff; border-radius:4px; padding:1px 6px;",
}
RESOLUTIONS = [("Giữ như video gốc", 0, 0), ("3840 × 2160 (4K)", 3840, 2160),
               ("2560 × 1440 (2K)", 2560, 1440), ("1920 × 1080 (Full HD)", 1920, 1080)]
SPEEDS = [("Nhanh nhất (khuyên dùng)", "fast"), ("Cân bằng", "balanced"), ("Chất lượng cao (chậm)", "quality")]
CFG = "export_settings"


def safe_filename(name: str) -> str:
    name = (name or "").strip() or "video"
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name


class ExportDialog(QDialog):
    start_requested = Signal()

    def __init__(self, parent, video_path: str, video_info, languages: list, output_dir: str):
        """languages: [(row, ten_ngon_ngu, co_logo)]"""
        super().__init__(parent)
        self.video_path = video_path
        self.info = video_info
        self.languages = languages
        self.multi = len(languages) > 1
        self.settings_store = QSettings()
        self.rows = {}          # row -> dict widget cua dong video
        self.setWindowTitle("Xuất video" + (f" — {len(languages)} ngôn ngữ" if self.multi else ""))
        self.resize(900 if self.multi else 600, 760 if self.multi else 480)
        self._build(output_dir)
        self._load()
        self._refresh()

    # ------------------------------------------------------------------ UI
    def _build(self, output_dir):
        root = QVBoxLayout(self)
        src = QLabel(f"Video gốc: <b>{os.path.basename(self.video_path)}</b> · {self.info.width}×{self.info.height} · "
                     f"{self.info.fps:g} fps · {self.info.codec.upper()} · ~{self.info.bitrate_kbps / 1000:.0f} Mbps")
        src.setObjectName("hint")
        root.addWidget(src)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._recommended_tab(), "Khuyến nghị (chuẩn YouTube)")
        self.tabs.addTab(self._custom_tab(), "Tùy chỉnh")
        self.tabs.currentChanged.connect(self._refresh)
        root.addWidget(self.tabs)

        if self.multi:
            shared = QLabel(f"Cấu hình trên áp dụng cho <b>tất cả {len(self.languages)} video</b> bên dưới.")
            shared.setObjectName("hint")
            root.addWidget(shared)

        folder = QHBoxLayout()
        folder.addWidget(QLabel("Lưu tại:"))
        self.dir_edit = QLineEdit(output_dir)
        folder.addWidget(self.dir_edit, 1)
        browse = QPushButton("Chọn…")
        browse.clicked.connect(self._browse)
        folder.addWidget(browse)
        root.addLayout(folder)

        if self.multi:
            root.addWidget(self._video_list(), 1)
        else:
            root.addLayout(self._single_video())

        self.estimate = QLabel()
        self.estimate.setWordWrap(True)
        self.estimate.setObjectName("hint")
        root.addWidget(self.estimate)

        if self.multi:
            self.total_label = QLabel()
            root.addWidget(self.total_label)

        buttons = QHBoxLayout()
        self.open_btn = QPushButton("Mở thư mục")
        self.open_btn.clicked.connect(self._open_folder)
        self.open_btn.hide()
        buttons.addWidget(self.open_btn)
        buttons.addStretch(1)
        self.close_btn = QPushButton("Huỷ")
        self.close_btn.clicked.connect(self.reject)
        buttons.addWidget(self.close_btn)
        self.start_btn = QPushButton("Bắt đầu xuất" + (f" {len(self.languages)} video" if self.multi else ""))
        self.start_btn.setStyleSheet(EXPORT_BUTTON_STYLE)
        self.start_btn.clicked.connect(self._start)
        buttons.addWidget(self.start_btn)
        root.addLayout(buttons)

    def _recommended_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        is_4k = (self.info.width, self.info.height) == (RECOMMENDED.width, RECOMMENDED.height)
        form.addRow("Độ phân giải:", QLabel("3840 × 2160 (4K UHD)" + (
            "" if is_4k else f" — video gốc {self.info.width} × {self.info.height} sẽ được phóng lên 4K")))
        form.addRow("Codec video:", QLabel("H.265 / HEVC"))
        form.addRow("Bitrate video:", QLabel(f"{RECOMMENDED.bitrate_kbps / 1000:.0f} Mbps (VBR)"))
        form.addRow("Khung hình/giây:", QLabel(f"{RECOMMENDED.fps:g} fps"))
        form.addRow("Định dạng:", QLabel("MP4"))
        form.addRow("Audio:", QLabel(f"AAC {RECOMMENDED.audio_kbps} kbps, 48 kHz stereo"))
        form.addRow("Tốc độ:", QLabel("Nhanh nhất — toàn bộ trên card rời (NVENC), chất lượng như gốc"))
        note = QLabel("Video gốc đã 4K + <b>không logo</b>: giữ nguyên hình gốc (copy, không nén lại) — "
                      "gần như tức thì, chất lượng 100% như gốc.")
        note.setWordWrap(True)
        form.addRow("", note)
        return w

    def _custom_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        self.res_combo = QComboBox()
        for label, rw, rh in RESOLUTIONS:
            self.res_combo.addItem(label if rw else f"{label} ({self.info.width} × {self.info.height})", (rw, rh))
        form.addRow("Độ phân giải:", self.res_combo)
        self.codec_combo = QComboBox()
        self.codec_combo.addItem("H.265 / HEVC (nhẹ hơn, chuẩn YouTube)", "h265")
        self.codec_combo.addItem("H.264 (tương thích rộng)", "h264")
        form.addRow("Codec video:", self.codec_combo)
        bitrate = QHBoxLayout()
        self.bitrate = QSlider(Qt.Orientation.Horizontal)
        self.bitrate.setRange(5, 100)
        self.bitrate_label = QLabel()
        self.bitrate_label.setMinimumWidth(70)
        self.bitrate.valueChanged.connect(lambda v: self.bitrate_label.setText(f"{v} Mbps"))
        bitrate.addWidget(self.bitrate)
        bitrate.addWidget(self.bitrate_label)
        form.addRow("Bitrate video:", bitrate)
        self.fps_combo = QComboBox()
        self.fps_combo.addItem(f"Giữ như video gốc ({self.info.fps:g} fps)", 0.0)
        self.fps_combo.addItem("60 fps", 60.0)
        self.fps_combo.addItem("30 fps", 30.0)
        form.addRow("Khung hình/giây:", self.fps_combo)
        self.audio_combo = QComboBox()
        for kbps in (384, 256, 192, 128):
            self.audio_combo.addItem(f"{kbps} kbps", kbps)
        form.addRow("Bitrate audio:", self.audio_combo)
        self.speed_combo = QComboBox()
        for label, key in SPEEDS:
            self.speed_combo.addItem(label, key)
        form.addRow("Tốc độ nén:", self.speed_combo)
        for c in (self.res_combo, self.codec_combo, self.fps_combo, self.audio_combo, self.speed_combo):
            c.currentIndexChanged.connect(self._refresh)
        self.bitrate.valueChanged.connect(self._refresh)
        return w

    def _single_video(self):
        row, name, has_logo = self.languages[0]
        box = QVBoxLayout()
        line = QHBoxLayout()
        line.addWidget(QLabel("Tên video:"))
        edit = QLineEdit(self._default_name(name))
        line.addWidget(edit, 1)
        line.addWidget(QLabel(".mp4"))
        tag = QLabel()
        line.addWidget(tag)
        box.addLayout(line)
        bar = QProgressBar()
        bar.setMinimumHeight(24)
        bar.hide()
        box.addWidget(bar)
        status = QLabel()
        status.setObjectName("hint")
        box.addWidget(status)
        self.rows[row] = {"name": name, "logo": has_logo, "edit": edit, "tag": tag, "bar": bar, "status": status}
        return box

    def _video_list(self):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        for c, text in enumerate(("Ngôn ngữ", "Tên file", "", "Tiến độ")):
            h = QLabel(text)
            h.setObjectName("hint")
            grid.addWidget(h, 0, c)
        for i, (row, name, has_logo) in enumerate(self.languages, start=1):
            lang = QLabel(f"<b>{name}</b>")
            edit = QLineEdit(self._default_name(name))
            tag = QLabel()
            bar = QProgressBar()
            bar.setValue(0)
            bar.setFormat("Chờ")
            status = QLabel()
            status.setObjectName("hint")
            status.setMinimumWidth(150)
            grid.addWidget(lang, i, 0)
            grid.addWidget(edit, i, 1)
            grid.addWidget(tag, i, 2)
            grid.addWidget(bar, i, 3)
            grid.addWidget(status, i, 4)
            self.rows[row] = {"name": name, "logo": has_logo, "edit": edit, "tag": tag, "bar": bar, "status": status}
        grid.setRowStretch(len(self.languages) + 1, 1)
        area.setWidget(inner)
        return area

    def _default_name(self, language):
        stem = os.path.splitext(os.path.basename(self.video_path))[0]
        return safe_filename(f"{language} - {stem}")

    # ------------------------------------------------------------ cau hinh
    def render_settings(self) -> RenderSettings:
        if self.tabs.currentIndex() == 0:
            return RenderSettings(**RECOMMENDED.to_dict())
        rw, rh = self.res_combo.currentData()
        # Tuy chinh: nguoi dung chu dong chon codec -> nen dung codec do, khong tu copy hinh goc
        return RenderSettings(width=rw, height=rh, codec=self.codec_combo.currentData(),
                              bitrate_kbps=self.bitrate.value() * 1000, fps=self.fps_combo.currentData(),
                              audio_kbps=self.audio_combo.currentData(), speed=self.speed_combo.currentData(),
                              copy_if_no_logo=False)

    def output_paths(self) -> dict:
        folder = self.dir_edit.text().strip()
        return {row: os.path.join(folder, safe_filename(r["edit"].text()) + ".mp4") for row, r in self.rows.items()}

    def _refresh(self):
        from core.render import needs_encode
        s = self.render_settings()
        encode_count = 0
        for r in self.rows.values():
            enc = needs_encode(s, self.info, r["logo"])
            encode_count += enc
            kind = "logo" if r["logo"] else ("encode" if enc else "copy")
            text = {"logo": "có logo · nén lại", "encode": "nén lại", "copy": "copy nhanh"}[kind]
            r["tag"].setText(text)
            r["tag"].setStyleSheet(TAG_STYLE[kind])
        minutes = self.info.duration / 60
        # Toc do do tren may that (RTX 2060S): 4K ~3.2x (1080p phong len 4K ~2x), 1080p ~7x; copy ~50x
        w, h = (s.width or self.info.width), (s.height or self.info.height)
        if w * h > 2560 * 1440:
            speed = 3.2 if (w, h) == (self.info.width, self.info.height) else 2.0
        else:
            speed = 7.0
        if s.speed == "balanced":
            speed /= 1.8
        elif s.speed == "quality":
            speed /= 2.2
        est = encode_count * minutes / speed + (len(self.rows) - encode_count) * minutes / 50
        size_gb = (s.bitrate_kbps + s.audio_kbps) * self.info.duration / 8 / 1e6
        self.estimate.setText(
            f"Thời lượng {int(minutes)} phút · {encode_count} video phải nén lại, "
            f"{len(self.rows) - encode_count} video copy nhanh · ước tính ~{max(1, round(est))} phút "
            f"(card rời NVIDIA) · ~{size_gb:.1f} GB/video")

    def _load(self):
        st = self.settings_store
        self.tabs.setCurrentIndex(int(st.value(f"{CFG}/tab", 0)))
        c = RenderSettings.from_dict(st.value(f"{CFG}/custom") or {})
        self.res_combo.setCurrentIndex(max(0, self.res_combo.findData((c.width, c.height))))
        self.codec_combo.setCurrentIndex(max(0, self.codec_combo.findData(c.codec)))
        self.bitrate.setValue(c.bitrate_kbps // 1000)
        self.fps_combo.setCurrentIndex(max(0, self.fps_combo.findData(c.fps)))
        self.audio_combo.setCurrentIndex(max(0, self.audio_combo.findData(c.audio_kbps)))
        self.speed_combo.setCurrentIndex(max(0, self.speed_combo.findData(c.speed)))

    def _save(self):
        st = self.settings_store
        st.setValue(f"{CFG}/tab", self.tabs.currentIndex())
        tab = self.tabs.currentIndex()
        self.tabs.setCurrentIndex(1)
        st.setValue(f"{CFG}/custom", self.render_settings().to_dict())
        self.tabs.setCurrentIndex(tab)
        st.setValue(f"{CFG}/last_dir", self.dir_edit.text().strip())

    # ------------------------------------------------------------ hanh dong
    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu video", self.dir_edit.text())
        if d:
            self.dir_edit.setText(os.path.normpath(d))

    def _start(self):
        if not self.dir_edit.text().strip():
            QMessageBox.warning(self, "Xuất video", "Hãy chọn thư mục lưu video.")
            return
        paths = list(self.output_paths().values())
        if len({p.lower() for p in paths}) != len(paths):
            QMessageBox.warning(self, "Xuất video", "Có 2 video trùng tên file — hãy sửa lại tên.")
            return
        # Du cho trong cho TAT CA video chua (bao truoc, khong de render 10 phut roi moi loi)
        import shutil
        from core.render import estimate_size_bytes, needs_encode
        s = self.render_settings()
        need = sum(estimate_size_bytes(s, self.info, not needs_encode(s, self.info, r["logo"]))
                   for r in self.rows.values())
        folder = self.dir_edit.text().strip()
        os.makedirs(folder, exist_ok=True)
        free = shutil.disk_usage(folder).free
        if free < need:
            drive = os.path.splitdrive(os.path.abspath(folder))[0]
            QMessageBox.warning(self, "Không đủ chỗ trống",
                                f"Ổ {drive} chỉ còn trống {free / 1e9:.1f} GB, {len(self.rows)} video cần khoảng "
                                f"{need / 1e9:.1f} GB.\n\nHãy chọn thư mục lưu ở ổ khác.")
            return
        existing = [os.path.basename(p) for p in paths if os.path.exists(p)]
        if existing and QMessageBox.question(self, "Xuất video", "Đã có file:\n" + "\n".join(existing[:5])
                                             + "\n\nGhi đè?") != QMessageBox.StandardButton.Yes:
            return
        self._save()
        self.tabs.setEnabled(False)
        self.dir_edit.setEnabled(False)
        for r in self.rows.values():
            r["edit"].setEnabled(False)
            r["bar"].show()
            r["bar"].setValue(0)
            r["bar"].setFormat("Chờ")
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Đang xử lý…")
        self.close_btn.setText("Ẩn (vẫn chạy)")
        self.close_btn.clicked.disconnect()
        self.close_btn.clicked.connect(self.hide)
        self._update_total()
        self.start_requested.emit()

    # ---------------------------------------------------- tien do tu worker
    def set_progress(self, row, pct, speed="", eta=0.0):
        r = self.rows.get(row)
        if not r:
            return
        r["bar"].setValue(pct)
        r["bar"].setFormat(f"{pct}%")
        eta_txt = f"{int(eta // 60)}:{int(eta % 60):02d}" if eta else "--:--"
        r["status"].setText(f"{speed} · còn {eta_txt}" if speed else "")

    def set_status(self, row, text, kind="run"):
        r = self.rows.get(row)
        if not r:
            return
        if kind in ("ok", "warn"):
            r["bar"].setValue(100)
            r["bar"].setFormat("Xong ✓")
        elif kind == "error":
            r["bar"].setFormat("Lỗi")
        elif r["bar"].value() == 0:
            r["bar"].setFormat(text)
        r["status"].setText(text if kind != "run" or r["bar"].value() == 0 else r["status"].text())
        r["done"] = kind in ("ok", "warn", "error")
        self._update_total()

    def _update_total(self):
        if not self.multi:
            return
        done = sum(1 for r in self.rows.values() if r.get("done"))
        self.total_label.setText(f"<b>{done}/{len(self.rows)} video xong</b>")

    def mark_finished(self, ok, err):
        self.start_btn.setText("Xong" if not err else f"Xong · {err} lỗi")
        self.close_btn.setText("Đóng")
        self.close_btn.clicked.disconnect()
        self.close_btn.clicked.connect(self.accept)
        self.open_btn.show()

    def _open_folder(self):
        folder = self.dir_edit.text().strip()
        if os.path.isdir(folder):
            os.startfile(folder)
