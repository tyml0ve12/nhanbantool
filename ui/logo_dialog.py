"""Hop thoai dat logo kenh: keo tha vi tri + keo goc de doi kich thuoc tren
khung hinh THAT cua video. Luu theo ti le khung hinh (dung moi do phan giai)."""
import os

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QPixmap, QColor, QPen, QImage
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QCheckBox, QWidget, QMessageBox,
)

from core.logo import LogoPlacement, load_logo, grab_frame
from ui.language_dialog import pick_logo

HANDLE = 10          # vung bat goc de keo doi kich thuoc (px man hinh)
MIN_W, MAX_W = 0.03, 0.40


class LogoCanvas(QWidget):
    changed = Signal()

    def __init__(self, frame: QPixmap, parent=None):
        super().__init__(parent)
        self.frame = frame
        self.frame_aspect = frame.width() / frame.height() if not frame.isNull() else 16 / 9
        self.logo = QImage()
        self.placement = LogoPlacement()
        self._drag = None          # ("move"|"resize", diem bat dau, placement luc bat dau, goc co dinh)
        self.setMinimumSize(720, int(720 / self.frame_aspect))
        self.setMouseTracking(True)

    # --- toa do ---
    def frame_rect(self) -> QRectF:
        w, h = self.width(), self.height()
        if w / h > self.frame_aspect:
            fw, fh = h * self.frame_aspect, h
        else:
            fw, fh = w, w / self.frame_aspect
        return QRectF((w - fw) / 2, (h - fh) / 2, fw, fh)

    def logo_aspect(self) -> float:
        return self.logo.width() / self.logo.height() if not self.logo.isNull() else 1.0

    def logo_rect(self) -> QRectF:
        f = self.frame_rect()
        lw = self.placement.width * f.width()
        lh = lw / self.logo_aspect()
        return QRectF(f.x() + self.placement.x * f.width(), f.y() + self.placement.y * f.height(), lw, lh)

    def _clamp(self):
        p = self.placement
        p.width = min(MAX_W, max(MIN_W, p.width))
        h_frac = p.width * self.frame_aspect / self.logo_aspect()
        p.x = min(max(0.0, p.x), 1 - p.width)
        p.y = min(max(0.0, p.y), max(0.0, 1 - h_frac))

    # --- ve ---
    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#151515"))
        f = self.frame_rect()
        if self.frame.isNull():
            painter.fillRect(f, QColor("#3a3a3a"))
            painter.setPen(QColor("#9a9a9a"))
            painter.drawText(f, Qt.AlignmentFlag.AlignCenter, "Chưa chọn video gốc — khung xem trước 16:9")
        else:
            painter.drawPixmap(f, self.frame, QRectF(self.frame.rect()))
        if self.logo.isNull():
            return
        r = self.logo_rect()
        painter.drawImage(r, self.logo)
        painter.setPen(QPen(QColor("white"), 1.5, Qt.PenStyle.DashLine))
        painter.drawRect(r)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("white"))
        for c in (r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()):
            painter.drawRect(QRectF(c.x() - HANDLE / 2, c.y() - HANDLE / 2, HANDLE, HANDLE))

    # --- chuot ---
    def _corner_at(self, pos):
        r = self.logo_rect()
        corners = {"tl": r.topLeft(), "tr": r.topRight(), "bl": r.bottomLeft(), "br": r.bottomRight()}
        for name, c in corners.items():
            if abs(pos.x() - c.x()) <= HANDLE and abs(pos.y() - c.y()) <= HANDLE:
                return name
        return None

    def mousePressEvent(self, e):
        if self.logo.isNull():
            return
        pos = e.position()
        corner = self._corner_at(pos)
        r = self.logo_rect()
        start = LogoPlacement(**self.placement.to_dict())
        if corner:
            opposite = {"tl": r.bottomRight(), "tr": r.bottomLeft(), "bl": r.topRight(), "br": r.topLeft()}[corner]
            self._drag = ("resize", pos, start, opposite, corner)
        elif r.contains(pos):
            self._drag = ("move", pos, start, None, None)

    def mouseMoveEvent(self, e):
        pos = e.position()
        if not self._drag:
            corner = self._corner_at(pos) if not self.logo.isNull() else None
            if corner in ("tl", "br"):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif corner in ("tr", "bl"):
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif not self.logo.isNull() and self.logo_rect().contains(pos):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.unsetCursor()
            return
        kind, origin, start, anchor, corner = self._drag
        f = self.frame_rect()
        if kind == "move":
            self.placement.x = start.x + (pos.x() - origin.x()) / f.width()
            self.placement.y = start.y + (pos.y() - origin.y()) / f.height()
        else:
            # Giu goc doi dien co dinh, doi kich thuoc theo chieu ngang (giu ti le logo)
            new_w_px = abs(pos.x() - anchor.x())
            new_w_px = max(new_w_px, abs(pos.y() - anchor.y()) * self.logo_aspect())
            self.placement.width = min(MAX_W, max(MIN_W, new_w_px / f.width()))
            w_px = self.placement.width * f.width()
            h_px = w_px / self.logo_aspect()
            left = anchor.x() - w_px if corner in ("tl", "bl") else anchor.x()
            top = anchor.y() - h_px if corner in ("tl", "tr") else anchor.y()
            self.placement.x = (left - f.x()) / f.width()
            self.placement.y = (top - f.y()) / f.height()
        self._clamp()
        self.update()
        self.changed.emit()

    def mouseReleaseEvent(self, _):
        self._drag = None


class LogoDialog(QDialog):
    """Tra ve (qua .result_logo) dict {"path", "placement"} hoac None neu bo logo."""

    def __init__(self, parent, language: str, video_path: str, logo: dict = None, start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle(f"Logo kênh — {language}")
        self.start_dir = start_dir
        self.path = (logo or {}).get("path", "")
        self.result_logo = logo
        frame_path = grab_frame(video_path)
        self.canvas = LogoCanvas(QPixmap(frame_path) if frame_path else QPixmap())
        self.canvas.placement = LogoPlacement.from_dict((logo or {}).get("placement"))
        self.canvas.changed.connect(self._sync_slider)

        v = QVBoxLayout(self)
        v.addWidget(self.canvas, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Kích thước"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(MIN_W * 100), int(MAX_W * 100))
        self.slider.valueChanged.connect(self._on_slider)
        row.addWidget(self.slider, 1)
        self.size_label = QLabel()
        self.size_label.setMinimumWidth(110)
        row.addWidget(self.size_label)
        v.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Đặt nhanh:"))
        for text, corner in (("↖ Trên trái", "tl"), ("↗ Trên phải", "tr"), ("↙ Dưới trái", "bl"), ("↘ Dưới phải", "br")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, c=corner: self._snap(c))
            row.addWidget(b)
        row.addStretch(1)
        self.remove_bg = QCheckBox("Xoá nền logo")
        self.remove_bg.setChecked(self.canvas.placement.remove_bg)
        self.remove_bg.toggled.connect(self._reload_logo)
        row.addWidget(self.remove_bg)
        v.addLayout(row)

        row = QHBoxLayout()
        change = QPushButton("Đổi ảnh…")
        change.clicked.connect(self._change_image)
        drop = QPushButton("Bỏ logo")
        drop.clicked.connect(self._drop_logo)
        row.addWidget(change)
        row.addWidget(drop)
        hint = QLabel("Kéo logo để di chuyển · kéo góc để phóng to/thu nhỏ")
        hint.setObjectName("hint")
        row.addWidget(hint, 1)
        cancel = QPushButton("Huỷ")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Lưu vị trí")
        save.setStyleSheet("QPushButton { background: #2f8fff; color: white; border: none; "
                           "font-weight: bold; padding: 6px 18px; }")
        save.clicked.connect(self._save)
        row.addWidget(cancel)
        row.addWidget(save)
        v.addLayout(row)

        is_new = not (logo or {}).get("placement")
        self._reload_logo()
        if is_new:
            self._snap("tr")          # logo moi: mac dinh goc tren phai
        self._sync_slider()

    def _reload_logo(self):
        try:
            self.canvas.logo = load_logo(self.path, self.remove_bg.isChecked()) if self.path else QImage()
        except ValueError as e:
            QMessageBox.warning(self, "Logo", str(e))
            self.canvas.logo = QImage()
        self.canvas.placement.remove_bg = self.remove_bg.isChecked()
        self.canvas._clamp()
        self.canvas.update()

    def _snap(self, corner):
        c = self.canvas
        c.placement = LogoPlacement.corner(corner, c.placement.width, c.logo_aspect(), c.frame_aspect)
        c.placement.remove_bg = self.remove_bg.isChecked()
        c._clamp()
        c.update()
        self._sync_slider()

    def _on_slider(self, value):
        c = self.canvas
        r_before = c.logo_rect()
        c.placement.width = value / 100
        # Giu goc tren-phai co dinh khi keo thanh truot (logo thuong o goc phai)
        f = c.frame_rect()
        c.placement.x = (r_before.right() - c.placement.width * f.width() - f.x()) / f.width()
        c._clamp()
        c.update()
        self.size_label.setText(f"{value}% bề ngang")

    def _sync_slider(self):
        self.slider.blockSignals(True)
        self.slider.setValue(round(self.canvas.placement.width * 100))
        self.slider.blockSignals(False)
        self.size_label.setText(f"{round(self.canvas.placement.width * 100)}% bề ngang")

    def _change_image(self):
        path = pick_logo(self, os.path.dirname(self.path) if self.path else self.start_dir)
        if path:
            self.path = path
            self._reload_logo()

    def _drop_logo(self):
        self.result_logo = None
        self.accept()

    def _save(self):
        if not self.path:
            QMessageBox.warning(self, "Logo", "Chưa chọn ảnh logo.")
            return
        self.result_logo = {"path": self.path, "placement": self.canvas.placement.to_dict()}
        self.accept()


def edit_logo(parent, language: str, video_path: str, logo: dict, start_dir: str):
    """Mo luong them/sua logo. Chua co logo -> chon anh truoc. Tra ve (thay_doi?, logo_moi)."""
    if not logo or not logo.get("path"):
        path = pick_logo(parent, start_dir)
        if not path:
            return False, logo
        logo = {"path": path, "placement": None}
    dlg = LogoDialog(parent, language, video_path, logo, start_dir)
    if dlg.exec():
        return True, dlg.result_logo
    return False, (logo if logo.get("placement") else None)


def logo_label(logo: dict) -> str:
    if not logo or not logo.get("path"):
        return "＋ Thêm logo"
    p = LogoPlacement.from_dict(logo.get("placement"))
    corner = ("↖" if p.x < 0.5 else "↗") if p.y < 0.5 else ("↙" if p.x < 0.5 else "↘")
    return f"{os.path.basename(logo['path'])} · {corner} {round(p.width * 100)}%"
