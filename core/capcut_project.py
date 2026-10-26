"""Doc (CHI DOC) thong tin project CapCut: vi tri cac file draft_content.json,
cac track audio, video goc. Khong ghi gi vao project o day.

CapCut 6.x luu 2 ban draft_content.json giong het nhau:
  <File Drafts>/draft_content.json
  <File Drafts>/Timelines/<main_timeline_id>/draft_content.json
-> khi ap dung ket qua phai thay CA 2.
"""
import ctypes
import json
import os
import subprocess
import time
from dataclasses import dataclass, field

from core.paths import is_network_path

DRAFT_FILE = "draft_content.json"
DRAFTS_DIR_NAME = "File Drafts"
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac")
US = 1_000_000  # CapCut dung microsecond


class ProjectError(Exception):
    """Loi doc project - thong diep hien thang cho nguoi dung."""


@dataclass
class AudioTrack:
    index: int          # vi tri trong d["tracks"]
    track_id: str
    segment_count: int
    name: str


@dataclass
class ProjectInfo:
    drafts_dir: str
    draft_files: list                  # tat ca draft_content.json can thay khi ap dung
    app_version: str
    duration_us: int
    audio_tracks: list = field(default_factory=list)
    video_path_in_draft: str = ""
    video_path_found: str = ""         # video tim thay that tren may (co the rong)
    canvas: tuple = (1920, 1080)       # khung hinh cua project CapCut (de quy doi vi tri logo)

    @property
    def main_track(self):
        return max(self.audio_tracks, key=lambda t: t.segment_count) if self.audio_tracks else None


def find_drafts_dir(path: str) -> str:
    """Nguoi dung co the chon thu muc project (chua 'File Drafts') hoac chon
    thang thu muc chua draft_content.json."""
    if os.path.isfile(os.path.join(path, DRAFT_FILE)):
        return path
    sub = os.path.join(path, DRAFTS_DIR_NAME)
    if os.path.isfile(os.path.join(sub, DRAFT_FILE)):
        return sub
    raise ProjectError(
        f"Không tìm thấy {DRAFT_FILE} trong thư mục đã chọn.\n"
        f"Hãy chọn thư mục project CapCut (có thư mục '{DRAFTS_DIR_NAME}' bên trong)."
    )


def list_draft_files(drafts_dir: str) -> list:
    files = [os.path.join(drafts_dir, DRAFT_FILE)]
    project_json = os.path.join(drafts_dir, "Timelines", "project.json")
    if os.path.isfile(project_json):
        try:
            with open(project_json, encoding="utf-8") as f:
                main_id = json.load(f).get("main_timeline_id", "")
        except (OSError, ValueError):
            main_id = ""
        timeline_draft = os.path.join(drafts_dir, "Timelines", main_id, DRAFT_FILE)
        if main_id and os.path.isfile(timeline_draft):
            files.append(timeline_draft)
    return files


def _guess_video(project_root: str, video_path_in_draft: str) -> str:
    # Duong dan mang (UNC): khong kiem ton tai (treo cho mang), tim theo ten canh project
    if video_path_in_draft and not is_network_path(video_path_in_draft) and os.path.isfile(video_path_in_draft):
        return video_path_in_draft
    # Duong dan trong draft hay bi cu (project da bi chuyen o dia) -> tim file
    # cung ten canh project, roi toi bat ky .mp4 nao canh project.
    wanted = os.path.basename(video_path_in_draft) if video_path_in_draft else ""
    try:
        names = os.listdir(project_root)
    except OSError:
        return ""
    if wanted and wanted in names:
        return os.path.join(project_root, wanted)
    mp4s = [n for n in names if n.lower().endswith(".mp4")]
    return os.path.join(project_root, mp4s[0]) if len(mp4s) == 1 else ""


def load_project(path: str) -> ProjectInfo:
    drafts_dir = find_drafts_dir(path)
    draft_path = os.path.join(drafts_dir, DRAFT_FILE)
    try:
        with open(draft_path, encoding="utf-8") as f:
            d = json.load(f)
    except ValueError as e:
        raise ProjectError(f"File {DRAFT_FILE} bị lỗi, không đọc được: {e}")

    if not isinstance(d.get("tracks"), list) or not isinstance(d.get("materials"), dict):
        raise ProjectError("Cấu trúc draft_content.json không như mong đợi "
                           "(có thể do phiên bản CapCut khác). Tool dừng lại để tránh hỏng project.")

    audio_tracks = []
    for i, t in enumerate(d["tracks"]):
        if t.get("type") == "audio":
            audio_tracks.append(AudioTrack(i, t.get("id", ""), len(t.get("segments", [])), t.get("name", "")))
    if not audio_tracks:
        raise ProjectError("Project không có track audio nào để làm track thoại gốc.")

    video_path = ""
    videos = d["materials"].get("videos") or []
    if videos:
        video_path = videos[0].get("path", "")

    project_root = os.path.dirname(drafts_dir) if os.path.basename(drafts_dir) == DRAFTS_DIR_NAME else drafts_dir
    return ProjectInfo(
        drafts_dir=drafts_dir,
        draft_files=list_draft_files(drafts_dir),
        app_version=(d.get("last_modified_platform") or d.get("platform") or {}).get("app_version", "?"),
        duration_us=int(d.get("duration", 0)),
        audio_tracks=audio_tracks,
        video_path_in_draft=video_path,
        video_path_found=_guess_video(project_root, video_path),
        canvas=(int((d.get("canvas_config") or {}).get("width") or 1920),
                int((d.get("canvas_config") or {}).get("height") or 1080)),
    )


def language_name_from_file(path: str) -> str:
    """'Tiếng Anh.MP3' -> 'Tiếng Anh'."""
    return os.path.splitext(os.path.basename(path))[0].strip()


def scan_audio_folder(folder: str) -> list:
    """Tra ve [(ten_ngon_ngu, duong_dan_audio)] cua cac file audio trong thu muc."""
    out = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in AUDIO_EXTS:
            path = os.path.join(folder, name)
            out.append((language_name_from_file(path), path))
    return out


@dataclass
class CapCutProject:
    name: str
    folder: str
    modified_us: int
    in_capcut: bool = True      # False: thu muc project trong thu muc goc nhung CapCut chua mo lan nao (T-031)
    capcut_root: bool = True    # nam trong thu muc goc CapCut tu nhan (False: chi trong thu muc app nho)


# T-031: project vua tai / chep vao thu muc CapCut Drafts ma chua mo bang CapCut thi chua co trong root_meta_info.json
# -> quet 1 CAP cac thu muc goc: cha cua moi project trong index + thu muc draft mac dinh trong cai dat CapCut + thu muc
# cha cac project nguoi dung da chon / da chay (app nho, toi da REMEMBERED_ROOTS_MAX). Khong quet o dia / doan thu muc khac.
REMEMBERED_ROOTS_MAX = 10
SKIP_DIR_SUFFIXES = (".7z", ".rar", ".zip", ".tmp", ".bak")
# Moi thu muc goc quet toi da SCAN_MAX_ENTRIES muc / SCAN_MAX_SEC giay (khong treo giao dien). Thu muc app nho nam tren
# Google Drive (o ao "Google Drive" / duong dan shortcut) -> khong quet (liet ke cham, co the tai file ve).
SCAN_MAX_ENTRIES = 2000
SCAN_MAX_SEC = 1.0
CLOUD_MARKERS = (".shortcut-targets-by-id", "my drive", "drive của tôi", "shared drives", "bộ nhớ dùng chung")


def capcut_index_path() -> str:
    """CapCut luu danh sach project (ten + thu muc that) o day, ke ca khi
    thu muc chua project nam o o dia khac."""
    return os.path.join(os.environ.get("LOCALAPPDATA", ""), "CapCut", "User Data", "Projects",
                        "com.lveditor.draft", "root_meta_info.json")


def _key(folder: str) -> str:
    return os.path.normcase(os.path.normpath(folder))


def _capcut_index() -> list:
    """[(item, thu muc)] trong root_meta_info.json cua CapCut (ke ca project da xoa). Loi doc -> []."""
    try:
        with open(capcut_index_path(), encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    return [(item, os.path.normpath(item.get("draft_fold_path", ""))) for item in data.get("all_draft_store", [])
            if item.get("draft_fold_path")]


def capcut_default_drafts_dir() -> str:
    """Thu muc luu draft trong cai dat CapCut (globalSetting: currentCustomDraftPath); khong doc duoc -> ""."""
    path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "CapCut", "User Data", "Config", "globalSetting")
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("currentCustomDraftPath="):
                    value = line.split("=", 1)[1].strip().strip('"').replace("\\\\", "\\")
                    return os.path.normpath(value) if value else ""
    except OSError:
        pass
    return ""


def capcut_roots() -> list:
    """Thu muc goc CapCut tu nhan project: cha cua moi project trong index + thu muc draft mac dinh."""
    roots = {}
    for _, folder in _capcut_index():
        parent = os.path.dirname(folder)
        roots.setdefault(_key(parent), parent)
    default = capcut_default_drafts_dir()
    if default:
        roots.setdefault(_key(default), default)
    return list(roots.values())


def _on_google_drive(path: str) -> bool:
    """Thu muc nam tren Google Drive for desktop: duong dan co dau hieu Drive, hoac o dia ten "Google Drive"."""
    low = os.path.normpath(path).lower()
    if any(m in low for m in CLOUD_MARKERS):
        return True
    drive = os.path.splitdrive(os.path.abspath(path))[0]
    if os.name != "nt" or not drive or drive.startswith("\\\\"):
        return False
    name = ctypes.create_unicode_buffer(261)
    try:
        ok = ctypes.windll.kernel32.GetVolumeInformationW(drive + "\\", name, 261, None, None, None, None, 0)
    except (AttributeError, OSError):
        return False
    return bool(ok) and "google drive" in name.value.lower()


def remember_root(remembered: list, drafts_dir: str) -> list:
    """Them thu muc cha cua project vao danh sach thu muc da nho (moi nhat dau, toi da REMEMBERED_ROOTS_MAX)."""
    parent = os.path.dirname(os.path.normpath(drafts_dir))
    rest = [r for r in remembered or [] if r and _key(r) != _key(parent)]
    return ([parent] + rest)[:REMEMBERED_ROOTS_MAX]


def list_capcut_projects(remembered=()) -> list:
    """Project CapCut dang quan ly (index) + project trong cac thu muc goc chua mo bang CapCut (in_capcut=False, quet
    1 cap, ten = ten thu muc, thoi gian = mtime draft_content.json - khong doc noi dung). Moi sua gan nhat truoc."""
    index = _capcut_index()
    known = {_key(folder) for _, folder in index}          # ke ca project da xoa trong CapCut: khong hien lai
    out = []
    for item, folder in index:
        if item.get("tm_draft_removed") or not os.path.isfile(os.path.join(folder, DRAFT_FILE)):
            continue
        out.append(CapCutProject(item.get("draft_name") or os.path.basename(folder), folder,
                                 int(item.get("tm_draft_modified") or 0)))
    cc = {_key(r) for r in capcut_roots()}
    roots = {_key(r): r for r in capcut_roots() + [r for r in remembered or () if r]}
    for rk, root in roots.items():
        if is_network_path(root) or (rk not in cc and _on_google_drive(root)):
            continue
        try:
            names = os.listdir(root)[:SCAN_MAX_ENTRIES]
        except OSError:
            continue
        deadline = time.monotonic() + SCAN_MAX_SEC
        for name in names:
            if time.monotonic() > deadline:
                break
            folder = os.path.join(root, name)
            if name.startswith((".", "~")) or name.lower().endswith(SKIP_DIR_SUFFIXES) or _key(folder) in known:
                continue
            draft = os.path.join(folder, DRAFT_FILE)
            try:
                mtime = os.path.getmtime(draft)
            except OSError:              # khong phai thu muc / khong co draft_content.json
                continue
            known.add(_key(folder))
            out.append(CapCutProject(name, os.path.normpath(folder), int(mtime * US), in_capcut=False,
                                     capcut_root=rk in cc))
    return sorted(out, key=lambda p: p.modified_us, reverse=True)


def capcut_status(drafts_dir: str) -> str:
    """"index": project CapCut dang quan ly; "root": nam trong thu muc goc cua CapCut nhung chua mo (CapCut tu nhan khi
    mo - T-031, khong chan chay); "removed": da XOA trong CapCut (thu muc con) -> CapCut khong hien, canh bao nhu
    "outside"; "outside": ngoai cac thu muc do -> CapCut se khong doc thay doi ghi vao day (loi that da gap: sua nham ban
    copy)."""
    target = _key(drafts_dir)
    removed = False
    for item, folder in _capcut_index():
        if _key(folder) == target:
            if not item.get("tm_draft_removed"):
                return "index"
            removed = True
    if removed:
        return "removed"
    parent = _key(os.path.dirname(os.path.normpath(drafts_dir)))
    return "root" if any(_key(r) == parent for r in capcut_roots()) else "outside"


def is_registered_in_capcut(drafts_dir: str) -> bool:
    """Project CapCut se thay thay doi (dang quan ly, hoac trong thu muc goc CapCut tu nhan - T-031)."""
    return capcut_status(drafts_dir) in ("index", "root")


def is_capcut_running() -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq CapCut.exe", "/NH"],
            capture_output=True, text=True, errors="replace", timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return "capcut.exe" in result.stdout.lower()
    except Exception:
        return False


def format_us(us: int) -> str:
    s = us // US
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
