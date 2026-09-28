"""Doc (CHI DOC) thong tin project CapCut: vi tri cac file draft_content.json,
cac track audio, video goc. Khong ghi gi vao project o day.

CapCut 6.x luu 2 ban draft_content.json giong het nhau:
  <File Drafts>/draft_content.json
  <File Drafts>/Timelines/<main_timeline_id>/draft_content.json
-> khi ap dung ket qua phai thay CA 2.
"""
import json
import os
import subprocess
from dataclasses import dataclass, field

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
    if video_path_in_draft and os.path.isfile(video_path_in_draft):
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


def capcut_index_path() -> str:
    """CapCut luu danh sach project (ten + thu muc that) o day, ke ca khi
    thu muc chua project nam o o dia khac."""
    return os.path.join(os.environ.get("LOCALAPPDATA", ""), "CapCut", "User Data", "Projects",
                        "com.lveditor.draft", "root_meta_info.json")


def list_capcut_projects() -> list:
    """Cac project CapCut dang quan ly, moi sua gan nhat truoc. Loi doc -> []."""
    try:
        with open(capcut_index_path(), encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    out = []
    for item in data.get("all_draft_store", []):
        folder = os.path.normpath(item.get("draft_fold_path", ""))
        if item.get("tm_draft_removed") or not os.path.isfile(os.path.join(folder, DRAFT_FILE)):
            continue
        out.append(CapCutProject(item.get("draft_name") or os.path.basename(folder), folder,
                                 int(item.get("tm_draft_modified") or 0)))
    return sorted(out, key=lambda p: p.modified_us, reverse=True)


def is_registered_in_capcut(drafts_dir: str) -> bool:
    """Thu muc nay co dung la project CapCut dang mo khong? Neu khong, CapCut se
    khong bao gio doc thay doi ghi vao day (loi that da gap: sua nham ban copy)."""
    target = os.path.normcase(os.path.normpath(drafts_dir))
    return any(os.path.normcase(p.folder) == target for p in list_capcut_projects())


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
