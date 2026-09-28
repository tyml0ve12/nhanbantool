"""Ghi ket qua long tieng vao draft CapCut.

- Moi ngon ngu = 1 audio track moi, dat ten theo ngon ngu, N segment dung
  CHUNG 1 material (giong cach CapCut tu lam voi track goc).
- Moi segment moi co du 6 extra_material_refs (speed, placeholder_info, beats,
  sound_channel_mapping, loudness, vocal_separation) nhan ban tu segment goc, doi id.
- Track goc: volume = 0 tung segment (giu last_nonzero_volume de bat lai), KHONG xoa.
- Chi 1 ngon ngu duoc bat tieng, cac ngon ngu con lai volume = 0.
- Khong bao gio ghi de file goc o day: ghi ra draft_content.dubbed.json;
  apply_to_project() moi thay file goc (sau khi backup).
"""
import copy
import glob
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime

from core.capcut_project import ProjectError, DRAFT_FILE, US

DUBBED_FILE = "draft_content.dubbed.json"
BACKUP_PREFIX = "draft_content.backup_"


@dataclass
class OriginalSentence:
    start_us: int
    duration_us: int


@dataclass
class LanguageTrack:
    name: str
    audio_path: str
    audio_duration_us: int
    placements: list     # [(target_start_us, source_start_us, duration_us)]
    audible: bool


def _new_id() -> str:
    return str(uuid.uuid4()).upper()


def load_draft(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def original_sentences(draft: dict, track_index: int) -> list:
    """Danh sach cau goc (vi tri tren timeline) cua track thoai, sap theo thoi gian."""
    try:
        track = draft["tracks"][track_index]
        segs = track["segments"]
        _ = [s["target_timerange"]["start"] for s in segs]
    except (KeyError, IndexError, TypeError):
        raise ProjectError("Không đọc được track thoại gốc trong draft (cấu trúc lạ).")
    if track.get("type") != "audio":
        raise ProjectError("Track đã chọn không phải track audio.")
    segs = sorted(segs, key=lambda s: s["target_timerange"]["start"])
    return [OriginalSentence(int(s["target_timerange"]["start"]), int(s["target_timerange"]["duration"]))
            for s in segs]


def _index_materials(materials: dict) -> dict:
    """id -> (ten mang trong materials, entry)."""
    out = {}
    for key, value in materials.items():
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict) and "id" in entry:
                    out[entry["id"]] = (key, entry)
    return out


def build_dubbed_draft(draft: dict, track_index: int, languages: list) -> dict:
    """Tra ve 1 ban sao draft da them track cac ngon ngu + mute track goc."""
    d = copy.deepcopy(draft)
    materials = d["materials"]
    orig_track = d["tracks"][track_index]
    _remove_previous_dub_tracks(d, orig_track, {lang.name for lang in languages})
    template_seg = sorted(orig_track["segments"], key=lambda s: s["target_timerange"]["start"])[0]
    index = _index_materials(materials)
    if template_seg.get("material_id") not in index:
        raise ProjectError("Không tìm thấy material của track gốc (cấu trúc draft lạ).")
    template_material = index[template_seg["material_id"]][1]

    # 1. Mute track goc
    for seg in orig_track["segments"]:
        vol = float(seg.get("volume", 1.0))
        if vol > 0:
            seg["last_nonzero_volume"] = vol
        seg["volume"] = 0.0

    # 2. Them 1 track cho moi ngon ngu
    for lang in languages:
        material = copy.deepcopy(template_material)
        material["id"] = _new_id()
        material["unique_id"] = uuid.uuid4().hex
        material["name"] = os.path.basename(lang.audio_path)
        material["path"] = lang.audio_path.replace("\\", "/")
        material["duration"] = int(lang.audio_duration_us)
        material["wave_points"] = []
        materials.setdefault("audios", []).append(material)

        track_pos = len(d["tracks"])
        new_track = {k: copy.deepcopy(v) for k, v in orig_track.items() if k != "segments"}
        new_track["id"] = _new_id()
        new_track["name"] = lang.name
        new_track["is_default_name"] = False
        new_track["segments"] = []

        volume = 1.0 if lang.audible else 0.0
        for target_us, source_us, dur_us in lang.placements:
            seg = copy.deepcopy(template_seg)
            seg["id"] = _new_id()
            seg["material_id"] = material["id"]
            seg["source_timerange"] = {"start": int(source_us), "duration": int(dur_us)}
            seg["target_timerange"] = {"start": int(target_us), "duration": int(dur_us)}
            seg["volume"] = volume
            seg["last_nonzero_volume"] = 1.0
            seg["speed"] = 1.0
            if "track_render_index" in seg:
                seg["track_render_index"] = track_pos
            seg["extra_material_refs"] = _clone_refs(template_seg.get("extra_material_refs", []),
                                                     index, materials)
            new_track["segments"].append(seg)
        d["tracks"].append(new_track)
    return d


def _remove_previous_dub_tracks(d: dict, orig_track: dict, names: set):
    """Chay lai tren cung project: xoa track long tieng cu cung ten (do lan
    chay truoc tao) thay vi cong don track moi, va don material chi track do dung."""
    old = [t for t in d["tracks"]
           if t is not orig_track and t.get("type") == "audio" and t.get("name") in names]
    if not old:
        return
    d["tracks"] = [t for t in d["tracks"] if not any(t is o for o in old)]
    still_used = set()
    for t in d["tracks"]:
        for s in t.get("segments", []):
            still_used.add(s.get("material_id"))
            still_used.update(s.get("extra_material_refs", []))
    drop = set()
    for t in old:
        for s in t.get("segments", []):
            drop.add(s.get("material_id"))
            drop.update(s.get("extra_material_refs", []))
    drop -= still_used
    for key, value in d["materials"].items():
        if isinstance(value, list):
            d["materials"][key] = [e for e in value if not (isinstance(e, dict) and e.get("id") in drop)]


def _clone_refs(ref_ids, index, materials) -> list:
    new_ids = []
    for ref in ref_ids:
        if ref not in index:
            continue
        key, entry = index[ref]
        clone = copy.deepcopy(entry)
        clone["id"] = _new_id()
        materials[key].append(clone)
        new_ids.append(clone["id"])
    return new_ids


def validate_dubbed(d: dict, orig_track_id: str, languages: list):
    """Kiem tra lai ket qua truoc khi ghi: dung so segment, khong chong nhau,
    track goc da mute, khong con track trung ten. Sai -> nem loi, khong ghi."""
    orig = next((t for t in d["tracks"] if t.get("id") == orig_track_id), None)
    if orig is None:
        raise ProjectError("Kiểm tra thất bại: mất track gốc.")
    for lang in languages:
        if sum(1 for t in d["tracks"] if t.get("name") == lang.name) != 1:
            raise ProjectError(f"Kiểm tra thất bại: có nhiều track tên {lang.name}.")
    for seg in orig["segments"]:
        if seg["volume"] != 0.0:
            raise ProjectError("Kiểm tra thất bại: track gốc chưa được tắt tiếng hết.")
    new_tracks = d["tracks"][-len(languages):] if languages else []
    for lang, track in zip(languages, new_tracks):
        segs = sorted(track["segments"], key=lambda s: s["target_timerange"]["start"])
        if len(segs) != len(lang.placements):
            raise ProjectError(f"Kiểm tra thất bại: track {lang.name} sai số segment.")
        for a, b in zip(segs, segs[1:]):
            if a["target_timerange"]["start"] + a["target_timerange"]["duration"] > b["target_timerange"]["start"]:
                raise ProjectError(f"Kiểm tra thất bại: track {lang.name} có 2 câu chồng nhau.")


def write_dubbed(draft_files: list, dubbed: dict) -> list:
    """Ghi draft_content.dubbed.json canh MOI file draft. Tra ve cac duong dan da ghi."""
    written = []
    text = json.dumps(dubbed, ensure_ascii=False, separators=(",", ":"))
    for path in draft_files:
        out = os.path.join(os.path.dirname(path), DUBBED_FILE)
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, out)
        written.append(out)
    return written


def apply_to_project(draft_files: list) -> str:
    """Backup tung draft_content.json roi thay bang ban dubbed canh no.
    Tra ve nhan thoi gian cua ban backup."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pairs = []
    for path in draft_files:
        dubbed = os.path.join(os.path.dirname(path), DUBBED_FILE)
        if not os.path.isfile(dubbed):
            raise ProjectError(f"Chưa có file kết quả {DUBBED_FILE} cạnh {path}. Hãy chạy lại.")
        pairs.append((path, dubbed))
    for path, _ in pairs:
        shutil.copy2(path, os.path.join(os.path.dirname(path), f"{BACKUP_PREFIX}{stamp}.json"))
    for path, dubbed in pairs:
        shutil.copyfile(dubbed, path)
    return stamp


def list_backups(draft_files: list) -> list:
    """Cac nhan thoi gian backup co du cho MOI file draft, moi nhat truoc."""
    stamps = None
    for path in draft_files:
        found = {os.path.basename(p)[len(BACKUP_PREFIX):-5]
                 for p in glob.glob(os.path.join(os.path.dirname(path), f"{BACKUP_PREFIX}*.json"))}
        stamps = found if stamps is None else stamps & found
    return sorted(stamps or [], reverse=True)


def restore_backup(draft_files: list, stamp: str):
    for path in draft_files:
        backup = os.path.join(os.path.dirname(path), f"{BACKUP_PREFIX}{stamp}.json")
        if not os.path.isfile(backup):
            raise ProjectError(f"Thiếu bản backup {stamp} cạnh {path}.")
    for path in draft_files:
        shutil.copyfile(os.path.join(os.path.dirname(path), f"{BACKUP_PREFIX}{stamp}.json"), path)


def seconds_to_us(sec: float) -> int:
    return int(round(sec * US))
