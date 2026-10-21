"""Ghi ket qua long tieng vao draft CapCut.

- Moi ngon ngu = 1 audio track moi, dat ten theo ngon ngu, N segment dung
  CHUNG 1 material (giong cach CapCut tu lam voi track goc).
- Moi segment moi co du 6 extra_material_refs (speed, placeholder_info, beats,
  sound_channel_mapping, loudness, vocal_separation) nhan ban tu segment goc, doi id.
- Track goc: tat tieng bang NUT TAT TIENG TRACK cua CapCut (bit 0 cua
  track["attribute"]), giu nguyen volume tung segment, KHONG xoa.
- Chi 1 ngon ngu duoc bat tieng, cac track ngon ngu con lai cung tat bang nut
  tat tieng track (volume van 1.0 -> bam icon loa trong CapCut la nghe duoc).
- Khong bao gio ghi de file goc o day: ghi ra draft_content.dubbed.json;
  apply_to_project() moi thay file goc (sau khi backup).
"""
import copy
import glob
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from core.capcut_project import ProjectError, DRAFT_FILE, US
from core.paths import long_path_message, path_too_long

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
    placements: list     # [(target_start_us, source_start_us, duration_us, mute_head_us, mute_tail_us)]
    audible: bool
    # Cau THUA khong du cho tren track chinh -> track phu "<ten> – câu thừa", TAT TIENG (chot 2026-10-20)
    extra_placements: list = field(default_factory=list)
    # Vuot am luong tung doan [(ramp_dau_us, ramp_cuoi_us)] song song placements (T-001: mep giua tieng to vuot dai
    # hon); rong -> MUTE_RAMP_US moi mep nhu cu
    ramps: list = field(default_factory=list)


EXTRA_TRACK_SUFFIX = " – câu thừa"


def extra_track_name(language: str) -> str:
    return language + EXTRA_TRACK_SUFFIX


# Keyframe am luong CapCut (KFTypeVolume): cung dinh dang Tool edit nhac da ghi va chay that
# hang tram lan; values = he so tuyen tinh (1.0 = 100%).
# time_offset (micro giay) tinh theo THOI GIAN TRONG FILE AUDIO (material), KHONG phai tu dau doan:
# da xac nhan tren project 132 (2026-10-19) - ghi tu dau doan thi doan cat tu giay 0 cua file nghe duoc,
# moi doan cat tu giua file deu CAM. (Tool edit nhac khong gap vi clip nhac thuong cat tu giay 0.)
KF_VOLUME_TYPE = "KFTypeVolume"
MUTE_RAMP_US = 10_000      # vuot 10ms o cho tat/mo tieng -> khong nghe tieng "tach"


def _volume_keyframe(time_offset_us: int, value: float) -> dict:
    return {"id": _new_id(), "curveType": "Line", "time_offset": int(time_offset_us),
            "left_control": {"x": 0.0, "y": 0.0}, "right_control": {"x": 0.0, "y": 0.0},
            "values": [float(value)], "string_value": "", "graphID": ""}


def mute_keyframes(duration_us: int, mute_head_us: int, mute_tail_us: int, source_start_us: int,
                   ramp_head_us: int = MUTE_RAMP_US, ramp_tail_us: int = MUTE_RAMP_US) -> list:
    """Keyframe am luong cho 1 doan: tat (0) phan keo dai dau/cuoi, 100% o giua. [] neu khong can.
    source_start_us: doan cat tu dau trong file audio - time_offset = source_start_us + vi tri trong doan.
    ramp_*_us > MUTE_RAMP_US: mep nghe duoc nam giua tieng to (giong doc lien, T-001) -> vuot dai hon, ke ca khi
    khong co phan keo dai (0 -> 100% trong ramp tu dau doan)."""
    points = []
    off = duration_us - mute_tail_us
    # Vung nghe duoc ngan hon 2 lan vuot -> vuot ngan lai (moi ben toi da nua vung), giu thu tu diem tang dan
    half = max(0, (off - mute_head_us) // 2)
    ramp_head, ramp_tail = min(ramp_head_us, half), min(ramp_tail_us, half)
    if mute_head_us > 0 or ramp_head_us > MUTE_RAMP_US:
        points += [(0, 0.0), (mute_head_us, 0.0), (mute_head_us + ramp_head, 1.0)]
    if mute_tail_us > 0 or ramp_tail_us > MUTE_RAMP_US:
        points += [(off - ramp_tail, 1.0), (off, 0.0), (duration_us, 0.0)]
    if not points:
        return []
    # Luon co diem o 2 mep doan -> khong phu thuoc cach CapCut ngoai suy truoc diem dau / sau diem cuoi
    if points[0][0] > 0:
        points.insert(0, (0, 1.0))
    if points[-1][0] < duration_us:
        points.append((duration_us, 1.0))
    by_offset = {}
    for t, v in points:
        by_offset[int(t)] = v            # trung vi tri -> diem sau thang
    kfs = [_volume_keyframe(source_start_us + t, v) for t, v in sorted(by_offset.items())]
    return [{"id": _new_id(), "material_id": "", "property_type": KF_VOLUME_TYPE, "keyframe_list": kfs}]


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
    _remove_previous_dub_tracks(d, orig_track, {lang.name for lang in languages}
                                | {extra_track_name(lang.name) for lang in languages})
    template_seg = sorted(orig_track["segments"], key=lambda s: s["target_timerange"]["start"])[0]
    index = _index_materials(materials)
    if template_seg.get("material_id") not in index:
        raise ProjectError("Không tìm thấy material của track gốc (cấu trúc draft lạ).")
    template_material = index[template_seg["material_id"]][1]

    # 1. Tat tieng track goc bang NUT TAT TIENG TRACK cua CapCut (giu nguyen volume
    #    tung doan). Ban cu cua tool keo volume ve 0 -> tra lai muc cu da luu.
    for seg in orig_track["segments"]:
        if float(seg.get("volume", 1.0)) == 0.0 and float(seg.get("last_nonzero_volume", 0) or 0) > 0:
            seg["volume"] = float(seg["last_nonzero_volume"])
    set_track_muted(orig_track, True)

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

        new_track = _new_dub_track(d, orig_track, template_seg, index, materials, material, lang.name, lang.placements,
                                   lang.ramps)
        # Chi 1 ngon ngu bat tieng; cac ngon ngu khac tat bang nut tat tieng track
        set_track_muted(new_track, not lang.audible)
        d["tracks"].append(new_track)
        if lang.extra_placements:
            extra = _new_dub_track(d, orig_track, template_seg, index, materials, material,
                                   extra_track_name(lang.name), lang.extra_placements)
            set_track_muted(extra, True)        # track phu cau thua luon tat tieng
            d["tracks"].append(extra)
    return d


def _new_dub_track(d, orig_track, template_seg, index, materials, material, name, placements, ramps=None) -> dict:
    """1 track long tieng (chua gan vao draft): cac doan cat tu material theo placements (ramps: vuot tung doan)."""
    track_pos = len(d["tracks"])
    new_track = {k: copy.deepcopy(v) for k, v in orig_track.items() if k != "segments"}
    new_track["id"] = _new_id()
    new_track["name"] = name
    new_track["is_default_name"] = False
    new_track["segments"] = []
    for n, placement in enumerate(placements):
        target_us, source_us, dur_us = placement[:3]
        ramp_head, ramp_tail = ramps[n] if ramps else (MUTE_RAMP_US, MUTE_RAMP_US)
        mute_head, mute_tail = (placement[3], placement[4]) if len(placement) >= 5 else (0, 0)
        seg = copy.deepcopy(template_seg)
        seg["id"] = _new_id()
        seg["material_id"] = material["id"]
        seg["source_timerange"] = {"start": int(source_us), "duration": int(dur_us)}
        seg["target_timerange"] = {"start": int(target_us), "duration": int(dur_us)}
        seg["volume"] = 1.0
        seg["last_nonzero_volume"] = 1.0
        seg["speed"] = 1.0
        if "track_render_index" in seg:
            seg["track_render_index"] = track_pos
        # Khong mang theo keyframe / "Ro dan-Mo dan" (audio_fades) cua doan goc lam mau: chung se
        # chong len keyframe tat tieng cua tool (Tool edit nhac da gap: CapCut ap dung ca hai).
        seg["common_keyframes"] = mute_keyframes(int(dur_us), int(mute_head), int(mute_tail), int(source_us),
                                                 int(ramp_head), int(ramp_tail))
        if "keyframe_refs" in seg:
            seg["keyframe_refs"] = []
        seg["extra_material_refs"] = _clone_refs(template_seg.get("extra_material_refs", []),
                                                 index, materials, skip_kinds={"audio_fades"})
        new_track["segments"].append(seg)
    return new_track


# Bit 0 cua track["attribute"] = nut "tat tieng" (icon loa) tren dau track trong
# CapCut. Da doi chieu voi project that: track hien loa gach cheo co attribute=1,
# track binh thuong attribute=0. Moi segment luu ban sao o "track_attribute".
TRACK_MUTE_BIT = 1


def is_track_muted(track: dict) -> bool:
    return bool(int(track.get("attribute", 0) or 0) & TRACK_MUTE_BIT)


def set_track_muted(track: dict, muted: bool):
    attr = int(track.get("attribute", 0) or 0)
    attr = attr | TRACK_MUTE_BIT if muted else attr & ~TRACK_MUTE_BIT
    track["attribute"] = attr
    for seg in track.get("segments", []):
        if "track_attribute" in seg:
            seg["track_attribute"] = attr


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


def _clone_refs(ref_ids, index, materials, skip_kinds=()) -> list:
    new_ids = []
    for ref in ref_ids:
        if ref not in index:
            continue
        key, entry = index[ref]
        if key in skip_kinds:
            continue
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
    if not is_track_muted(orig):
        raise ProjectError("Kiểm tra thất bại: track gốc chưa được tắt tiếng.")
    audible = [t.get("name") for t in d["tracks"]
               if t.get("name") in {lang.name for lang in languages} and not is_track_muted(t)]
    if len(audible) > 1:
        raise ProjectError(f"Kiểm tra thất bại: nhiều track lồng tiếng cùng bật tiếng ({', '.join(audible)}).")
    for lang in languages:
        checks = [(lang.name, lang.placements)]
        if lang.extra_placements:
            checks.append((extra_track_name(lang.name), lang.extra_placements))
        elif any(t.get("name") == extra_track_name(lang.name) for t in d["tracks"]):
            raise ProjectError(f"Kiểm tra thất bại: còn track cũ {extra_track_name(lang.name)}.")
        for name, placements in checks:
            tracks = [t for t in d["tracks"] if t.get("name") == name]
            if len(tracks) != 1:
                raise ProjectError(f"Kiểm tra thất bại: có {len(tracks)} track tên {name}.")
            if name != lang.name and not is_track_muted(tracks[0]):
                raise ProjectError(f"Kiểm tra thất bại: track {name} chưa tắt tiếng.")
            segs = sorted(tracks[0]["segments"], key=lambda s: s["target_timerange"]["start"])
            if len(segs) != len(placements):
                raise ProjectError(f"Kiểm tra thất bại: track {name} sai số segment.")
            for a, b in zip(segs, segs[1:]):
                if a["target_timerange"]["start"] + a["target_timerange"]["duration"] > b["target_timerange"]["start"]:
                    raise ProjectError(f"Kiểm tra thất bại: track {name} có 2 câu chồng nhau.")


def write_dubbed(draft_files: list, dubbed: dict) -> list:
    """Ghi draft_content.dubbed.json canh MOI file draft. Tra ve cac duong dan da ghi."""
    written = []
    text = json.dumps(dubbed, ensure_ascii=False, separators=(",", ":"))
    for path in draft_files:
        out = os.path.join(os.path.dirname(path), DUBBED_FILE)
        tmp = out + ".tmp"
        if path_too_long(tmp):
            raise ProjectError(long_path_message(tmp))
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
        if path_too_long(os.path.join(os.path.dirname(path), f"{BACKUP_PREFIX}{stamp}.json")):
            raise ProjectError(long_path_message(os.path.join(os.path.dirname(path), f"{BACKUP_PREFIX}{stamp}.json")))
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
