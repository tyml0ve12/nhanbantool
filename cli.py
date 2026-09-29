"""Chay tu dong lenh (dung chung luong xu ly voi app).

Vi du:
  python cli.py --project "D:\\CapCut\\Video_0412" --audio "Tiếng Anh.mp3" --audio "Tiếng Nhật.mp3" ^
                --video "D:\\Raw\\Video_0412.mp4" --mode both --default "Tiếng Anh"
"""
import argparse
import os
import sys

from PySide6.QtCore import QCoreApplication

from core.capcut_project import ProjectError, load_project, language_name_from_file
from core.parts import audio_files_in
from core.speech import detect_device
from ui.run_worker import RunWorker, RunConfig, LanguageJob, MODE_DRAFT, MODE_EXPORT, MODE_BOTH


def main():
    p = argparse.ArgumentParser(description="Ghép lồng tiếng nhiều ngôn ngữ vào project CapCut / xuất mp4.")
    p.add_argument("--project", required=True, help="Thư mục project CapCut (chứa 'File Drafts')")
    p.add_argument("--audio", action="append", required=True,
                   help="File audio ngôn ngữ mới; lặp lại cho nhiều ngôn ngữ. Tên ngôn ngữ = tên file. "
                        "Ngôn ngữ nhiều phần: trỏ tới THƯ MỤC chứa các phần (tên ngôn ngữ = tên thư mục).")
    p.add_argument("--mode", choices=(MODE_DRAFT, MODE_EXPORT, MODE_BOTH), default=MODE_BOTH)
    p.add_argument("--video", default="", help="Video gốc (.mp4), bắt buộc khi xuất mp4")
    p.add_argument("--out", default="", help="Thư mục xuất mp4 (mặc định: 'Xuất lồng tiếng' cạnh project)")
    p.add_argument("--track", type=int, default=None, help="Chỉ số track thoại gốc (mặc định: tự nhận)")
    p.add_argument("--default", default="", help="Ngôn ngữ bật tiếng trong CapCut (mặc định: ngôn ngữ đầu)")
    p.add_argument("--model", default="large-v3-turbo", help="Model faster-whisper")
    args = p.parse_args()

    try:
        info = load_project(args.project)
    except (ProjectError, OSError) as e:
        sys.exit(f"Lỗi: {e}")
    track = info.main_track if args.track is None else next(
        (t for t in info.audio_tracks if t.index == args.track), None)
    if track is None:
        sys.exit(f"Lỗi: không có track audio số {args.track}.")
    video = args.video or info.video_path_found
    if args.mode != MODE_DRAFT and not os.path.isfile(video):
        sys.exit("Lỗi: chế độ xuất mp4 cần --video trỏ tới video gốc.")
    jobs = []
    for i, a in enumerate(args.audio):
        if os.path.isdir(a):
            found = audio_files_in(a)
            if not found:
                sys.exit(f"Lỗi: thư mục {a} không có file audio")
            jobs.append(LanguageJob(i, os.path.basename(os.path.normpath(a)), [os.path.abspath(p) for p in found]))
        elif os.path.isfile(a):
            jobs.append(LanguageJob(i, language_name_from_file(a), [os.path.abspath(a)]))
        else:
            sys.exit(f"Lỗi: không tìm thấy file audio {a}")
    out = args.out or os.path.join(os.path.dirname(info.drafts_dir), "Xuất lồng tiếng")
    cfg = RunConfig(info.drafts_dir, info.draft_files, track.index, track.segment_count, video, out,
                    args.mode, args.default or jobs[0].name, args.model, detect_device(), jobs)

    app = QCoreApplication(sys.argv)
    worker = RunWorker(cfg)
    worker.log.connect(print)
    result = {}
    worker.finished_all.connect(lambda ok, err, pending: (result.update(ok=ok, err=err), app.quit()))
    worker.start()
    app.exec()
    worker.wait()
    sys.exit(1 if result.get("err") or not result.get("ok") else 0)


if __name__ == "__main__":
    main()
