#!/usr/bin/env python
# scripts/prepare_diart_tuning.py
#
# Chuẩn bị dữ liệu cho diart.tune MÀ KHÔNG cần gán nhãn tay.
#
# Ý tưởng: pyannote OFFLINE (speaker-diarization-3.1) cho kết quả chính xác
# (đã kiểm chứng 63/37 trên file 1977). Ta dùng output của nó làm "ground truth"
# (RTTM) để diart.tune tối ưu tham số online cho KHỚP với pyannote offline.
#
# Tạo ra cấu trúc diart.tune yêu cầu:
#   tune_data/
#     wav/   <stem>.wav     (16kHz mono)
#     rttm/  <stem>.rttm    (ground truth từ pyannote offline)
#
# Cách dùng:
#   python scripts/prepare_diart_tuning.py clip1.wav clip2.mp3 ...
#   → rồi chạy:
#   python -m diart.tune tune_data/wav --reference tune_data/rttm --output tune_data/study
#
# Lưu ý: nên dùng 3–10 clip ngắn (1–3 phút) ĐA DẠNG (nhiều người nói, giọng vùng
# miền khác nhau) để tham số tối ưu tổng quát tốt.

import os
import sys
from pathlib import Path

# Cho phép import package core khi chạy trực tiếp
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

from core.converter import convert_to_wav, get_audio_info
from core.diarizer import load_diarizer, diarize_file


def _write_rttm(segments, uri: str, rttm_path: str) -> None:
    """Ghi list[SpeakerSegment] ra file RTTM chuẩn.
    Format: SPEAKER <uri> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>
    uri PHẢI khớp tên file wav (diart.tune ghép theo stem)."""
    with open(rttm_path, "w", encoding="utf-8") as f:
        for s in segments:
            dur = max(0.0, s.end - s.start)
            if dur <= 0:
                continue
            f.write(
                f"SPEAKER {uri} 1 {s.start:.3f} {dur:.3f} "
                f"<NA> <NA> {s.speaker} <NA> <NA>\n"
            )


def prepare(audio_paths: list[str], out_dir: str = "tune_data") -> None:
    wav_dir  = Path(out_dir) / "wav"
    rttm_dir = Path(out_dir) / "rttm"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)

    print(f"⏳ Load pyannote offline (ground-truth generator)…")
    pipeline = load_diarizer()

    ok = 0
    for ap in audio_paths:
        ap = Path(ap)
        if not ap.exists():
            print(f"  ⚠️  Bỏ qua (không tồn tại): {ap}")
            continue

        stem    = ap.stem
        wav_out = wav_dir / f"{stem}.wav"

        # 1) Convert → 16k mono (idempotent; convert luôn để chuẩn hoá)
        info = get_audio_info(str(ap))
        if info["ok"] and info["sample_rate"] == 16000 and info["channels"] == 1 and ap.suffix.lower() == ".wav":
            # đã chuẩn → copy thẳng
            import shutil
            shutil.copyfile(ap, wav_out)
        else:
            print(f"  🔄 Convert {ap.name} → 16k mono…")
            if not convert_to_wav(str(ap), str(wav_out), sample_rate=16000, normalize=True):
                print(f"  ❌ Convert thất bại: {ap.name}")
                continue

        # 2) Diarize offline → RTTM ground truth
        print(f"  🎙️  Diarize (pyannote offline): {wav_out.name}")
        try:
            segments = diarize_file(pipeline, str(wav_out))
        except Exception as e:
            print(f"  ❌ Diarize lỗi {wav_out.name}: {e}")
            continue

        rttm_out = rttm_dir / f"{stem}.rttm"
        _write_rttm(segments, uri=stem, rttm_path=str(rttm_out))
        n_spk = len({s.speaker for s in segments})
        print(f"  ✅ {stem}: {len(segments)} đoạn, {n_spk} speakers → {rttm_out.name}")
        ok += 1

    print(f"\n✅ Chuẩn bị xong {ok}/{len(audio_paths)} clip vào '{out_dir}/'")
    if ok:
        print("\n── BƯỚC TIẾP THEO: chạy tuner (trong env có diart) ──")
        print("  # Dạng CỐT LÕI (chắc chắn chạy được — theo docs):")
        print(f"  diart.tune {out_dir}/wav --reference {out_dir}/rttm --output {out_dir}/study")
        print("\n  # Xem các cờ tuỳ chọn (segmentation/embedding/số vòng lặp):")
        print("  diart.tune -h")
        print("\n  # (nếu 'diart.tune' không có, dùng: python -m diart.tune ...)")
        print(f"\n  Sau khi xong → đọc tham số tốt nhất:")
        print(f"  python scripts/read_diart_best.py {out_dir}/study")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Dùng: python scripts/prepare_diart_tuning.py <audio1> [audio2 ...]")
        print("  Tạo tune_data/wav + tune_data/rttm để chạy diart.tune.")
        sys.exit(0)
    prepare(sys.argv[1:])
