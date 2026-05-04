# fix_podcast_metadata.py
# ============================================================
# Cập nhật thuật toán cắt âm thanh và ghép nối kịch bản chuẩn mực.
# Xử lý triệt để:
#   1. Lỗi chia text ngây thơ: Giữ nguyên các đoạn siêu dài để Whisper tự trượt cửa sổ.
#   2. Lỗi file rỗng (0.0s): Xử lý thuật toán bù trừ thời gian (Offset) chuẩn xác.
#   3. Gộp câu thông minh: Nối các thoại ngắn thành file ~20s để tạo ngữ cảnh.
# ============================================================

import sys
import os
import re
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# Cấu hình đường dẫn FFmpeg (Sửa lại cho đúng với máy của bạn nếu cần)
FFMPEG_PATH = r"E:\SOFTWARE\ffmpeg-2026-04-09-git-d3d0b7a5ee-essentials_build\ffmpeg-2026-04-09-git-d3d0b7a5ee-essentials_build\bin\ffmpeg.exe"

# ── Config ────────────────────────────────────────────────────────────────────
# Hãy đảm bảo 2 đường dẫn này trỏ đúng vào file của bạn
PODCAST_FULL_WAV = r"D:\FPT\semester 6 - ttap\New folder\whisper\Whisper Large V3 Turbo\test case scenerio\data\podcast\podcast_HAT_16k.wav"
TRANSCRIPT_TXT   = r"D:\FPT\semester 6 - ttap\New folder\whisper\Whisper Large V3 Turbo\test case scenerio\data\podcast\podcast.txt"

OUTPUT_DIR   = "benchmark_dataset_fixed"

OFFSET_SEC   = -8.39    # Bù trừ thời gian intro
MIN_DURATION = 3.0      # Bỏ qua các đoạn lắt nhắt < 3s (sau khi gộp)
MAX_DURATION = 28.0     # Ngưỡng gộp tối đa (Cảm biến 30s)

# ═════════════════════════════════════════════════════════════════════════════

def parse_transcript_file(txt_path: str) -> list[dict]:
    content = open(txt_path, encoding="utf-8").read()
    pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*-\s*(.+?)\n(.*?)(?=\[|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    segments = []
    for i, (timestamp, speaker, text) in enumerate(matches):
        h, m, s = timestamp.split(':')
        start_sec = int(h) * 3600 + int(m) * 60 + float(s)

        if i + 1 < len(matches):
            next_ts = matches[i+1][0]
            nh, nm, ns = next_ts.split(':')
            end_sec = int(nh) * 3600 + int(nm) * 60 + float(ns)
        else:
            # Cộng 120s để bắt trọn câu chuyện cuối cùng của Hà Anh Tuấn
            end_sec = start_sec + 120.0 

        text = text.strip()
        if not text: continue

        segments.append({
            "start"       : round(start_sec, 3),
            "end"         : round(end_sec, 3),
            "duration"    : round(end_sec - start_sec, 3),
            "speaker_raw" : speaker.strip(),
            "text"        : text,
        })
    return segments

def assign_speaker_ids(segments: list[dict]) -> list[dict]:
    speaker_map = {}
    for seg in segments:
        spk = seg["speaker_raw"]
        if spk not in speaker_map:
            speaker_map[spk] = f"SPEAKER_{len(speaker_map):02d}"

    name_map = {}
    for raw_name, spk_id in speaker_map.items():
        if "1" in raw_name or "minh" in raw_name.lower(): name_map[spk_id] = "Host Thuỳ Minh"
        elif "2" in raw_name or "tuấn" in raw_name.lower(): name_map[spk_id] = "Ca sĩ Hà Anh Tuấn"
        else: name_map[spk_id] = raw_name

    for seg in segments:
        seg["speaker_id"]   = speaker_map[seg["speaker_raw"]]
        seg["speaker_name"] = name_map[speaker_map[seg["speaker_raw"]]]
    return segments

def merge_and_filter(segments: list[dict], min_dur=MIN_DURATION, max_dur=MAX_DURATION) -> list[dict]:
    if not segments: return []
    
    final_segments = []
    cur = dict(segments[0])

    for seg in segments[1:]:
        # Khoảng cách giữa 2 câu nói
        gap = seg["start"] - cur["end"]
        potential_duration = seg["end"] - cur["start"]

        # LUẬT GỘP: Khoảng nghỉ < 3s VÀ tổng thời gian sau gộp <= 28s
        # (Nếu 1 đoạn vốn dĩ đã dài hơn 28s từ file gốc, nó sẽ KHÔNG bị chia nhỏ)
        if gap <= 3.0 and potential_duration <= max_dur:
            cur["end"]      = seg["end"]
            cur["duration"] = round(potential_duration, 3)
            
            # Gộp text (đánh dấu người nói nếu hội thoại đan xen)
            if cur["speaker_id"] != seg["speaker_id"]:
                 cur["text"] = cur["text"] + f" [{seg['speaker_name']}]: " + seg["text"]
            else:
                 cur["text"] = cur["text"] + " " + seg["text"]
        else:
            if cur["duration"] >= min_dur: final_segments.append(cur)
            cur = dict(seg)

    if cur["duration"] >= min_dur: final_segments.append(cur)
    return final_segments

def extract_segment_wav(full_wav: str, start_sec: float, end_sec: float, output_wav: str, offset_sec: float) -> bool:
    # Logic toán học chuẩn xác để tính toán offset (Fix 3)
    adj_start = max(0.0, start_sec + offset_sec)
    adj_end   = end_sec + offset_sec
    duration  = adj_end - adj_start
    
    # GUARD: Loại bỏ các đoạn âm thanh trống (bị trôi khỏi biên sau khi offset)
    if duration <= 0.5:
        print(f"⚠️ Bỏ qua {Path(output_wav).name}: Độ dài chỉ còn {duration:.2f}s do bù trừ offset.")
        return False
        
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        FFMPEG_PATH, "-y", 
        "-i", full_wav, 
        "-ss", str(adj_start), 
        "-t", str(duration), 
        "-ar", "16000", 
        "-ac", "1", 
        "-sample_fmt", "s16", 
        output_wav, 
        "-loglevel", "error"
    ]
    
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0

def main():
    print("🚀 ĐANG KHỞI TẠO DATASET TỐI ƯU")
    segments = parse_transcript_file(TRANSCRIPT_TXT)
    segments = assign_speaker_ids(segments)
    
    # Gộp các đoạn thoại ngắn, GIỮ NGUYÊN các đoạn thoại dài
    segments = merge_and_filter(segments)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    records = []
    
    print("✂️  Đang cắt audio...")
    for i, seg in enumerate(segments):
        seg_id  = f"segment_{i:03d}"
        wav_out = f"{OUTPUT_DIR}/{seg_id}.wav"
        
        ok = extract_segment_wav(PODCAST_FULL_WAV, seg["start"], seg["end"], wav_out, offset_sec=OFFSET_SEC)
        if ok:
            records.append({
                "id": seg_id, 
                "wav_path": wav_out, 
                "ground_truth": seg["text"], # Đổi tên cột chuẩn cho transcriber.py
                "speaker_name": seg["speaker_name"], 
                "duration": seg["duration"],
                "start": seg["start"], 
                "end": seg["end"]
            })

    df = pd.DataFrame(records)
    meta_path = f"{OUTPUT_DIR}/metadata_fixed.csv"
    df.to_csv(meta_path, index=False, encoding="utf-8-sig")

    print(f"✅ Xong! Đã tạo thành công {len(df)} files.")
    print("📊 Phân bố độ dài:")
    for lo, hi in [(0,10), (10,30), (30,60), (60, 150)]:
        n = len(df[(df['duration'] >= lo) & (df['duration'] < hi)])
        print(f"   {lo:3d} - {hi:3d}s : {'█' * n} ({n} files)")

if __name__ == "__main__":
    main()