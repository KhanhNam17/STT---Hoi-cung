import re
import os 
import sys
import wave
import subprocess
import unicodedata
import numpy as np 
import pandas as pd
from pathlib import Path

transcript_txt = 'data/podcast_2/podcast_2.mp3.txt'
full_wav = 'data/podcast_2/podcast_2.wav'
output_dir = 'data/podcast_2_sliding'

FFMPEG_PATH = r"E:\SOFTWARE\ffmpeg-2026-04-09-git-d3d0b7a5ee-essentials_build\ffmpeg-2026-04-09-git-d3d0b7a5ee-essentials_build\bin\ffmpeg.exe"

STRIP_SILENCE = True
turn_padding_sec = 0.1

def ts_to_sec(timestamp: str) -> float:
    """Convert [HH:MM:SS.mmm] → float seconds."""
    h, m, s = timestamp.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)
 
 
def slugify(name: str) -> str:
    """Convert tên speaker → filename an toàn."""
    # Bỏ dấu tiếng Việt
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    # Thay space và ký tự đặc biệt
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[\s]+', '_', name.strip())
    return name
 
 
def parse_transcript(txt_path: str) -> list[dict]:
    """
    Parse file .txt có format:
    [00:00:00.140] - Speaker Name
    Nội dung...
 
    Trả về list segments với start, end, speaker, text.
    """
    content = open(txt_path, encoding='utf-8').read()
    pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*-\s*(.+?)\n(.*?)(?=\[\d{2}:\d{2}:\d{2}|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
 
    segments = []
    for i, (ts, speaker, text) in enumerate(matches):
        start = ts_to_sec(ts)
        end   = ts_to_sec(matches[i+1][0]) if i+1 < len(matches) else start + 30.0
        text  = text.strip()
        if not text:
            continue
        segments.append({
            'start'   : round(start, 3),
            'end'     : round(end,   3),
            'duration': round(end - start, 3),
            'speaker' : speaker.strip(),
            'text'    : text,
        })
 
    speakers = list(dict.fromkeys(s['speaker'] for s in segments))
    print(f"✅ Parse xong: {len(segments)} turns | {len(speakers)} speakers: {speakers}")
    return segments
 
 
def extract_turn_wav(
    full_wav: str,
    start:    float,
    end:      float,
    tmp_path: str,
    padding:  float = turn_padding_sec,
) -> bool:
    """Cắt 1 turn từ full WAV ra file tạm."""
    adj_start = max(0.0, start - padding)
    duration  = (end + padding) - adj_start
 
    cmd = [
        FFMPEG_PATH, '-y',
        '-i',  full_wav,
        '-ss', str(adj_start),
        '-t',  str(duration),
        '-ar', '16000', '-ac', '1', '-sample_fmt', 's16',
        tmp_path,
        '-loglevel', 'error',
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0
 
 
def concat_wav_files(wav_list: list[str], output_path: str) -> float:
    """
    Concat nhiều WAV files thành 1 file duy nhất bằng Python wave module.
    Trả về tổng duration (giây).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
 
    frames_list = []
    params = None
 
    for wav_path in wav_list:
        with wave.open(wav_path, 'rb') as wf:
            if params is None:
                params = wf.getparams()
            frames_list.append(wf.readframes(wf.getnframes()))
 
    all_frames = b''.join(frames_list)
    total_frames = len(all_frames) // params.sampwidth // params.nchannels
 
    with wave.open(output_path, 'wb') as wout:
        wout.setparams(params)
        wout.writeframes(all_frames)
 
    duration = total_frames / params.framerate
    return round(duration, 3)
 
 
def build_speaker_wav(
    full_wav:    str,
    turns:       list[dict],
    output_wav:  str,
    tmp_dir:     str,
) -> tuple[float, list[dict]]:
    """
    Tạo WAV dài cho 1 speaker bằng cách:
    1. Cắt từng turn ra file tạm
    2. Concat tất cả lại thành 1 file
 
    Trả về (total_duration, turns_with_concat_timestamps)
    """
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    tmp_files = []
 
    for i, turn in enumerate(turns):
        tmp_path = str(Path(tmp_dir) / f"turn_{i:04d}.wav")
        ok = extract_turn_wav(full_wav, turn['start'], turn['end'], tmp_path)
        if ok:
            tmp_files.append((tmp_path, turn))
        else:
            print(f"   ⚠️  Skip turn {i}: extract failed")
 
    if not tmp_files:
        return 0.0, []
 
    # Concat tất cả turns
    wav_paths      = [f[0] for f in tmp_files]
    valid_turns    = [f[1] for f in tmp_files]
    total_duration = concat_wav_files(wav_paths, output_wav)
 
    # Tính lại timestamps trong file concat
    # (dùng để reference, không dùng cho sliding window)
    concat_turns = []
    cursor = 0.0
    for tmp_path, turn in zip(wav_paths, valid_turns):
        with wave.open(tmp_path, 'rb') as wf:
            dur = wf.getnframes() / wf.getframerate()
        concat_turns.append({
            **turn,
            'concat_start': round(cursor, 3),
            'concat_end'  : round(cursor + dur, 3),
            'concat_dur'  : round(dur, 3),
        })
        cursor += dur
 
    # Xóa files tạm
    for p in wav_paths:
        try:
            os.remove(p)
        except:
            pass
 
    return total_duration, concat_turns
 
 
def build_metadata(
    transcript_txt: str,
    full_wav:       str,
    output_dir:     str,
) -> pd.DataFrame:
    """
    Pipeline chính:
    1. Parse transcript
    2. Group turns theo speaker
    3. Tạo WAV dài cho mỗi speaker
    4. Tạo metadata.csv
 
    Format metadata (giống metadata_fixed.csv của Hướng 1):
      id, wav_path, ground_truth, speaker_name, duration,
      start, end, n_turns, session_type, dataset, split, wer_target
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tmp_dir = str(Path(output_dir) / "_tmp")
 
    # ── Step 1: Parse ─────────────────────────────────────────────────────────
    print(f"\n📋 BƯỚC 1: Parse transcript")
    segments = parse_transcript(transcript_txt)
 
    # ── Step 2: Group by speaker ──────────────────────────────────────────────
    print(f"\n👥 BƯỚC 2: Group turns theo speaker")
    speakers = list(dict.fromkeys(s['speaker'] for s in segments))
 
    speaker_groups = {}
    for spk in speakers:
        turns = [s for s in segments if s['speaker'] == spk]
        speaker_groups[spk] = turns
        print(f"   {spk}: {len(turns)} turns | "
              f"total {sum(t['duration'] for t in turns):.1f}s")
 
    # ── Step 3: Tạo WAV cho mỗi speaker ──────────────────────────────────────
    print(f"\n✂️  BƯỚC 3: Concat WAV theo speaker")
 
    if not Path(full_wav).exists():
        print(f"❌ Không tìm thấy file WAV: {full_wav}")
        print(f"   Cần convert file audio trước:")
        print(f"   ffmpeg -i podcast_2.mp3 -ar 16000 -ac 1 -sample_fmt s16 {full_wav}")
        print(f"\n   → Tạo metadata.csv với wav_path để điền sau")
        can_cut_wav = False
    else:
        can_cut_wav = True
 
    records = []
 
    for spk_idx, (spk, turns) in enumerate(speaker_groups.items()):
        spk_slug   = slugify(spk)
        wav_output = str(Path(output_dir) / f"{spk_slug}.wav")
        all_text   = ' '.join(t['text'] for t in turns)
        total_dur  = sum(t['duration'] for t in turns)
 
        print(f"\n   [{spk}]")
        print(f"   WAV output: {wav_output}")
 
        concat_turns = []
        actual_dur   = total_dur  # dùng giá trị ước tính nếu không có WAV
 
        if can_cut_wav:
            actual_dur, concat_turns = build_speaker_wav(
                full_wav, turns, wav_output, tmp_dir
            )
            print(f"   ✅ Tạo xong: {actual_dur:.1f}s "
                  f"({actual_dur/60:.1f} min, "
                  f"{int(np.ceil(actual_dur/30))} sliding windows x 30s)")
        else:
            print(f"   ⚠️  Bỏ qua cắt WAV (chưa có file gốc)")
 
        # Tạo file JSON chứa turn-level timestamps (để tham khảo)
        import json
        turns_json_path = str(Path(output_dir) / f"{spk_slug}_turns.json")
        with open(turns_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'speaker'         : spk,
                'total_duration'  : actual_dur,
                'n_turns'         : len(turns),
                'sliding_windows' : int(np.ceil(actual_dur / 30)),
                'original_turns'  : turns,
                'concat_turns'    : concat_turns,
            }, f, ensure_ascii=False, indent=2)
 
        # ── Tạo record cho metadata ──────────────────────────────────────────
        records.append({
            # ── Các cột giống metadata_fixed.csv của Hướng 1 ──────────────
            'id'           : f"speaker_{spk_idx:02d}",
            'wav_path'     : wav_output,
            'ground_truth' : all_text,
            'speaker_name' : spk,
            'duration'     : actual_dur,
            # start/end: thời gian trong file FULL (để tham khảo)
            'start'        : turns[0]['start'],
            'end'          : turns[-1]['end'],
 
            # ── Cột bổ sung riêng cho Hướng 2 ──────────────────────────────
            'n_turns'          : len(turns),
            'strategy'         : 'sliding_window_30s',
            'n_sliding_windows': int(np.ceil(actual_dur / 30)),
            'turns_json'       : turns_json_path,
            'session_type'     : 'podcast',
            'dataset'          : 'PODCAST_HAVE_A_SIP_EP2',
            'split'            : 'test',
            'wer_target'       : 0.20,  # target cao hơn Hướng 1 vì audio dài hơn
        })
 
    # ── Step 4: Lưu metadata ──────────────────────────────────────────────────
    print(f"\n📄 BƯỚC 4: Lưu metadata")
 
    df = pd.DataFrame(records)
    meta_path = str(Path(output_dir) / 'metadata_sliding.csv')
    df.to_csv(meta_path, index=False, encoding='utf-8-sig')
    print(f"✅ metadata_sliding.csv → {meta_path}")
 
    # Dọn tmp dir
    try:
        import shutil
        if Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir)
    except:
        pass
 
    # ── Báo cáo ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ HOÀN THÀNH — Hướng 2: Sliding Window")
    print(f"{'='*60}")
    print(f"\n📊 Metadata summary:")
    print(df[['id','speaker_name','duration','n_turns',
              'n_sliding_windows','wer_target']].to_string(index=False))
 
    print(f"\n📁 Files tạo ra:")
    for _, row in df.iterrows():
        print(f"   {row['speaker_name']}:")
        print(f"     WAV  : {row['wav_path']}")
        print(f"     JSON : {row['turns_json']}")
    print(f"   Metadata: {meta_path}")
 
    print(f"""
📌 SO SÁNH 2 HƯỚNG:
 
  Hướng 1 (metadata_fixed.csv):
    - {7} rows, mỗi row là 1 segment ngắn (5–50s)
    - WAV: nhiều files nhỏ
    - WER đo per-segment → dễ debug từng đoạn
 
  Hướng 2 (metadata_sliding.csv):
    - {len(df)} rows, mỗi row là 1 speaker toàn bộ (~5 phút)
    - WAV: {len(df)} files dài (1 per speaker)
    - Whisper sliding window 30s tự xử lý
    - WER đo trên toàn bộ text → realistic hơn
    - Context tốt hơn → model dự đoán chính xác hơn
 
📌 BƯỚC TIẾP THEO:
   Sửa transcriber.py:
     metadata_path = "{meta_path}"
   Chạy:
     python transcriber.py
""")
 
    return df
 
 
if __name__ == '__main__':
    df = build_metadata(
        transcript_txt = transcript_txt,
        full_wav       = full_wav,
        output_dir     = output_dir,
    )