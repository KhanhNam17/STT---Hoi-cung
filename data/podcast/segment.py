import re
import os
import pandas as pd
from pydub import AudioSegment

# 1. CẤU HÌNH ĐƯỜNG DẪN TỚI FILE
txt_file = r"D:\FPT\semester 6 - ttap\New folder\whisper\Whisper Large V3 Turbo\test case scenerio\data\podcast\Hà Anh Tuấn_ Chuyện ép mình đi hát và 'chiến lược' trong âm nhạc _ Have A Sip EP01 (320 kbps).mp3.txt"
audio_file = "podcast_HAT_16k.wav"
output_dir = "benchmark_dataset" # Thư mục chứa các file .wav sau khi cắt

os.makedirs(output_dir, exist_ok=True)

# 2. ĐỌC FILE TXT GỐC VÀ TRÍCH XUẤT THỜI GIAN
with open(txt_file, 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d+)\]\s*-\s*(Diễn giả \d+)'
matches = list(re.finditer(pattern, content))

data = []
for i in range(len(matches)):
    match = matches[i]
    time_str = match.group(1)
    speaker = match.group(2)
    
    # Tính thời gian bắt đầu bằng mili-giây
    h, m, s = time_str.split(':')
    sec, ms = s.split('.')
    start_ms = (int(h) * 3600 + int(m) * 60 + int(sec)) * 1000 + int(ms)
    
    # Lấy văn bản
    text_start = match.end()
    text_end = matches[i+1].start() if i + 1 < len(matches) else len(content)
    sentence = content[text_start:text_end].strip()
    sentence = " ".join(sentence.split())
    
    data.append({
        'speaker': speaker,
        'start_ms': start_ms,
        'sentence': sentence
    })

# 3. CẮT AUDIO VÀ TẠO METADATA VỚI ĐẦY ĐỦ CÁC CỘT
print("Đang nạp file audio gốc (có thể mất vài chục giây)...")
audio = AudioSegment.from_wav(audio_file)

benchmark_data = []

print("Đang tiến hành cắt file và xây dựng metadata.csv chuẩn...")
for i in range(len(data)):
    start_ms = data[i]['start_ms']
    sentence = data[i]['sentence']
    speaker_raw = data[i]['speaker']
    
    if not sentence:
        continue
        
    if i + 1 < len(data):
        end_ms = data[i+1]['start_ms']
    else:
        end_ms = start_ms + 20000
        
    if i == 0 and start_ms > 0:
        print(f"Bỏ qua đoạn intro dài {start_ms/1000}s đầu tiên.")

    # Tạo ID và đường dẫn file
    segment_id = f"segment_{i:03d}"
    filename = f"{segment_id}.wav"
    
    # Dùng dấu gạch chéo xuôi (/) để transcriber.py đọc được trên mọi HĐH
    file_path = f"{output_dir}/{filename}"

    # Cắt và lưu audio
    chunk = audio[start_ms:end_ms]
    chunk.export(os.path.join(output_dir, filename), format="wav")
    
    # Tính toán thời lượng (đơn vị: Giây)
    start_sec = start_ms / 1000.0
    end_sec = end_ms / 1000.0
    duration_sec = end_sec - start_sec

    # Phân loại Speaker chuẩn xác
    if "1" in speaker_raw:
        speaker_id = "SPEAKER_00"
        speaker_name = "Host Thuỳ Minh"
    else:
        speaker_id = "SPEAKER_01"
        speaker_name = "Ca sĩ Hà Anh Tuấn"

    # THÊM ĐẦY ĐỦ 14 CỘT THEO ĐÚNG CHUẨN YÊU CẦU
    benchmark_data.append({
        'id': segment_id,
        'file_path': file_path,
        'wav_path': file_path,
        'sentence': sentence,
        'speaker_id': speaker_id,
        'speaker_name': speaker_name,
        'duration': round(duration_sec, 3),
        'start': round(start_sec, 3),
        'end': round(end_sec, 3),
        'sample_rate': 16000,
        'session_type': 'podcast',
        'dataset': 'PODCAST',
        'split': 'test',
        'wer_target': 0.15
    })

# Xuất file CSV
df_benchmark = pd.DataFrame(benchmark_data)
csv_path = os.path.join(output_dir, 'metadata.csv')
df_benchmark.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"\n✅ HOÀN TẤT! Đã tạo thành công {len(benchmark_data)} file wav.")
print(f"👉 File metadata.csv đã được lưu tại: {csv_path}")
print("Bây giờ bạn chạy lại transcriber.py, model sẽ đọc được file bình thường!")