import pandas as pd
import re

def timestamp_to_seconds(ts):
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def main():
    # 1. Đọc file text
    file_path = 'D:/FPT/semester 6 - ttap/New folder/whisper/Whisper Large V3 Turbo/test case scenerio/data/podcast_2/podcast_2.mp3.txt'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{file_path}'. Hãy đảm bảo file này nằm cùng thư mục với script.")
        return

    # 2. Dùng Regex để tách các block thoại
    # Tìm kiếm chuỗi có dạng: [00:00:00.000] - Tên người nói \n Nội dung
    pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*-\s*(.+?)\n(.*?)(?=\[|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        print("❌ Lỗi: Không tìm thấy định dạng thời gian nào hợp lệ trong file.")
        return

    # 3. Xử lý và Gom nhóm dữ liệu
    raw_segments = []
    for i in range(len(matches)):
        start_ts, speaker, text = matches[i]
        start_time = timestamp_to_seconds(start_ts)
        
        # Lấy end_time bằng start_time của câu tiếp theo (hoặc +5s nếu là câu cuối)
        if i + 1 < len(matches):
            end_time = timestamp_to_seconds(matches[i+1][0])
        else:
            end_time = start_time + 5.0 
            
        raw_segments.append({
            "start": start_time,
            "end": end_time,
            "speaker": speaker.strip(),
            "text": text.strip().replace('\n', ' ')
        })

    # THUẬT TOÁN GỘP (Chống phân mảnh nếu cùng 1 người nói liên tục)
    merged_segments = []
    current = raw_segments[0]

    for nxt in raw_segments[1:]:
        # Nếu cùng người nói và khoảng nghỉ nhỏ hơn 2 giây -> Gộp làm 1 dòng
        if current["speaker"] == nxt["speaker"] and (nxt["start"] - current["end"] <= 2.0):
            current["end"] = nxt["end"]
            current["text"] += " " + nxt["text"]
        else:
            merged_segments.append(current)
            current = nxt

    merged_segments.append(current)

    # 4. Định dạng lại thành Cấu trúc CSV chuẩn
    final_data = []
    for seg in merged_segments:
        final_data.append({
            "file_id": "podcast_02",
            "wav_path": "data/podcast/podcast_2.wav",
            "scenario": "Podcast",
            "speaker": seg["speaker"],
            "start_time": round(seg["start"], 3),
            "end_time": round(seg["end"], 3),
            "sentence": seg["text"]
        })

    # 5. Xuất ra file CSV
    df = pd.DataFrame(final_data)
    out_file = 'metadata_podcast_2_v1.csv'
    df.to_csv(out_file, index=False, encoding='utf-8-sig')
    
    print(f"✅ Thành công! Đã tạo file '{out_file}' với {len(df)} lượt lời.")

if __name__ == "__main__":
    main()