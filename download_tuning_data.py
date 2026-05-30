import os
import subprocess

# Danh sách clip
clips = [
    {"id": "kaps8lM_BwY", "ss": "00:46:00", "to": "00:50:00", "name": "7nucuoixuan_4p"}, 
    {"id": "O0X2n6T23w8", "ss": "00:15:00", "to": "00:20:00", "name": "sharktank_tap7_5p"}, 
    {"id": "uLiIfZqmneQ", "ss": "00:25:00", "to": "00:30:00", "name": "anhtraivncg_tap6_5p"},
    {"id": "3Y5hUQzv0e8", "ss": "00:57:30", "to": "01:02:30", "name": "anhtraivncg_tap8_5p"},
    {"id": "nyCOypwckKI", "ss": "00:44:00", "to": "00:48:00", "name": "saonhapngu_5p"},
    {"id": "NzWe1MjJ7pU", "ss": "00:20:00", "to": "00:22:30", "name": "sharktank_tap4_5p"},
    {"id": "IMKUD21qf_g", "ss": "00:08:00", "to": "00:13:00", "name": "sharktank_tap2_5p"}
]

def download_and_process(clip):
    url = f"https://www.youtube.com/watch?v={clip['id']}"
    temp_audio = f"temp_{clip['id']}.m4a"
    final_wav = f"{clip['name']}.wav"
    
    print(f"\n>>> Đang tải audio gốc cho: {clip['name']}...")
    
    # Bước 1: Tải toàn bộ audio nén (m4a) - Rất nhanh vì dung lượng nhỏ
    download_cmd = [
        "yt-dlp",
        "-x", "--audio-format", "m4a",
        "--quiet", "--no-warnings",
        "-o", temp_audio,
        url
    ]
    
    # Bước 2: Dùng ffmpeg cắt và convert sang 16k mono cục bộ
    cut_cmd = [
        "ffmpeg", "-y",
        "-i", temp_audio,
        "-ss", clip['ss'],
        "-to", clip['to'],
        "-ar", "16000",
        "-ac", "1",
        final_wav
    ]

    try:
        # Chạy tải
        subprocess.run(download_cmd, check=True)
        # Chạy cắt
        print(f"--- Đang cắt và chuẩn hóa: {final_wav}")
        subprocess.run(cut_cmd, check=True, capture_output=True)
        # Xóa file tạm
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        print(f"✓ Hoàn thành: {final_wav}")
    except Exception as e:
        print(f"✗ Lỗi tại {clip['name']}: {e}")

if __name__ == "__main__":
    if not os.path.exists("tuning_data"):
        os.makedirs("tuning_data")
    os.chdir("tuning_data")
    
    for clip in clips:
        download_and_process(clip)
    
    print("\n======================================")
    print("XONG! Tất cả file đã sẵn sàng trong thư mục 'tuning_data'")