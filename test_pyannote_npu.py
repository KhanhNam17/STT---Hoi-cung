import numpy as np
import sounddevice as sd
import time
import queue
import requests
import io
import wave
import base64  # Import thư viện mã hóa

API_URL = "http://127.0.0.1:18182/v1/audio/diarize" 

audio_queue = queue.Queue()
SAMPLE_RATE = 16000
CHUNK_SEC = 4.0 
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_SEC)

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

print(f"⏳ Đang chuẩn bị kết nối tới NPU Server tại {API_URL}...")
print("\n🎙️ ĐÃ KẾT NỐI MICROPHONE. HÃY THỬ NÓI CHUYỆN!")
print("Nhấn Ctrl+C để dừng...\n")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_FRAMES, callback=audio_callback):
    try:
        t_cursor = 0.0
        while True:
            chunk = audio_queue.get()
            audio_i16 = (chunk * 32767).astype(np.int16)
            
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_i16.tobytes())
            
            wav_io.seek(0)
            
            # ─── ĐÃ FIX: Mã hóa WAV thành Base64 Data URL ───
            wav_bytes = wav_io.read()
            b64_audio = base64.b64encode(wav_bytes).decode('utf-8')
            data_url = f"data:audio/wav;base64,{b64_audio}"
            
            # Gửi gói tin JSON thay vì Multipart
            payload = {
                "file": data_url, 
                "audio": data_url, # Gửi kèm cả 2 tên key phổ biến để phòng hờ API
                "model": "NexaAI/Pyannote-NPU"
            }
            
            start_time = time.perf_counter()
            try:
                response = requests.post(API_URL, json=payload)
                process_time = time.perf_counter() - start_time
                
                if response.status_code == 200:
                    results = response.json()
                    print(f"\n[{t_cursor:.1f}s - {t_cursor+CHUNK_SEC:.1f}s] (NPU xử lý: {process_time:.3f}s)")
                    
                    # Tạm thời in toàn bộ JSON trả về để kiểm tra cấu trúc
                    print("Kết quả NPU trả về:", results)
                    
                else:
                    print(f"❌ NPU Server báo lỗi {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                print("❌ Lỗi: Không thể kết nối tới NPU Server.")
                
            t_cursor += CHUNK_SEC
            
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng thu âm.")