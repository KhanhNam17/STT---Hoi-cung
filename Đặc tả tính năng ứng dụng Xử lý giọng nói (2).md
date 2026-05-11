# Danh sách Tính năng ứng dụng: Xử lý giọng nói (Speech to Text)

# 1. Tổng quan: Speech to Text (STT)

Tính năng STT (Speech-to-Text) là thành phần cốt lõi phục vụ việc số hóa các phiên lấy lời khai. Hệ thống nhận đầu vào là âm thanh (trực tiếp hoặc từ file), chuyển đổi thành văn bản có cấu trúc, gán nhãn người nói và mốc thời gian, từ đó tạo transcript hoàn chỉnh và biên bản phiên làm việc.

Hệ thống chạy hoàn toàn offline trên phần cứng Snapdragon, ưu tiên độ ổn định trong phiên dài (>2 giờ) và hỗ trợ tiếng Việt có dấu với Tiếng Anh là các ngôn ngữ chính.

# 2. Yêu cầu chức năng



### File âm thanh (Batch Mode):
- Định dạng hỗ trợ: .mp3, .m4a, .wav, .ogg, .flac
- Tất cả định dạng được chuẩn hóa về WAV 16kHz mono trước khi xử lý
- Việc chuẩn hóa nhằm đảm bảo hiệu năng và độ ổn định tối ưu

## 2.2 Chuyển đổi giọng nói sang văn bản
- Ngôn ngữ: tiếng Việt (có dấu đầy đủ) và tiếng Anh
- Model: Whisper - tiny/ base cho Live Mode, large-v3-turbo cho Batch Mode và refinement
- Toàn bộ inference chạy local, không gọi bất kỳ API cloud nào

## 2.3 Phân tách người nói
- Tự phân biệt và tách nhiều người nói trong cùng luồng âm thanh thu vào
- **Hỗ trợ** gán nhãn vai trò:
    1. Người hỏi (điều tra viên)
    2. Người trả lời (đối tượng, nhân chứng)

## 2.4 Xuất kết quả
- **Transcript đầy đủ** (`.docx`): Mỗi dòng gồm `[timestamp] [Speaker_X]: nội dung`
- **Biên bản tóm tắt phiên** (`.docx`): tóm tắt nội dung chính theo template `bienbanhoicung.docx`
- Hỗ trợ chỉnh sửa tên speaker trước khi export

# 3. Luồng xử lý (Flow)
Dùng khi xử lý file ghi âm có sẵn, ưu tiên độ chính xác cao nhất.

```
Ghi âm / Tải file
    ↓
Tiền xử lý — convert → 16kHz mono WAV, normalize volume
    ↓
[Song song] Diarization  ←→  STT (Whisper large-v3-turbo)
    ↓
Alignment — ghép segment diarization + STT text theo timestamp
    ↓
Hiển thị transcript với màu theo speaker
    ↓
Người dùng gán tên / chỉnh sửa nội dung
    ↓
Xuất DOCX transcript + Biên bản tóm tắt
```
 
---


```
 
---
# 4. Ràng buộc kỹ thuật

1. Toàn bộ quá trình xử lý STT phải thực hiện **offline (local),** không phụ thuộc Internet
2. Model STT phải hỗ trợ tiếng Việt có dấu, gồm các phương ngữ (chủ yếu là giọng Bắc)
3. Hệ thống ổn định trong các phiên kéo dài (> 2 giờ liên tục). Worker thread phải xử lý exception mà không crash.
4. Ưu tiên QNN runtime cho Whisper inference. Chunk size 6–8s là optimal cho Whisper tiny/base trên NPU.
5. 


# 5. Cấu trúc tổ chức các file

```
stt_app/
│
├── app.py                          ← Entry point: python -m streamlit run app.py
│
├── core/                           ← Business logic, không phụ thuộc Streamlit
│   ├── __init__.py
│   ├── converter.py                ← convert_to_wav(): ffmpeg wrapper, chuẩn hóa 16kHz mono
│   ├── transcriber.py              ← load_model(), transcribe_file(): Whisper inference
│   ├── diarizer.py                 ← load_diarizer(), diarize_file(): pyannote wrapper
│   ├── aligner.py                  ← align_segments(): ghép STT text + diarization theo timestamp
│   └── streaming.py                ← RingBuffer, Chunker, VAD, StreamingEngine, dedup logic
│
├── pages/
│   ├── 1_Batch_Mode.py             ← Upload file → xử lý → xuất DOCX
│
├── components/
│   ├── transcript_viewer.py        ← Render transcript, màu theo speaker
│   ├── speaker_editor.py           ← Gán tên thật cho Speaker_0, Speaker_1, ...
│   └── export_docx.py              ← Đóng gói transcript → file DOCX
│
├── data/
│   ├── uploads/                    ← File audio tạm (tự xoá sau phiên)
│   ├── processed/                  ← WAV đã chuẩn hóa (16kHz mono)
│   └── results/                    ← JSON transcript output
│
├── templates/
│   └── bienbanhoicung.docx         ← Template Word biên bản hỏi cung
│
├── .env                            ← HF_TOKEN, FFMPEG_PATH (không commit git)
├── requirements.txt
└── README.md
```