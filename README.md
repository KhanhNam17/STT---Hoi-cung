# 🎙️ Trợ lý Hỏi Cung — STT + Diarization + DOCX

Hệ thống xử lý giọng nói tiếng Việt chạy **hoàn toàn offline** trên máy tính.  
Nhận file audio/video đầu vào → tự động nhận dạng giọng nói → phân tách người nói → xuất biên bản `.docx` theo mẫu chuẩn Bộ Công an.

---

## Mục lục

- [Tính năng](#tính-năng)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu hình `.env`](#cấu-hình-env)
- [Khởi chạy](#khởi-chạy)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Các module](#các-module)
- [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)

---

## Tính năng

| Tính năng | Mô tả |
|---|---|
| **STT tiếng Việt** | Whisper Large v3 Turbo (Qualcomm AI Hub) — độ chính xác cao |
| **Speaker Diarization** | pyannote/speaker-diarization-community-1 — tự động phân tách người nói |
| **Khôi phục dấu câu** | Rule-based pipeline — thêm `.` `?` `,` cho output thô của Whisper |
| **Alignment** | Ghép text + timestamp theo ranh giới câu (sentence-boundary) |
| **Xuất DOCX** | Điền tự động vào mẫu biên bản hỏi cung (TT 119/2021/TT-BCA) |
| **Offline hoàn toàn** | Không gửi dữ liệu ra ngoài, phù hợp môi trường bảo mật |
| **Chạy trên CPU** | Không cần GPU — chạy được trên máy tính văn phòng thông thường |

**Định dạng file đầu vào được hỗ trợ:** `.mp3` `.wav` `.mp4` `.mkv` `.m4a` `.aac`

---

## Kiến trúc hệ thống

```
File audio/video
       │
       ▼
  converter.py          ← ffmpeg: convert → WAV 16kHz mono
       │
       ├──────────────────────────┐
       ▼                          ▼
 transcriber.py            diarizer.py
 (Whisper v3 Turbo)        (pyannote community-1)
 → raw text                → [SpeakerSegment(speaker, start, end), ...]
       │                          │
       ▼                          │
punctuation_restorer.py           │
 → text có dấu câu                │
       │                          │
       └──────────────────────────┘
                    │
                    ▼
              aligner.py
     → [AlignedTurn(speaker, start, end, text), ...]
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
  transcript_viewer.py    export_docx.py
  (hiển thị Streamlit)    (xuất biên bản .docx)
```

---

## Cấu trúc thư mục

```
project/
│
├── app.py                      # Trang chủ Streamlit
├── pages/
│   └── 1_Batch_Mode.py         # Giao diện xử lý file
│
├── core/                       # Các module xử lý chính
│   ├── __init__.py
│   ├── converter.py            # Tiền xử lý audio (ffmpeg)
│   ├── transcriber.py          # STT — Whisper + PROMPT_PRESETS
│   ├── diarizer.py             # Speaker diarization — pyannote
│   ├── aligner.py              # Ghép text + timestamp
│   └── punctuation_restorer.py # Khôi phục dấu câu
│
├── components/                 # UI components Streamlit
│   ├── transcript_viewer.py    # Hiển thị transcript
│   ├── speaker_editor.py       # Widget gán tên người nói
│   ├── export_docx.py          # Xuất file DOCX
│   └── templates/
│       └── bienbanhoicung.docx # Mẫu biên bản gốc
│
├── data/
│   └── uploads/                # File WAV tạm (tự dọn sau khi xử lý)
│
├── _env                        # File mẫu cấu hình (đổi tên thành .env)
└── README.md
```

---

## Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|---|---|
| **OS** | Windows 10/11 · Ubuntu 20.04+ · macOS 12+ |
| **Python** | 3.10 trở lên |
| **RAM** | 8 GB (khuyến nghị 16 GB) |
| **Ổ cứng** | 10 GB trống (cho model weights) |
| **FFmpeg** | 4.x trở lên |
| **Internet** | Chỉ cần khi tải model lần đầu |

---

## Cài đặt

### Bước 1 — Tải FFmpeg

**Windows:**
1. Tải tại [ffmpeg.org/download.html](https://ffmpeg.org/download.html) → chọn bản essentials build
2. Giải nén, ghi lại đường dẫn tới `ffmpeg.exe` (sẽ điền vào `.env` ở bước sau)

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Bước 2 — Tạo môi trường ảo Python

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Bước 3 — Cài đặt thư viện Python

```bash
pip install streamlit python-dotenv librosa soundfile torch
pip install pyannote.audio
pip install python-docx
pip install "qai_hub_models[whisper-large-v3-turbo]"
```

> **Lưu ý:** `qai_hub_models` sẽ tải weights Whisper (~1.5 GB) trong lần chạy đầu tiên.  
> Cần kết nối internet cho bước này.

### Bước 4 — Cấu hình token HuggingFace

Truy cập [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → tạo token loại **Read**.

Vào trang model và bấm **Agree** để chấp nhận điều khoản:
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

---

## Cấu hình `.env`

Sao chép file `_env` thành `.env` và điền giá trị thực:

```env
# Đường dẫn ffmpeg
# Windows: điền đường dẫn đầy đủ tới ffmpeg.exe
# Linux/macOS: để trống nếu ffmpeg đã có trong PATH
FFMPEG_PATH=C:\ffmpeg\bin\ffmpeg.exe

# HuggingFace token — dùng cho pyannote diarization
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Chế độ Qualcomm
# false = chạy PyTorch FP trên CPU/GPU máy tính (khuyến nghị)
# true  = compile và chạy trên chip Snapdragon thật (cần thiết bị thật)
ON_DEVICE=false

# Tên thiết bị Qualcomm — chỉ dùng khi ON_DEVICE=true
QUALCOMM_DEVICE=Samsung Galaxy S25

# Model diarization (tùy chọn, mặc định = community-1)
# DIARIZATION_MODEL=pyannote/speaker-diarization-3.1

# API key cho cloud diarization precision-2 (tùy chọn)
# PYANNOTE_API_KEY=your_key_from_dashboard.pyannote.ai
```

> ⚠️ **Không commit file `.env` lên git.** Thêm `.env` vào `.gitignore`.

---

## Khởi chạy

```bash
# Kích hoạt môi trường ảo trước
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS

# Chạy app
streamlit run app.py
```

Trình duyệt sẽ tự mở tại `http://localhost:8501`.  
Lần đầu chạy sẽ mất **2–5 phút** để load model Whisper vào bộ nhớ.

---

## Hướng dẫn sử dụng

### Batch Mode — Xử lý file audio/video

1. Mở **📁 Xử lý File Audio/Video** từ sidebar
2. **Tải file lên** — kéo thả hoặc click chọn file (MP3/WAV/MP4/MKV/M4A/AAC)
3. **Cấu hình xử lý:**
   - *Ngôn ngữ:* chọn 🇻🇳 Tiếng Việt
   - *Ngữ cảnh ghi âm:* chọn đúng loại để dấu câu chính xác hơn
   - *Số người nói:* điền số người thực tế (mặc định 2)
   - *Mã phiên:* tên hồ sơ để dùng làm tên file xuất
4. Bấm **▶️ Bắt đầu xử lý** — hệ thống sẽ tự động chạy 4 bước:
   - Bước 1: Convert sang WAV 16kHz
   - Bước 2: Nhận dạng giọng nói + khôi phục dấu câu
   - Bước 3: Phân tách người nói
   - Bước 4: Ghép nối transcript
5. **Gán tên người nói** — đặt tên thực (VD: Điều tra viên, Đối tượng)
6. **Xem và chỉnh sửa** — bấm *📄 Xem toàn bộ* để sửa từng dòng nếu cần
7. **Xuất file** — tải về DOCX (điền vào mẫu biên bản) hoặc TXT thuần

---

## Các module

### `core/converter.py`
Tiền xử lý file audio bằng FFmpeg. Chuyển đổi bất kỳ định dạng nào sang WAV 16kHz mono — chuẩn đầu vào cho Whisper và pyannote.

### `core/transcriber.py`
Nhận dạng giọng nói bằng Qualcomm Whisper Large v3 Turbo. Chứa `PROMPT_PRESETS` — từ điển các prompt mẫu theo ngữ cảnh để cải thiện dấu câu đầu ra.

### `core/diarizer.py`
Phân tách người nói bằng pyannote. Hỗ trợ 3 model:
- `community-1` (mặc định) — local, chất lượng tốt nhất
- `3.1` — legacy fallback
- `precision-2` — cloud, cần `PYANNOTE_API_KEY`

Pipeline hậu xử lý 4 bước: drop segment ngắn → merge cùng speaker → smooth nhãn sai → drop ghost speaker.

### `core/aligner.py`
Ghép text từ Whisper vào timestamp từ pyannote theo thuật toán **Dynamic Sentence-Boundary Alignment** — chia văn bản tại ranh giới câu thay vì cắt giữa chừng.

### `core/punctuation_restorer.py`
Khôi phục dấu câu cho output thô của Whisper bằng rule-based pipeline:
- Chuẩn hóa khoảng trắng
- Fix lỗi Whisper đặc thù (viết hoa giữa câu, lặp từ)
- Thêm dấu phẩy trước liên từ
- Thêm `?` cho câu hỏi
- Tách câu bị dính liền
- Viết hoa đầu câu

---

## Xử lý lỗi thường gặp

**`ffmpeg error` khi convert**
```
Kiểm tra FFMPEG_PATH trong .env — đường dẫn phải trỏ đúng tới ffmpeg.exe
```

**`HF_TOKEN` lỗi xác thực**
```
Kiểm tra token tại huggingface.co/settings/tokens
Đảm bảo đã Agree điều khoản tại trang model pyannote
```

**Load model Whisper quá chậm**
```
Bình thường — lần đầu tải ~1.5 GB weights, sau đó cache lại
Streamlit cache_resource giữ model trong RAM suốt session
```



**Diarization nhầm người nói**
```
Điền đúng "Số người nói" trong cấu hình — không để tự detect nếu biết trước
Tăng min_duration hoặc merge_gap trong diarize_file() nếu cần
```
