@echo off
echo Dang khoi dong 2 app Streamlit va Nexa NPU Server...

:: Đảm bảo đang ở đúng thư mục chứa code
cd /d "C:\Users\local_account\Desktop\STT---Hoi-cung"

:: Thiết lập đường dẫn tuyệt đối đến file activate.bat của Anaconda
set CONDA_ACTIVATE="C:\Users\local_account\anaconda3\Scripts\activate.bat"

:: 1. Mở cửa sổ chạy App 1 trên môi trường AI_hub
start "App 1 - AI_hub" cmd /k "%CONDA_ACTIVATE% AI_hub && streamlit run app.py --server.port 8501"

:: 2. Mở cửa sổ chạy App 2 trên môi trường env_live
start "App 2 - env_live" cmd /k "%CONDA_ACTIVATE% env_live && streamlit run app.py --server.port 8502"

:: 3. Mở cửa sổ chạy Nexa NPU Server trên môi trường env_live
:: LƯU Ý: Hãy thay thế đoạn token bên dưới bằng token đầy đủ của bạn để tránh lỗi copy thiếu chữ
start "Nexa NPU Server" cmd /k "%CONDA_ACTIVATE% env_live && set "NEXA_TOKEN=key/eyJhY2NvdW50Ijp7ImlkIjoiNDI1Y2JiNWQtNjk1NC00NDYxLWJiOWMtYzhlZjBiY2JlYzA2In0sInByb2R1Y3QiOnsiaWQiOiJkYjI4ZTNmYy1mMjU4LTQ4ZTctYmNkYi0wZmE4YjRkYTJhNWYifSwicG9saWN5Ijp7ImlkIjoiMmYyOWQyMjctNDVkZS00MzQ3LTg0YTItMjUwNTYwMmEzYzMyIiwiZHVyYXRpb24iOjMxMTA0MDAwMH0sInVzZXIiOnsiaWQiOiI3MGE2YzA4NS1jYjc3LTQ3YmEtOWUxNC1lNjFjYTA2ZThmZjUiLCJlbWFpbCI6ImFsYW40QG5leGE0YWkuY29tIn0sImxpY2Vuc2UiOnsiaWQiOiI4OTlhZGQ2NS1lOTI2LTQ2M2ItODllNi0xMjc0NzM3ZjA1MzYiLCJjcmVhdGVkIjoiMjAyNS0wOS0wNlQwMDo1MzozNi4yMDNaIiwiZXhwaXJ5IjoiMjAzNS0xMi0zMVQyMzo1OTo1OS4wMDBaIn19.BXoUHIEzFMuuZbBT7RvsKO9nTi5950C6kHO64blF7XBnfKvZ6ClA8a55tmszI1ZWdngzpNFTzMM5PV5euuzMCA==" && nexa serve NexaAI/Qwen3-4B-Instruct-2507-npu --host 127.0.0.1:18183"