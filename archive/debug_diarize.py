# debug_diarize2.py
# Chay: python debug_diarize2.py "d2c - 2min_16k.wav"

import sys, json, requests
from pathlib import Path

WAV   = sys.argv[1] if len(sys.argv) > 1 else "test_16k.wav"
URL   = "http://127.0.0.1:18182/v1/audio/diarize"
MODEL = "nexaai/pyannote-npu"
TIMEOUT = 120

def show(label, r):
    print(f"  HTTP {r.status_code}")
    try:
        print(f"  body: {json.dumps(r.json())[:400]}")
    except:
        print(f"  raw : {r.text[:300]}")

abs_path = str(Path(WAV).resolve())

# Cach A: multipart/form-data voi file bytes
print("\n[A] multipart/form-data (file bytes)")
with open(WAV, "rb") as f:
    wav_bytes = f.read()
try:
    r = requests.post(
        URL,
        data    = {"model": MODEL},
        files   = {"audio": ("audio.wav", wav_bytes, "audio/wav")},
        timeout = TIMEOUT,
    )
    show("A", r)
except Exception as e:
    print(f"  ERROR: {e}")

# Cach B: multipart chi files, model trong files
print("\n[B] multipart files only")
try:
    r = requests.post(
        URL,
        files = {
            "model": (None, MODEL),
            "audio": ("audio.wav", open(WAV, "rb"), "audio/wav"),
        },
        timeout = TIMEOUT,
    )
    show("B", r)
except Exception as e:
    print(f"  ERROR: {e}")

# Cach C: file:// path qua JSON (server local)
print("\n[C] JSON file:// path")
try:
    r = requests.post(
        URL,
        json    = {"model": MODEL, "audio": f"file:///{abs_path}"},
        timeout = TIMEOUT,
    )
    show("C", r)
except Exception as e:
    print(f"  ERROR: {e}")

# Cach D: path thuan qua JSON
print("\n[D] JSON path thuan")
try:
    r = requests.post(
        URL,
        json    = {"model": MODEL, "audio": abs_path},
        timeout = TIMEOUT,
    )
    show("D", r)
except Exception as e:
    print(f"  ERROR: {e}")

print("\nDone.")