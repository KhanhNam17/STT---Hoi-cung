import wave
import numpy as np
import pandas as pd
from datasets import load_dataset, Audio
from pathlib import Path
from tqdm import tqdm 
import os

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = "1"

def save_wav(audio_array: np.ndarray, sample_rate: int, output_path: str):
    audio = np.array(audio_array, dtype=np.float32)

    audio = np.clip(audio, -1.0, 1.0)

    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

def download_common_voice(
        output_dir: str = 'data/raw',
        split: str = "test",
        max_samples: int = 100,
) -> pd.DataFrame:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"⌛ Loading VIVOS dataset (split={split})...")
    dataset = load_dataset(
        "vivos",
        split = split,
        trust_remote_code=True
    )
    
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    total = min(max_samples, len(dataset))
    print(f"Tổng dataset: {len(dataset)} | Sẽ lấy: {total}")

    records = []
    failed = []

    for i in tqdm(range(total), desc = "Saving audio"):
        sample      = dataset[i]
        audio_array = np.array(sample["audio"]["array"])
        sample_rate = sample["audio"]["sampling_rate"]  # đã là 16000
 
        file_id  = f"vi_{i:04d}"
        out_path = Path(output_dir) / f"{file_id}.wav"
 
        try:
            save_wav(audio_array, sample_rate, str(out_path))
 
            # Verify ngay sau khi lưu
            with wave.open(str(out_path), "rb") as wf:
                assert wf.getnframes() > 0, "Empty WAV"
 
            records.append({
                "file_id"    : file_id,
                "file_path"  : str(out_path),
                "wav_path"   : str(out_path),
                "sentence"   : sample["sentence"],
                "speaker_id" : sample.get("speaker_id", ""),
                "sample_rate": sample_rate,
                "duration"   : round(len(audio_array) / sample_rate, 3),
                "dataset"    : "VIVOS",
                "split"      : split,
            })
 
        except Exception as e:
            print(f"\n⚠️  Lỗi {file_id}: {e}")
            failed.append(file_id)
 
    df = pd.DataFrame(records)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/metadata.csv", index=False, encoding="utf-8")
 
    print(f"\n✅ Đã lưu {len(records)}/{total} files → {output_dir}/")
    if failed:
        print(f"⚠️  Failed: {failed}")
    print(f"   Duration TB     : {df['duration'].mean():.2f}s")
    print(f"   Tổng thời lượng : {df['duration'].sum()/60:.1f} phút")
    return df
 
 
if __name__ == "__main__":
    # Xóa data cũ bị corrupt trước khi chạy lại
    import shutil
    for old in ["data/raw", "data/processed"]:
        if Path(old).exists():
            shutil.rmtree(old)
            print(f"🗑️  Đã xóa {old} cũ")
    if Path("data/metadata.csv").exists():
        Path("data/metadata.csv").unlink()
        print("🗑️  Đã xóa metadata.csv cũ")

    df = download_common_voice(
        output_dir  = "data/raw",
        split       = "test",
        max_samples = 100,
    )
    print("\n📋 Metadata mẫu:")
    print(df[["file_id", "wav_path", "duration", "sentence"]].head(5).to_string())