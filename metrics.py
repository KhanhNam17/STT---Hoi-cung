# File này dùng để đánh giá model qua các metrics
# ĐÃ SỬA: Cập nhật model_name mặc định sang Qualcomm
import pandas as pd
import matplotlib.pyplot as plt
from jiwer import wer, cer
from pathlib import Path

def normalize_text(text: str) -> str:
    import unicodedata, re
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s]", "", text)             # bỏ dấu câu
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in results_df.iterrows():
        gt = normalize_text(str(row['ground_truth']))
        hyp = normalize_text(str(row['transcript']))

        w = wer(gt, hyp) if gt else None
        c = cer(gt, hyp) if gt else None

        rows.append({**row.to_dict(), "wer": w, "cer": c})
    return pd.DataFrame(rows)

def generate_benchmark_report(
        results_csv : str,
        output_dir : str = 'results',
) -> dict:
    """
    Đọc results CSV, tính metrics tổng hợp,
    lưu benchmark_results.csv và vẽ charts.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    df_eval = compute_metrics(df)

    # Tổng hợp theo model
    summary = df_eval.groupby("model").agg(
        avg_wer     = ("wer",     "mean"),
        avg_cer     = ("cer",     "mean"),
        avg_latency = ("latency", "mean"),
        avg_rtf     = ("rtf",     "mean"),
        total_files = ("file_id", "count"),
        wer_std     = ("wer",     "std"),
    ).reset_index()

    summary["avg_wer_pct"] = (summary["avg_wer"] * 100).round(2)
    summary["avg_cer_pct"] = (summary["avg_cer"] * 100).round(2)
    summary["avg_rtf"]     = summary["avg_rtf"].round(4)
    summary["speedup"]     = (1 / summary["avg_rtf"]).round(1)

    # Lưu
    summary.to_csv(f"{output_dir}/benchmark_results.csv", index=False, encoding='utf-8-sig')
    df_eval.to_csv(f"{output_dir}/detailed_results.csv",  index=False, encoding='utf-8-sig')

    # ── Charts ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Màu xanh Qualcomm
    colors = ["#3253DC", "#E74C3C", "#27AE60", "#F39C12"]

    models = summary["model"].tolist()

    # Rút gọn tên model dài cho trục X
    model_labels = [m.replace("qualcomm-", "").replace("-", "\n") for m in models]

    # Chart 1: WER
    b = axes[0].bar(model_labels, summary["avg_wer_pct"], color=colors[:len(models)],
                    edgecolor="white", linewidth=1.5)
    axes[0].set_title("Word Error Rate (%)\nlower is better", fontweight="bold")
    axes[0].set_ylabel("WER (%)")
    axes[0].set_facecolor("#F8F9FA")
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, v in zip(b, summary["avg_wer_pct"]):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f"{v:.1f}%", ha="center", fontweight="bold")

    # Chart 2: Latency
    b = axes[1].bar(model_labels, summary["avg_latency"] * 1000,
                    color=colors[:len(models)], edgecolor="white", linewidth=1.5)
    axes[1].set_title("Average Latency (ms)\nlower is better", fontweight="bold")
    axes[1].set_ylabel("Milliseconds")
    axes[1].set_facecolor("#F8F9FA")
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, v in zip(b, summary["avg_latency"] * 1000):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 10,
                     f"{v:.0f}ms", ha="center", fontweight="bold")

    # Chart 3: Speedup vs Realtime
    b = axes[2].bar(model_labels, summary["speedup"],
                    color=colors[:len(models)], edgecolor="white", linewidth=1.5)
    axes[2].set_title("Speed vs Realtime\nhigher is better", fontweight="bold")
    axes[2].set_ylabel("Times faster than realtime")
    axes[2].axhline(y=1, color="red", linestyle="--", alpha=0.5, label="Realtime threshold")
    axes[2].set_facecolor("#F8F9FA")
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].legend(fontsize=9)
    for bar, v in zip(b, summary["speedup"]):
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.1,
                     f"{v:.1f}x", ha="center", fontweight="bold")

    plt.suptitle(
        "Qualcomm Whisper large-v3-turbo ASR Benchmark — Common Voice 22.0 Vietnamese",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_chart.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY — Qualcomm Whisper large-v3-turbo - Test Case")
    print("=" * 60)
    print(summary[["model", "avg_wer_pct", "avg_cer_pct",
                    "avg_latency", "avg_rtf", "speedup",
                    "total_files"]].to_string(index=False))
    print("=" * 60)

    return summary.to_dict("records")

if __name__ == "__main__":
    generate_benchmark_report(
        results_csv="results/transcripts_qualcomm-whisper-large-v3-turbo.csv",
        output_dir="results"
    )