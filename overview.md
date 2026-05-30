Here's a full recap of the project, structured so you can spot improvement areas.

## 1. What it is

**"Trợ lý Ảo Phân tích Hỏi cung"** — a Vietnamese speech-to-text + speaker-diarization app for transcribing interrogations/meetings into official DOCX records (biên bản). Key constraints: **Vietnamese language**, **offline-capable** (runs locally, designed for Qualcomm Snapdragon NPU), and it must answer **who said what**, then export to a Word template.

## 2. Two operating modes

| | **Batch Mode** ([pages/1_Batch_Mode.py](pages/1_Batch_Mode.py)) | **Live Mode** ([pages/2_Live_Mode.py](pages/2_Live_Mode.py)) |
|---|---|---|
| Input | uploaded file (mp3/mp4/wav…) | microphone (real-time) + file-test |
| STT | Whisper Large V3 Turbo (accurate, offline) | Zipformer streaming (sherpa-onnx, low-latency) |
| Diarization | **offline pyannote 3.1** (default) | **diart** (real-time streaming) |
| Attribution | post-hoc forced alignment | live provisional labels → clean pass on stop |

## 3. The pipeline stages

**Batch:** `convert → transcribe → diarize → align → punctuation → [summary] → DOCX`
**Live:** `mic → Zipformer STT + diart (parallel) → provisional labels → [stop] → final_recluster + align → [summary] → DOCX`

Stage-by-stage with the model used:

1. **Convert** ([core/converter.py](core/converter.py)) — ffmpeg → 16kHz mono s16 + loudness-normalize. Handles any format, stereo→mono.
2. **Transcribe** — Whisper (batch) / Zipformer (live) → Vietnamese text.
3. **Diarize** — who-spoke-when as `SpeakerSegment(speaker, start, end)`. Backends below.
4. **Align** ([core/aligner.py](core/aligner.py)) — `stable-ts` forced alignment gives per-word timestamps → each word assigned to the overlapping speaker → grouped into turns. Falls back to ratio-based split if stable-ts missing.
5. **Punctuation** ([core/punctuation_restorer.py](core/punctuation_restorer.py)) — rule-based VN punctuation/capitalization.
6. **Summary** ([core/test_qwen.py](core/test_qwen.py)) — Qwen on NPU. **Now optional** in both modes.
7. **Export** ([components/export_docx.py](components/export_docx.py)) — fills `bienbanhoicung.docx` template; either summary doc or full speaker-attributed transcript.

## 4. Module map (post-refactor)

```
core/
  diarization/         ← new package (pluggable backends)
    diar_types.py        SpeakerSegment (shared type)
    base.py              Protocol interfaces
    pyannote.py          offline pyannote (community-1/3.1/precision-2)
    nexa.py              Nexa NPU CLI
    npu.py               sherpa-onnx hybrid
    streaming.py         ★ DiartStreamingDiarizer + diarize_file_diart (offline diart)
  transcription/       ← new package
    whisper.py, base.py
  pipeline/            ← new package
    postprocess.py       ★ final_recluster (global re-cluster, wespeaker embedding)
  converter.py, aligner.py, punctuation_restorer.py, transcriber.py, npu_workers.py
components/  speaker_editor, transcript_viewer, export_docx, templates/
scripts/     prepare_diart_tuning.py, read_diart_best.py   ← diart tuning workflow
```

## 5. Diarization backends (the heart of the work)

| Backend | Where | Strength | Weakness |
|---|---|---|---|
| **pyannote 3.1 offline** | Batch (default) | most accurate (global clustering + wespeaker); validated **63/37** on your test | not real-time |
| **diart** | Live | real-time streaming (~0.5–1s latency); **tuned** params (`tau=0.72, rho=0.33, delta=0.94`) | online clustering weaker than offline; needs HF token |
| **Nexa Pyannote-NPU** | Batch (optional) | NPU-accelerated | needs Nexa runtime |

Live uses the **Otter-style two-pass model**: provisional labels stream live, then on stop `final_recluster` re-extracts embeddings over the whole recording and re-clusters globally for consistent IDs.

## 6. Current state — what we fixed this session

- ✅ Refactored into clean `diarization/transcription/pipeline` packages
- ✅ diart real-time integration (custom push `AudioSource`, `attach_hooks`, drain-on-stop)
- ✅ Pinned dependency stack (resolution-too-deep, speechbrain k2, pyannote token/version issues)
- ✅ Stereo→mono everywhere; converter wired into diarizer
- ✅ `final_recluster` upgraded weak `pyannote/embedding` → strong **wespeaker**
- ✅ diart hyperparameter **tuning** workflow (bootstraps RTTM from offline pyannote, optuna search)
- ✅ Summary made optional; DOCX export works without it (both modes)
- ✅ **Partial-flush fix** — trailing in-progress text no longer dropped on stop
- ✅ Terminal debug logging for live ASR/diart

## 7. Where I'd focus improvement (for your analysis)

**Accuracy**
- diart online clustering still trails offline pyannote; the tuning DER (~38%) is inflated by 5-speaker clips — re-tune on 2–3 person interrogation-like audio only.
- The **twin-voice / similar-voice** case is unsolved — consider **voiceprint enrollment** (the Phase 2 we never built): pre-register known officers' voices, match by cosine similarity. Biggest reliability win for recurring interrogations.
- Live speaker labels use **estimated** turn timestamps (`words/2.5`) — switching to word-level timing from a streaming aligner would sharpen attribution.

**Performance**
- Everything runs **CPU/PyTorch FP32** right now (Whisper FP32, pyannote CPU). The whole NPU value-prop (Qualcomm, Zipformer-30m, Qwen-NPU) isn't actually exercised in your current runs. Wiring the real NPU path is the headline perf opportunity.
- diart real-time keeps up but final_recluster is slow on long files (per-window embedding extraction).

**Robustness / architecture**
- Three diarizer implementations + facades — consolidate to the protocol interface fully.
- The Streamlit rerun-every-200ms loop for live is fragile under CPU load; a proper async/websocket transport (diart supports it) would be more robust.
- `aligner` depends on stable-ts (old openai-whisper build) — fragile install.

**Vietnamese-specific**
- No **hot-word/custom-vocabulary biasing** yet (names, legal terms) — would cut domain errors.
- No regional-accent handling (Bắc/Trung/Nam) — fine-tuning Zipformer per region.
- Punctuation is rule-based; a small VN punctuation-restoration model would be more accurate.

**UX**
- Live panel only shows last 6 turns truncated to 180 chars — add a full scrollback.
- One-tap speaker relabel that propagates (Otter-style) is partially there via `speaker_editor`.

Want me to turn section 7 into a prioritized roadmap (effort vs. impact), or deep-dive any one area (e.g., voiceprint enrollment, or wiring the real NPU path)?