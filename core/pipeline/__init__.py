# core/pipeline/__init__.py
#
# Orchestrators kết hợp diarization + transcription + post-processing.
#
# Cấu trúc:
#   offline.py    — pipeline cho file batch (sẽ sống ở đây từ Phase 2 trở đi)
#   streaming.py  — pipeline 2-pass cho live mode (Phase 3, two-pass theo Otter)
#   postprocess.py— re-cluster + voiceprint match khi recording kết thúc (Phase 3)
#
# Hiện tại pages/1_Batch_Mode.py và pages/2_Live_Mode.py vẫn gọi trực tiếp
# diarizer + transcriber + aligner. Phase 2+ sẽ gom logic đó vào đây.
