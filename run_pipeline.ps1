.\.venv\Scripts\Activate.ps1
python -m bubble.cli extract --input-dir videos --output-dir output_frames
python -m bubble.cli filter --input-dir output_frames --output-dir filtered_frames
python -m bubble.cli resize --input-dir filtered_frames --output-dir resized_frames
python -m bubble.cli train --config configs/default.toml