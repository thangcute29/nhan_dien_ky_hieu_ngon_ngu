# WLASL sequence pipeline

The files in `Research_and_Data/Dataset/Sequences/videos` use numeric WLASL
video IDs. The class label and official split come from `nslt_100.json`; class
IDs are resolved with `wlasl_class_list.txt`.

The corrected output is intentionally stored in `processed_wlasl100`. The old
`processed` directory is retained only for comparison because its folder names
do not represent a consistent vocabulary.

```powershell
# Validate metadata without writing data
python Data_preparation/Prepare_sequences.py --dry-run

# Extract all locally available WLASL-100 videos
python Data_preparation/Prepare_sequences.py

# Train after extraction completes
python Cloud_server/Trainer/train_scripts/train_gru.py

# Run checks
python -m unittest discover -s tests -v
```

Extraction is resumable: existing `.npy` files are skipped. Use `--overwrite`
only when preprocessing rules change. Each output has shape `(30, 126)` and
contains raw `[Left hand | Right hand]` MediaPipe xyz landmarks. One shared
fixed-anchor transform is applied during both training and inference.

Configure Gemini through the environment instead of source code:

```powershell
$env:GEMINI_API_KEY = "your-key"
```
