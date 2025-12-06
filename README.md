# Whisper Batch Transcriber (Video/Audio → TXT)

A small CLI tool for transcribing **video files** (and any media with supported extensions) into text using **OpenAI Whisper**.  
It supports processing a **single file** or an **entire directory**, merging results into one output `.txt` with clear per-file headers.

The script also improves readability by inserting **line breaks when a pause in speech exceeds a configurable threshold**.

---

## Features

- Transcribe a **single file** or **multiple files in a folder**
- Uses **OpenAI Whisper** with CPU or CUDA
- Loads the model **once** for batch processing
- Adds per-file headers:
  ```
  ===== PLIK: filename.mkv =====
  ```
- Inserts new lines based on long pauses in the audio
- Smart output naming for directory mode:
  - If filenames contain a start datetime in one of these formats:
    - `YYYY-MM-DD HH-MM-SS`
    - `YYYY-MM-DD_HH-MM-SS`
  - The output will be named using inferred start/end timestamps:
    ```
    2025-11-26 16-44-59__2025-11-26 18-10-12_medium.txt
    ```
  - Otherwise, it falls back to:
    ```
    firstfile_len_02h15m_medium.txt
    ```

---

## Supported Extensions

The script currently detects media by extension:

```txt
.mp4
.mkv
.avi
.mov
.flv
.wmv
.mpg
.mpeg
.m4v
.webm
```

> If you want audio-only formats (`.mp3`, `.wav`, `.m4a`, `.flac`), you can extend `VIDEO_EXTENSIONS`.

---

## Requirements

- Python **3.10+**
- **FFmpeg** (required by Whisper for decoding media)
- Python packages:
  - `torch`
  - `openai-whisper`

---

## Installation

### 1) Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**
- Install FFmpeg and add it to your `PATH`.

### 2) Install Python dependencies

```bash
pip install -U openai-whisper
```

Or explicitly:

```bash
pip install -U torch openai-whisper
```

> For GPU usage, install a `torch` build matching your CUDA version.

---

## Usage

### Transcribe a single file

```bash
python transcribe.py "2025-12-03 18-29-46.mkv"
```

Default output (same folder):

```txt
2025-12-03 18-29-46.txt
```

---

### Transcribe a directory

```bash
python transcribe.py ./recordings
```

The script will:

1. find supported media files,
2. sort them by filename,
3. transcribe each file,
4. merge everything into one `.txt`.

---

## CLI Arguments

| Argument | Description | Default |
|---|---|---|
| `input` | Path to a media file or a directory | required |
| `-o, --output` | Output `.txt` path | auto |
| `-m, --model` | Whisper model: `tiny/base/small/medium/large` | `small` |
| `-l, --language` | Spoken language code | `pl` |
| `-d, --device` | Force device: `cpu` or `cuda` | auto |
| `--pause-threshold` | Pause (seconds) after which a new line starts | `2.0` |

---

## Examples

### Force CUDA

```bash
python transcribe.py ./recordings -d cuda
```

### Use a larger model

```bash
python transcribe.py "lecture.mp4" -m medium
```

### Adjust pause threshold

More frequent line breaks:

```bash
python transcribe.py "lecture.mp4" --pause-threshold 1.2
```

Fewer line breaks:

```bash
python transcribe.py "lecture.mp4" --pause-threshold 3.0
```

### Custom output path

```bash
python transcribe.py ./recordings -o ./full_transcription.txt
```

---

## How line splitting works

Whisper returns segments with `start` and `end` timestamps.  
This script groups segment text into lines and starts a new line when:

```
(next_start - previous_end) >= pause_threshold
```

---

## Output Structure

In directory mode you will get:

```txt
===== PLIK: file1.mkv =====

[transcription...]

===== PLIK: file2.mkv =====

[transcription...]
```

---

## Notes

- `small` is a good speed/quality balance for Polish.
- The script runs Whisper with:
  ```python
  verbose=True
  ```
  so you will see segment-level logs in the console.

---

## Possible Improvements

- Add audio-only extensions to detection
- Optional timestamps in output
- Save raw `result["text"]` as an alternative format
- Parallel processing (mind GPU VRAM)

---

## License

Choose a license that fits your project (e.g., MIT).

