# VoxNovel commands

This is the command reference for the scripts currently present in this repository. Run commands from the repository root unless a section says otherwise.

The repository contains two different audiobook paths:

1. **Legacy VoxNovel workflow** — GUI/headless scripts and the upstream README's CPU/GPU/Docker launch examples.
2. **Incremental XTTS workflow** — the resumable `voxnovel_incremental.py` renderer and `assemble_voxnovel_incremental.py` M4B assembler. This is the recommended path for the current *The City & The City* project when using the incremental WAV directory and its manifest.

Do not mix the legacy `Working_files/generated_audio_clips` output with the incremental assembler. The incremental assembler expects one WAV per `audio_id` in `Working_files/generated_audio_clips_incremental`, plus `manifest.json` unless explicitly bypassed.

## 1. Get the repository and enter its root

```bash
git clone https://github.com/m-ibm/AudioNovel.git
cd AudioNovel
```

If the repository is already present:

```bash
cd /path/to/AudioNovel
git pull
```

The README filename in this repository is lowercase: [readme.md](readme.md).

## 2. Prepare the book and speaker/voice assignments

The legacy entry points documented by `readme.md` are:

| Stage / mode | Command | Notes |
|---|---|---|
| GUI workflow | `python gui_run.py` | Interactive GUI entry point. |
| Headless workflow | `python headless_voxnovel.py` | Legacy headless script; its README references include Colab-oriented usage. |
| Automatic no-GUI workflow | `python auto_noGui_run.py` | Legacy automated no-GUI entry point. |
| Low-VRAM book processing stage | `python 1CPU_Book_processing.py` | Run before the paired audio-generation stage in the documented low-VRAM sequence. |
| Low-VRAM audio generation stage | `python 2GPU_Audio_generation.py` | Run after book processing. |

These scripts are interactive/legacy entry points; the repository README does not document command-line flags for them. Do not append guessed flags. Follow their prompts and the README's setup instructions.

The incremental renderer does **not** run BookNLP or infer speaker/voice assignments. It consumes a prepared `book.csv`. Before rendering, check that the CSV contains at least:

- `Text`
- `Speaker`
- `audio_id`
- `voice_actor`
- `language`

Each `voice_actor` must have at least one reference audio file under `tortoise/voices/<voice_actor>/`. The incremental renderer validates these references before loading the TTS model.

## 3. Incremental XTTS audio generation

### Standard full-book generation

From the repository root:

```bash
python voxnovel_incremental.py \
  --book-csv Working_files/Book/book.csv \
  --output-dir Working_files/generated_audio_clips_incremental \
  --model tts_models/multilingual/multi-dataset/xtts_v2 \
  --language en \
  --device auto \
  --min-pad-words 10
```

The defaults already select these paths/model/device/padding threshold, so the shortest equivalent command is:

```bash
python voxnovel_incremental.py
```

The renderer writes per-row WAV files and `manifest.json` to the output directory. A row is skipped on a later run only when its manifest fingerprint and WAV hash still match. Completed rows are committed to the manifest immediately; rerun the same command to retry failed rows while preserving completed clips.

### Targeted tests and reruns

Render at most the first 5 rows that still need generation:

```bash
python voxnovel_incremental.py --max-rows 5
```

Render selected IDs only:

```bash
python voxnovel_incremental.py --only-audio-ids audio_25_1,audio_47_1,audio_62_1
```

Combine an ID allowlist with a row limit if useful:

```bash
python voxnovel_incremental.py --only-audio-ids audio_25_1,audio_47_1,audio_62_1 --max-rows 2
```

Retry failures by rerunning the normal command; there is no separate retry flag:

```bash
python voxnovel_incremental.py
```

### Padding/alignment variations

Disable short-fragment padding/cropping:

```bash
python voxnovel_incremental.py --no-padding
```

Change the short-fragment threshold (fragments with fewer than 8 words are padded):

```bash
python voxnovel_incremental.py --min-pad-words 8
```

Preload the wav2vec2 CTC aligner and exit without generating audiobook clips:

```bash
python voxnovel_incremental.py --preload-aligner-only
```

The aligner preload option cannot be combined with `--no-padding`. Padding/cropping is implemented only for XTTS models.

### Device selection

Native macOS with Apple Silicon may use MPS when available:

```bash
python voxnovel_incremental.py --device mps
```

CPU:

```bash
python voxnovel_incremental.py --device cpu
```

CUDA:

```bash
python voxnovel_incremental.py --device cuda
```

Automatic device selection (default):

```bash
python voxnovel_incremental.py --device auto
```

**Docker/Colima note:** the script explicitly warns that Apple MPS is available only to native macOS Python; Docker/Colima does not expose Apple MPS. Use `--device cpu` or `--device auto` inside a container, depending on the container's PyTorch runtime. Do not use `--device mps` in Docker.

### Incremental renderer parameters

All parameters below are defined by the script's argument parser.

| Parameter | Default | Meaning |
|---|---|---|
| `--book-csv PATH` | `Working_files/Book/book.csv` | Prepared input CSV. |
| `--output-dir PATH` | `Working_files/generated_audio_clips_incremental` | Per-row WAVs, manifest and audit outputs. |
| `--model NAME` | `VOXNOVEL_TTS_MODEL` if set, otherwise `tts_models/multilingual/multi-dataset/xtts_v2` | Coqui TTS model identifier. |
| `--language CODE` | empty; fallback is `en` | Language used when a row's `language` cell is empty. |
| `--device auto\|cpu\|cuda\|mps` | `auto` | Compute device. MPS only in native macOS Python. |
| `--min-pad-words N` | `10` | Pad each fragment shorter than this word count. Must be at least 1. |
| `--no-padding` | off | Disable short-fragment padding and cropping. |
| `--max-rows N` | `0` | Generate at most the first N rows that need generation; 0 means no limit. |
| `--only-audio-ids ID,...` | empty | Restrict work to a comma-separated allowlist of `audio_id` values. Unknown IDs are errors. |
| `--preload-aligner-only` | off | Load the CTC aligner, then exit. Cannot be combined with `--no-padding`. |

The model can also be selected through the `VOXNOVEL_TTS_MODEL` environment variable; an explicit `--model` argument takes precedence.

## 4. Assemble the incremental clips into an M4B

Run only after the required row WAVs have been generated:

```bash
python assemble_voxnovel_incremental.py \
  --book-csv Working_files/Book/book.csv \
  --clips-dir Working_files/generated_audio_clips_incremental \
  --output-dir output_audiobook \
  --title "The City & The City" \
  --author "China Miéville" \
  --silence-ms 750
```

The assembler validates each WAV, checks manifest status and SHA-256 hashes, verifies consistent sample rate/channel count, inserts silence between clips, writes chapter metadata, and encodes an AAC M4B using FFmpeg. It requires both `ffmpeg` and `ffprobe` on `PATH`.

The input CSV must include `audio_id`, `chapter`, `chapter_title`, `Text`, and `voice_actor`. If `Start Location` exists, it is used for stable within-chapter ordering.

In the current `main` version, the output filename is hard-coded by the script, regardless of `--title`:

`output_audiobook/The_City_and_The_City.m4b`

This is a book-specific default inherited from the current project. It is not derived from `book.csv`; pass `--title` and `--author` explicitly for correct embedded metadata. The companion progress-bar PR proposes replacing these defaults with generic metadata defaults and deriving the filename from `--title`.

### Assembler parameters

| Parameter | Default | Meaning |
|---|---|---|
| `--book-csv PATH` | `Working_files/Book/book.csv` | Input CSV and chapter order. |
| `--clips-dir PATH` | `Working_files/generated_audio_clips_incremental` | Directory containing row WAVs and normally `manifest.json`. |
| `--output-dir PATH` | `output_audiobook` | Destination directory for the M4B. |
| `--title TEXT` | `The City & The City` | Embedded audiobook title metadata. |
| `--author TEXT` | `China Miéville` | Embedded artist/author metadata. |
| `--silence-ms N` | `750` | Silence inserted between adjacent clips; negative values are treated as zero. |
| `--allow-missing-manifest` | off | Permit assembly without a manifest; WAV existence/audio-format checks still run, but manifest status/hash verification is skipped. |

For this book, pass the actual title and author explicitly, as in the example. The current defaults are hard-coded project values and should not be relied on for other books.

### If the manifest is absent

Only use this for a directory whose WAVs you have independently verified:

```bash
python assemble_voxnovel_incremental.py \
  --book-csv Working_files/Book/book.csv \
  --clips-dir Working_files/generated_audio_clips_incremental \
  --output-dir output_audiobook \
  --title "The City & The City" \
  --author "China Miéville" \
  --allow-missing-manifest
```

This flag disables the manifest state and hash checks; it does not repair missing or invalid WAVs.

## 5. Legacy Docker launch examples from the upstream README

The following commands are documented in this repository's inherited `readme.md`. They are legacy upstream examples, not a guarantee that the current incremental scripts are installed in those images.

### M1 Mac headless image (upstream example)

```bash
cd ~
git clone https://github.com/DrewThomasson/VoxNovel.git
docker run -v "$HOME/VoxNovel:/VoxNovel/" -it athomasson2/voxnovel:headless_m1_v2
```

### CPU-only headless Docker (upstream example)

```bash
cd ~
git clone https://github.com/DrewThomasson/VoxNovel.git
docker run -v "$HOME/VoxNovel:/VoxNovel/" -it athomasson2/voxnovel:latest_headless
```

### NVIDIA GPU headless Docker (upstream example)

```bash
cd ~
git clone https://github.com/DrewThomasson/VoxNovel.git
docker run --gpus all -v "$HOME/VoxNovel:/VoxNovel/" -it athomasson2/voxnovel:latest_headless
```

### Windows PowerShell CPU-only headless Docker (upstream example)

```powershell
cd $env:USERPROFILE
git clone https://github.com/DrewThomasson/VoxNovel.git
docker run -v "${env:USERPROFILE}/VoxNovel/:/VoxNovel/" -it athomasson2/voxnovel:latest_headless
```

### Windows PowerShell NVIDIA GPU headless Docker (upstream example)

```powershell
cd $env:USERPROFILE
git clone https://github.com/DrewThomasson/VoxNovel.git
docker run --gpus all -v "${env:USERPROFILE}/VoxNovel/:/VoxNovel/" -it athomasson2/voxnovel:latest_headless
```

The README does not provide a current Colima-specific image build/run command for the incremental renderer. Colima deployments using a locally built image must use that image's actual name, mount, working directory and Python environment; those are deployment-specific and are deliberately not fabricated here.

## 6. Checks before assembly

From the repository root:

```bash
# Confirm the expected input and incremental output paths exist.
test -f Working_files/Book/book.csv
test -f Working_files/generated_audio_clips_incremental/manifest.json

# Count expected book rows and generated WAVs.
python -c 'import pandas as pd; print("book rows:", len(pd.read_csv("Working_files/Book/book.csv")))'
find Working_files/generated_audio_clips_incremental -maxdepth 1 -name 'audio_*.wav' | wc -l

# Confirm the required media tools are available.
ffmpeg -version
ffprobe -version
```

A matching manifest entry and SHA-256 hash are required by default. If assembly stops on a missing clip, failed manifest status, hash mismatch, or incompatible WAV format, fix or regenerate that clip rather than bypassing validation automatically.

## 7. Scope and non-goals

- `voxnovel_incremental.py` is the resumable row-level XTTS renderer documented here.
- `assemble_voxnovel_incremental.py` assembles that incremental output only.
- The legacy GUI/headless scripts are separate paths with their own assumptions and output folders.
- Book-specific character-attribution/voice-assignment scripts that are not in this repository are not listed as repository commands.
- The inherited README contains additional historical setup/UI material; this file focuses on executable entry points and the current incremental workflow.
