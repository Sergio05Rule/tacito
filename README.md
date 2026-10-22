# tacito

*Named for the Roman historian who kept the record, and for what stays unspoken.*

Offline speech-to-text for Apple Silicon. One script, no cloud, no API keys, no `ffmpeg` for common formats. Audio never leaves the machine.

```
$ tacito.py reel.wav
Audio    : reel.wav
Duration : 120.0 s  (source 16000 Hz -> 16000 Hz)
Model    : mlx-community/whisper-large-v3-turbo
Transcribing...

Language : it
Segments : 32  |  words: 332
Time     : 7.4 s  (16.3x realtime)
Wrote    : /Users/you/audio/reel.txt
Wrote    : /Users/you/audio/reel.srt
```

## What it does

Takes an audio file, returns a transcript in up to four formats:

| Format | Contents |
|---|---|
| `.txt` | Flowing prose, split into paragraphs on long pauses and sentence ends |
| `.srt` | Subtitles, 2 lines per caption where the text allows, more when it does not |
| `.vtt` | WebVTT, same segmentation |
| `.json` | Raw segments with `start`/`end` in seconds, detected language, model id |

Language is detected automatically. Default is `txt,srt`. Outputs land next to the source file unless `--outdir` says otherwise, and same-stem files are overwritten without asking.

## Why

Transcription is a solved problem that is still annoying to actually do, because every option costs you something:

- **Cloud APIs** bill per minute and require uploading the file. Fine for public content, wrong for a private recording, a client call, or anything under NDA.
- **Free web tools** cap your minutes, watermark the output, or keep your audio as the real price.
- **Free local engines exist**, so this is no monopoly. `whisper.cpp` is excellent and far more portable. macOS 26 ships `SpeechTranscriber`, which costs nothing and needs no download, but degrades when speech sits over a music bed, which is the normal condition for social video.

What is missing is the boring part: one file you can read in five minutes, that handles the formats you actually have, that does not shell out to `ffmpeg` for a WAV, and that runs on the GPU you already paid for.

`mlx.default_device()` returns `Device(gpu, 0)`: inference runs on the GPU through Metal with unified memory, so there are no host-to-device tensor copies. The network is used twice, once to install and once to fetch the weights. After that it works on a plane. No keys, no telemetry, no account.

## Model and weights

Default is `mlx-community/whisper-large-v3-turbo`: 587 tensors, 806,958,092 parameters, float16, 1.5 GiB.

Two artifacts, often mistaken for one. The pip package holds **117 KB of Python and zero weights**: it defines the architecture, the decoding loop, the spectrogram. (It occupies 1.9 MB on disk, but 1.6 MB of that is tiktoken BPE vocabulary plus the precomputed mel filter bank, not code.) The trained numbers arrive separately from HuggingFace as a `safetensors` file, which is nothing but a table of `tensor name -> block of numbers`. Nothing is overwritten: the code is born with empty matrices and the weights fill them, which is why the same package serves `tiny`, `large-v3` and quantized variants, selected at runtime by `--model`.

The `config.json` shows what "turbo" means:

```json
{"n_audio_layer": 32, "n_text_layer": 4, "n_audio_state": 1280, "n_vocab": 51866}
```

Encoder untouched at 32 layers, decoder distilled from 32 down to 4. The encoder runs once per 30 s window while the decoder runs once per generated token, so cutting the decoder is where the speedup comes from, at marginal accuracy cost.

## Architecture

```
wav/flac/ogg/mp3 ──soundfile──> float32 mono ──FFT resample──> 16 kHz
     │                                                            │
     └── m4a/mp4/mov/aac/webm/opus ──ffmpeg──────────────────────> ┤
                                                                   ▼
                           30 s windows ──STFT + mel──> (128, 3000) log-mel
                                                                   │
                 2x conv1d (stride 2) ──> 1500 frames              │
                 32 self-attn blocks, 635 M params  <──────────────┘
                                         │
                                         ▼  cross-attention
                 4 decoder blocks, 172 M params ──> token ids
                                         │
                         tiktoken BPE ───┴──> text + timestamp tokens
                                         │
                                         ▼
                            txt / srt / vtt / json
```

Four details in that chain are worth knowing:

- **Decoding bypasses `mlx_whisper.audio.load_audio()`**, which shells out to `ffmpeg`, so wav, flac, ogg and mp3 need no external binary.
- **Resampling is a frequency-domain brick-wall filter.** Verified: RMS ratio 1.0000 across 19 frequency and rate combinations, and a 10 kHz tone resampled 44100 to 16000 leaves zero residue instead of aliasing down to 6 kHz.
- **The mel stage is plain signal processing**, no network: Hann STFT, `N_FFT = 400` (25 ms), `HOP_LENGTH = 160` (10 ms), 3000 frames onto 128 bands. Decoding is greedy at temperature 0, falling back to 0.2, 0.4 and up when the compression ratio exceeds 2.4 (the model has started looping) or mean logprob drops below -1.0. Timestamps quantize to 20 ms.
- **Windows are not independent.** Each one advances to the last emitted timestamp rather than a blind +30 s, and inherits the previous window's text as its prompt. That is why splitting one file across parallel workers makes the output worse rather than faster.

## Performance

Measured on a 120 s Italian clip, not estimated.

| Run | Inference | Wall clock |
|---|---|---|
| Cold | 158.4 s (0.8x realtime) | ~2 min 40 s |
| Warm | **7.4 s (16.3x realtime)** | 9.1 s |

The 21x gap is entirely outside inference: the cold run pays for the weight download plus Metal kernel compilation. Only 3.9 s of the warm run is CPU time, the rest is GPU.

Also verified on that clip: 32 SRT blocks matching 32 JSON segments, sequential indices, zero overlaps, monotonic timestamps, nothing past the audio duration, 95.7% temporal coverage with no gap above 1.5 s, no hallucinated repeats, and a second run producing byte-identical text with zero diverging timestamps.

## Design trade-offs

| Decision | Why | Cost |
|---|---|---|
| MLX over PyTorch MPS | Unified memory, no host-to-device copies | Apple Silicon only, no portability |
| `turbo` over `large-v3` | Much faster from the 4-layer decoder | Slightly higher word error rate |
| Bypass `load_audio()` | Drops the hard `ffmpeg` dependency | Exotic containers still need ffmpeg |
| FFT resampling | Correct anti-aliasing in 10 lines, no scipy chain | Brick-wall filter rings at clip edges |
| Sequential windows | Preserves cross-window context and punctuation | Cannot parallelize a single file |
| Greedy, temperature 0 | Reproducible output, verified identical across runs | No n-best alternatives to inspect |
| `word_timestamps=False` | Skips the numba DTW pass over cross-attention | Timestamps per segment, not per word, so dense captions can exceed 2 lines |
| Dedicated venv | Homebrew Python is PEP 668 externally managed | 1.2 GB, of which 583 MB is unused `torch` |

That last row deserves a mention: `mlx-whisper` declares `torch` as a dependency but only imports it in `torch_whisper.py`, the converter for original OpenAI checkpoints. With pre-converted weights that file never runs, so the 583 MB is pure disk cost. Total footprint is about 2.7 GB, of which 117 KB is the code that matters.

## Install

Apple Silicon and macOS only, Python 3.10 or newer. MLX has no CUDA or x86 path.

```bash
python3 -m venv ~/.venvs/tacito
~/.venvs/tacito/bin/pip install -r requirements.txt
```

Call it through that interpreter:

```bash
~/.venvs/tacito/bin/python tacito.py AUDIO
```

For a bare `./tacito.py`, repoint the shebang and keep the edit out of your commits:

```bash
sed -i '' '1s|.*|#!'"$HOME"'/.venvs/tacito/bin/python|' tacito.py
git update-index --skip-worktree tacito.py   # so your home path never gets committed
chmod +x tacito.py
```

Weights download on first use and are cached in `~/.cache/huggingface/`.

## Usage

```text
tacito.py AUDIO [--lang it] [--model REPO] [--outdir DIR]
                [--formats txt,srt,vtt,json] [--prompt "..."]
```

- `--lang` skips detection. Worth forcing under 30 seconds, on noisy audio, or when two languages are mixed, since detection reads only the first window.
- `--prompt` seeds proper nouns and jargon, for example `--prompt "Kubernetes, PostgreSQL, gRPC"`.
- `--model` accepts any MLX Whisper repo. `whisper-large-v3-turbo-q4` is 460 MB instead of 1.5 GiB and faster, at some accuracy cost.

To remove everything:

```bash
rm -rf ~/.venvs/tacito ~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo
```

## Known limits

- Whisper can repeat phrases over long silence or instrumental passages. The temperature fallback suppresses most of it, but check the SRT for repeated lines before trusting a long transcript.
- Proper nouns and homophones are where this model actually errs, and only context disambiguates them. Proofread names.
- When `ffmpeg` does the decoding it resamples in the same pass and never reports the original rate, so the `Duration` line says so instead of inventing a number.

## Claude Code skill

`claude-skill/SKILL.md` wraps the script for [Claude Code](https://claude.com/claude-code). Copy the script to `~/scripts/tacito.py` and the skill to `~/.claude/skills/transcribe/SKILL.md`, then invoke with `/transcribe`. The skill carries the operational judgment, when to force a language and when a prompt helps, so the agent does not rediscover it each session.

## License

MIT, see [LICENSE](LICENSE).
