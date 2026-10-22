---
name: transcribe
description: Transcribe an audio or video file locally (no cloud, no upload) with mlx-whisper on Apple Silicon, producing flowing text and timestamped subtitles. Use when the user supplies an audio/video file and asks for a transcript, subtitles, an .srt or .vtt, to "find out what this reel says", or for a voice memo, meeting recording or podcast; also when they ask to locate a specific moment inside a spoken file.
---

# Local audio transcription

Everything runs on the user's machine: the audio never leaves the computer. Do
not propose cloud services (OpenAI, AssemblyAI, Deepgram) unless the user asks
explicitly, because sending the file to a third party publishes it off-device.

## Command

```bash
~/scripts/tacito.py <audio file> [--lang it] [--formats txt,srt,vtt,json] [--outdir DIR] [--prompt "..."]
```

The script's shebang already points at its virtualenv, so invoke it directly
with its absolute path and never activate anything. If it lives elsewhere,
adjust the path rather than assuming a working directory.

Outputs are written next to the source file, same stem, with the format's
extension, and same-stem files are overwritten without warning.

Defaults: `--formats txt,srt`, autodetected language, model
`mlx-community/whisper-large-v3-turbo`.

## Choosing options

- **Language.** Leave autodetection on except when the audio is short (under 30
  seconds), very noisy, or mixes two languages. In those cases pass `--lang`,
  because detection reads only the first window and a wrong guess can make
  Whisper translate instead of transcribe, or switch languages mid-file.
- **Formats.** `txt` to read, `srt` to re-cut a video or find a timestamp,
  `json` when the segments feed later processing, `vtt` only for the web.
- **`--prompt`.** For proper nouns, acronyms and jargon. Example for a technical
  recording: `--prompt "Kubernetes, PostgreSQL, gRPC, Terraform"`. It improves
  spelling of domain terms, which is where this model errs most.
- **Model.** `--model mlx-community/whisper-large-v3-turbo-q4` downloads roughly
  460 MB instead of 1.5 GB and runs faster, at the cost of a few more errors.
  Use the full turbo by default.

## Accepted formats

WAV, FLAC, OGG and MP3 work with no external dependency. For M4A, MP4, MOV, AAC,
WebM and Opus the script uses `ffmpeg` if it is on PATH; if it is missing, the
error says to install it with `brew install ffmpeg`. To pull audio out of a video
when ffmpeg is absent, ask the user how they want to proceed rather than
installing software behind their back.

## After transcribing

Read the resulting `.txt` and report the content to the user, not just the file
path. If the text is long, give the main thesis and the key points, then offer
the full transcript. The `.srt` is the source to cite when the user asks "where
does it say X": give the timestamp.

## Operational notes

- The first run with a new model downloads weights from HuggingFace: launch it
  in the background and warn the user about the wait.
- Expected speed with turbo once the model is cached is roughly 16x realtime,
  so a long wait on a second run signals a problem, not normal behavior.
- Whisper can hallucinate repeated phrases over long silence or over music with
  no voice. If the same line repeats many times in the `.srt`, say so instead of
  presenting it as real speech.
- Proofread proper nouns. Homophones and names are where this model actually
  errs, and only context disambiguates them.
