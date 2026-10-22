#!/usr/bin/env python3
"""Transcribe an audio file locally with mlx-whisper on Apple Silicon.

Examples:
    tacito.py "~/Downloads/reel.wav"
    tacito.py reel.m4a --lang it --formats txt,srt,vtt
    tacito.py interview.mp3 --model mlx-community/whisper-large-v3-turbo-q4

No ffmpeg dependency when the file is readable by soundfile (wav, flac, ogg,
mp3). For m4a, mp4, mov, aac, webm and opus, ffmpeg is used if present on PATH.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

TARGET_SR = 16_000
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def load_audio(path: Path) -> tuple[np.ndarray, int | None]:
    """Return (mono float32 samples at 16 kHz, source sample rate).

    The source rate is None when ffmpeg did the decoding, because it resamples
    in the same pass and never reports the original rate back to us.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and path.suffix.lower() in {".m4a", ".mp4", ".aac", ".mov", ".webm", ".opus"}:
        cmd = [
            ffmpeg, "-nostdin", "-threads", "0", "-i", str(path),
            "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR), "-",
        ]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.float32), None

    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
    except Exception as exc:
        if path.suffix.lower() != ".wav":
            raise SystemExit(
                f"Cannot decode {path.name}: {exc}\n"
                "This format needs ffmpeg: brew install ffmpeg"
            ) from exc
        import wave

        with wave.open(str(path)) as w:
            sr = w.getframerate()
            width, nch = w.getsampwidth(), w.getnchannels()
            frames = w.readframes(w.getnframes())
        if width == 1:
            # 8-bit PCM in WAV is unsigned (0..255, zero at 128), unlike every
            # other bit depth, which is signed.
            pcm = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            pcm = (pcm - 128.0) / 128.0
        else:
            dtype = {2: np.int16, 4: np.int32}.get(width)
            if dtype is None:
                raise SystemExit(f"Unsupported bit depth: {width * 8} bit")
            pcm = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            pcm /= float(np.iinfo(dtype).max)
        mono = pcm.reshape(-1, nch).mean(axis=1) if nch > 1 else pcm

    return resample(mono, sr, TARGET_SR), sr


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Frequency-domain resampling: a brick-wall filter, so anti-aliased."""
    if sr_in == sr_out:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    spec = np.fft.rfft(x)
    out_bins = n_out // 2 + 1
    keep = min(len(spec), out_bins)
    new_spec = np.zeros(out_bins, dtype=spec.dtype)
    new_spec[:keep] = spec[:keep]
    y = np.fft.irfft(new_spec, n=n_out) * (n_out / len(x))
    return y.astype(np.float32)


def ts(seconds: float, sep: str = ",") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def wrap_caption(text: str, width: int = 42, max_lines: int = 2) -> str:
    """Widen the wrap width until the text fits max_lines, as subtitling
    convention requires. Never truncate: past the limit, accept extra lines."""
    text = text.strip()
    for w in range(width, 57, 7):
        lines = textwrap.wrap(text, width=w)
        if len(lines) <= max_lines:
            return "\n".join(lines)
    return "\n".join(textwrap.wrap(text, width=56) or [""])


def to_srt(segments: list[dict]) -> str:
    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(
            f"{i}\n{ts(seg['start'])} --> {ts(seg['end'])}\n{wrap_caption(seg['text'])}\n"
        )
    return "\n".join(blocks)


def to_vtt(segments: list[dict]) -> str:
    blocks = ["WEBVTT\n"]
    for seg in segments:
        blocks.append(
            f"{ts(seg['start'], '.')} --> {ts(seg['end'], '.')}\n{wrap_caption(seg['text'])}\n"
        )
    return "\n".join(blocks)


def to_paragraphs(segments: list[dict], gap: float = 2.0, max_chars: int = 600) -> str:
    """Flowing text: break on a long pause, or at a sentence end past max_chars."""
    paragraphs: list[str] = []
    buf: list[str] = []
    prev_end = None
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        long_pause = prev_end is not None and seg["start"] - prev_end > gap
        current = " ".join(buf)
        full_enough = len(current) >= max_chars and current.endswith((".", "!", "?", "…"))
        # Without the hard cap, dense speech that never ends a sentence would
        # stay one unreadable paragraph.
        too_long = len(current) >= max_chars * 2
        if buf and (long_pause or full_enough or too_long):
            paragraphs.append(current)
            buf = []
        buf.append(text)
        prev_end = seg["end"]
    if buf:
        paragraphs.append(" ".join(buf))
    return "\n\n".join(paragraphs) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Local transcription with mlx-whisper")
    ap.add_argument("audio", help="audio file to transcribe")
    ap.add_argument("--lang", default=None, help="language code (it, en, ...); default: autodetect")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"model HF repo (default: {DEFAULT_MODEL})")
    ap.add_argument("--outdir", default=None, help="output directory (default: same as the audio file)")
    ap.add_argument("--formats", default="txt,srt", help="comma-separated: txt,srt,vtt,json (default: txt,srt)")
    ap.add_argument("--prompt", default=None, help="initial prompt to seed proper nouns and jargon")
    args = ap.parse_args()

    path = Path(os.path.expanduser(args.audio)).resolve()
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")
    formats = {f.strip().lower() for f in args.formats.split(",") if f.strip()}
    unknown = formats - {"txt", "srt", "vtt", "json"}
    if unknown:
        raise SystemExit(f"Unknown formats: {', '.join(sorted(unknown))}")

    outdir = Path(os.path.expanduser(args.outdir)).resolve() if args.outdir else path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    audio, sr_in = load_audio(path)
    duration = len(audio) / TARGET_SR
    origin = (
        f"source {sr_in} Hz -> {TARGET_SR} Hz" if sr_in
        else f"decoded by ffmpeg at {TARGET_SR} Hz"
    )
    print(f"Audio    : {path.name}", file=sys.stderr)
    print(f"Duration : {duration:.1f} s  ({origin})", file=sys.stderr)
    print(f"Model    : {args.model}", file=sys.stderr)
    print("Transcribing...", file=sys.stderr)

    import mlx_whisper

    started = time.monotonic()
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=args.model,
        language=args.lang,
        initial_prompt=args.prompt,
        word_timestamps=False,
        verbose=None,
    )
    elapsed = time.monotonic() - started

    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result.get("segments", [])
        if s.get("text", "").strip()
    ]
    if not segments:
        print("No speech detected in the audio.", file=sys.stderr)
        return 1

    stem = path.stem
    written = []
    if "txt" in formats:
        p = outdir / f"{stem}.txt"
        p.write_text(to_paragraphs(segments), encoding="utf-8")
        written.append(p)
    if "srt" in formats:
        p = outdir / f"{stem}.srt"
        p.write_text(to_srt(segments), encoding="utf-8")
        written.append(p)
    if "vtt" in formats:
        p = outdir / f"{stem}.vtt"
        p.write_text(to_vtt(segments), encoding="utf-8")
        written.append(p)
    if "json" in formats:
        p = outdir / f"{stem}.json"
        p.write_text(
            json.dumps(
                {
                    "source": str(path),
                    "duration_s": round(duration, 3),
                    "language": result.get("language"),
                    "model": args.model,
                    "segments": segments,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        written.append(p)

    words = sum(len(s["text"].split()) for s in segments)
    print(
        f"\nLanguage : {result.get('language')}\n"
        f"Segments : {len(segments)}  |  words: {words}\n"
        f"Time     : {elapsed:.1f} s  ({duration / elapsed:.1f}x realtime)",
        file=sys.stderr,
    )
    for p in written:
        print(f"Wrote    : {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
