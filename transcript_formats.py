"""Transcript side-exports: plain text, timestamped text, SRT, and WebVTT.

Captions are worth generating even when nobody asked: if the lecture recording
goes into the LMS, an accurate caption file is usually the cheapest accessibility
win available, and the transcript already has the timings.
"""

from __future__ import annotations

from ..schema import Transcript


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _vtt_time(seconds: float) -> str:
    return _srt_time(seconds).replace(",", ".")


def export_txt(transcript: Transcript) -> bytes:
    return transcript.text.encode("utf-8")


def export_timestamped_txt(transcript: Transcript) -> bytes:
    return transcript.text_with_timestamps().encode("utf-8")


def export_srt(transcript: Transcript) -> bytes:
    blocks = [
        f"{i}\n{_srt_time(s.start)} --> {_srt_time(s.end)}\n{s.text.strip()}\n"
        for i, s in enumerate(transcript.segments, start=1)
    ]
    return "\n".join(blocks).encode("utf-8")


def export_vtt(transcript: Transcript) -> bytes:
    blocks = ["WEBVTT", ""]
    for s in transcript.segments:
        blocks.append(f"{_vtt_time(s.start)} --> {_vtt_time(s.end)}")
        blocks.append(s.text.strip())
        blocks.append("")
    return "\n".join(blocks).encode("utf-8")
