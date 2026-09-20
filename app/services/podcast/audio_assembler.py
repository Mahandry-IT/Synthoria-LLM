"""Assemblage audio : chronologie, sous-titres VTT, chapitres et mixage ffmpeg.

La chronologie (timings, VTT, métadonnées) est calculée en Python à partir de la durée
des WAV. ffmpeg est appelé via create_subprocess_exec avec une liste d'arguments (jamais
de shell) ; aucun texte utilisateur n'apparaît dans la ligne de commande : titres et
chapitres passent par un fichier ffmetadata.
"""

import asyncio
import contextlib
import logging
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.core.exceptions import AudioAssemblyError

logger = logging.getLogger(__name__)

PAUSE_TURN_SECONDS = 0.35
PAUSE_CHAPTER_SECONDS = 0.9
FFMPEG_TIMEOUT_SECONDS = 900


@dataclass(frozen=True)
class TimedTurn:
    """Une réplique : ses morceaux WAV, son texte affiché et le chapitre qu'elle ouvre (le cas échéant)."""

    speaker: str
    text: str
    wav_paths: list[Path]
    chapter: str | None = None


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True)
class Chapter:
    start: float
    end: float
    title: str


@dataclass
class Timeline:
    cues: list[Cue] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    # Ordre exact de concaténation : ("wav", chemin) ou ("silence", secondes).
    items: list[tuple[str, Path | float]] = field(default_factory=list)
    duration: float = 0.0


def wav_duration(path: Path) -> float:
    with contextlib.closing(wave.open(str(path), "rb")) as handle:
        return handle.getnframes() / float(handle.getframerate())


def build_timeline(
    turns: list[TimedTurn],
    *,
    duration_of: Callable[[Path], float] = wav_duration,
    pause_turn: float = PAUSE_TURN_SECONDS,
    pause_chapter: float = PAUSE_CHAPTER_SECONDS,
) -> Timeline:
    """Calcule cues, chapitres et ordre de concaténation (silences inclus)."""
    timeline = Timeline()
    clock = 0.0
    open_chapter: tuple[str, float] | None = None

    for index, turn in enumerate(turns):
        if turn.chapter is not None:
            if open_chapter is not None:
                timeline.chapters.append(Chapter(open_chapter[1], clock, open_chapter[0]))
            if index > 0:
                timeline.items.append(("silence", pause_chapter))
                clock += pause_chapter
            open_chapter = (turn.chapter, clock)
        elif index > 0:
            timeline.items.append(("silence", pause_turn))
            clock += pause_turn

        start = clock
        for path in turn.wav_paths:
            timeline.items.append(("wav", path))
            clock += duration_of(path)
        timeline.cues.append(Cue(start, clock, turn.speaker, turn.text))

    if open_chapter is not None:
        timeline.chapters.append(Chapter(open_chapter[1], clock, open_chapter[0]))
    timeline.duration = clock
    return timeline


def _vtt_time(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def render_vtt(cues: list[Cue], speaker_labels: dict[str, str] | None = None) -> str:
    """Transcript WebVTT synchronisé (un cue par réplique)."""
    labels = speaker_labels or {"HOST": "Animateur", "EXPERT": "Expert"}
    lines = ["WEBVTT", ""]
    for number, cue in enumerate(cues, start=1):
        text = cue.text.replace("-->", "->").replace("\n", " ").strip()
        lines += [str(number), f"{_vtt_time(cue.start)} --> {_vtt_time(cue.end)}", f"<v {labels.get(cue.speaker, cue.speaker)}>{text}", ""]
    return "\n".join(lines)


def _escape_metadata(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for char in ("=", ";", "#"):
        escaped = escaped.replace(char, f"\\{char}")
    return escaped.replace("\r", " ").replace("\n", " ")


def render_ffmetadata(title: str, chapters: list[Chapter]) -> str:
    lines = [";FFMETADATA1", f"title={_escape_metadata(title)}", "artist=Synthoria"]
    for chapter in chapters:
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={round(chapter.start * 1000)}",
            f"END={max(round(chapter.end * 1000), round(chapter.start * 1000) + 1)}",
            f"title={_escape_metadata(chapter.title)}",
        ]
    return "\n".join(lines) + "\n"


def _write_silence(path: Path, seconds: float, *, rate: int, channels: int, width: int) -> None:
    with contextlib.closing(wave.open(str(path), "wb")) as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(b"\x00" * (round(seconds * rate) * channels * width))


def _wav_params(path: Path) -> tuple[int, int, int]:
    with contextlib.closing(wave.open(str(path), "rb")) as handle:
        return handle.getframerate(), handle.getnchannels(), handle.getsampwidth()


def _list_entry(path: Path) -> str:
    return "file '" + str(path.resolve()).replace("'", "'\\''") + "'"


async def _run_ffmpeg(args: list[str], timeout: float = FFMPEG_TIMEOUT_SECONDS) -> None:
    try:
        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise AudioAssemblyError("ffmpeg est introuvable") from exc
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        raise AudioAssemblyError("ffmpeg a dépassé le délai imparti") from exc
    if process.returncode != 0:
        raise AudioAssemblyError(f"ffmpeg a échoué : {stderr.decode(errors='replace')[-500:]}")


async def _uniformize(paths: list[Path], work_dir: Path, ffmpeg: str) -> tuple[dict[Path, Path], tuple[int, int, int]]:
    """Rééchantillonne les WAV dont le format diffère du premier (voix de taux différents)."""
    target = _wav_params(paths[0])
    mapping: dict[Path, Path] = {}
    for path in dict.fromkeys(paths):
        if _wav_params(path) == target:
            continue
        converted = work_dir / f"uniform_{path.stem}.wav"
        await _run_ffmpeg([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-ar", str(target[0]), "-ac", str(target[1]), "-c:a", f"pcm_s{target[2] * 8}le", str(converted),
        ])
        mapping[path] = converted
    return mapping, target


async def assemble_mp3(
    timeline: Timeline,
    *,
    title: str,
    output_path: Path,
    work_dir: Path,
    bitrate: str = "96k",
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Concatène clips et silences, normalise (loudnorm −16 LUFS) et encode en MP3 mono avec chapitres."""
    wavs = [item for kind, item in timeline.items if kind == "wav"]
    if not wavs:
        raise AudioAssemblyError("Aucun clip audio à assembler")

    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    replacements, (rate, channels, width) = await _uniformize(wavs, work_dir, ffmpeg)  # type: ignore[arg-type]

    silences: dict[float, Path] = {}
    entries: list[str] = []
    for kind, value in timeline.items:
        if kind == "wav":
            entries.append(_list_entry(replacements.get(value, value)))  # type: ignore[arg-type]
        else:
            seconds = float(value)  # type: ignore[arg-type]
            if seconds not in silences:
                silences[seconds] = work_dir / f"silence_{round(seconds * 1000)}ms.wav"
                _write_silence(silences[seconds], seconds, rate=rate, channels=channels, width=width)
            entries.append(_list_entry(silences[seconds]))

    list_file = work_dir / "concat.txt"
    meta_file = work_dir / "meta.txt"
    list_file.write_text("\n".join(entries) + "\n", encoding="utf-8")
    meta_file.write_text(render_ffmetadata(title, timeline.chapters), encoding="utf-8")

    temporary = output_path.with_suffix(".part.mp3")
    await _run_ffmpeg([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-i", str(meta_file), "-map_metadata", "1", "-map_chapters", "1",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ac", "1", "-codec:a", "libmp3lame", "-b:a", bitrate, "-id3v2_version", "3",
        str(temporary),
    ])
    temporary.replace(output_path)
    return output_path
