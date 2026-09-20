import asyncio
import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import httpx
import pytest

from app.core.exceptions import AudioAssemblyError, TTSInvalidVoiceError, TTSUnavailableError
from app.services.podcast.audio_assembler import (
    Chapter,
    Cue,
    TimedTurn,
    assemble_mp3,
    build_timeline,
    render_ffmetadata,
    render_vtt,
    wav_duration,
)
from app.services.podcast.tts_client import PiperHTTPEngine, cache_key, synthesize_cached, synthesize_many

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent")


def make_wav(path: Path, seconds: float, *, rate: int = 22050, freq: float = 440.0) -> Path:
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate))) for i in range(int(seconds * rate))
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return path


def wav_bytes(tmp_path: Path, seconds: float = 0.1) -> bytes:
    return make_wav(tmp_path / "x.wav", seconds).read_bytes()


# ─── Client TTS ──────────────────────────────────────────────


async def _no_sleep(_: float) -> None:
    return None


def _engine(handler, retries: int = 3) -> PiperHTTPEngine:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://piper:5000")
    return PiperHTTPEngine("http://piper:5000", client=client, max_retries=retries, sleep=_no_sleep)


@pytest.mark.asyncio
async def test_piper_returns_wav_and_sends_text_and_voice(tmp_path):
    audio = wav_bytes(tmp_path)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=audio)

    assert await _engine(handler).synthesize("Bonjour", "fr_FR-siwis-medium") == audio
    assert seen == {"text": "Bonjour", "voice": "fr_FR-siwis-medium"}


@pytest.mark.asyncio
async def test_piper_retries_5xx_then_succeeds(tmp_path):
    audio = wav_bytes(tmp_path)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, content=audio)

    assert await _engine(handler).synthesize("x", "v") == audio
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_piper_gives_up_after_max_retries():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500)

    with pytest.raises(TTSUnavailableError):
        await _engine(handler, retries=3).synthesize("x", "v")
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_piper_4xx_is_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404)

    with pytest.raises(TTSInvalidVoiceError):
        await _engine(handler).synthesize("x", "voix-inconnue")
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_piper_timeout_is_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ReadTimeout("lent")

    with pytest.raises(TTSUnavailableError):
        await _engine(handler, retries=2).synthesize("x", "v")
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_piper_rejects_non_wav_payload():
    def handler(request):
        return httpx.Response(200, content=b"<html>oops</html>")

    with pytest.raises(TTSUnavailableError):
        await _engine(handler).synthesize("x", "v")


@pytest.mark.asyncio
async def test_cache_hit_avoids_second_synthesis(tmp_path):
    audio = wav_bytes(tmp_path)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, content=audio)

    engine = _engine(handler)
    cache = tmp_path / "cache"
    first = await synthesize_cached(engine, cache, "v", "Bonjour")
    second = await synthesize_cached(engine, cache, "v", "Bonjour")
    other_voice = await synthesize_cached(engine, cache, "w", "Bonjour")

    assert first == second and first != other_voice
    assert calls["n"] == 2
    assert first.name == f"{cache_key('piper', 'v', 'Bonjour')}.wav"
    assert not list(cache.glob("*.tmp"))


@pytest.mark.asyncio
async def test_synthesize_many_keeps_order_and_bounds_concurrency(tmp_path):
    audio = wav_bytes(tmp_path)
    state = {"running": 0, "max": 0}

    class SlowEngine:
        name = "slow"

        async def synthesize(self, text, voice):
            state["running"] += 1
            state["max"] = max(state["max"], state["running"])
            await asyncio.sleep(0.01)
            state["running"] -= 1
            return audio

        async def close(self):
            return None

    progress: list[tuple[int, int]] = []

    async def on_progress(done, total):
        progress.append((done, total))

    items = [("v", f"texte {i}") for i in range(8)]
    paths = await synthesize_many(SlowEngine(), tmp_path / "c", items, concurrency=2, on_progress=on_progress)

    assert [p.name for p in paths] == [f"{cache_key('slow', 'v', t)}.wav" for _, t in items]
    assert state["max"] <= 2
    assert progress[-1] == (8, 8)


# ─── Chronologie, VTT, métadonnées ───────────────────────────


def _turns(tmp_path: Path) -> list[TimedTurn]:
    a = make_wav(tmp_path / "a.wav", 1.0)
    b = make_wav(tmp_path / "b.wav", 2.0)
    return [
        TimedTurn("HOST", "Bienvenue.", [a], chapter="Introduction"),
        TimedTurn("EXPERT", "Voici le sujet.", [b, a]),
        TimedTurn("HOST", "Passons à la suite.", [a], chapter="Notion 1"),
    ]


def test_timeline_cues_chapters_and_silences(tmp_path):
    timeline = build_timeline(_turns(tmp_path), pause_turn=0.5, pause_chapter=1.0)

    assert [(c.start, c.end) for c in timeline.cues] == [(0.0, 1.0), (1.5, 4.5), (5.5, 6.5)]
    assert [(c.title, c.start, c.end) for c in timeline.chapters] == [("Introduction", 0.0, 4.5), ("Notion 1", 5.5, 6.5)]
    assert timeline.duration == pytest.approx(6.5)
    kinds = [kind for kind, _ in timeline.items]
    assert kinds == ["wav", "silence", "wav", "wav", "silence", "wav"]
    assert [v for k, v in timeline.items if k == "silence"] == [0.5, 1.0]


def test_timeline_duration_matches_sum_of_real_wavs(tmp_path):
    turns = _turns(tmp_path)
    timeline = build_timeline(turns)
    expected = sum(wav_duration(p) for t in turns for p in t.wav_paths) + 0.35 + 0.9
    assert timeline.duration == pytest.approx(expected)


def test_render_vtt_format():
    vtt = render_vtt([Cue(0.0, 1.5, "HOST", "Bonjour --> monde"), Cue(3661.25, 3662.0, "EXPERT", "Salut")])
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:01.500" in vtt
    assert "01:01:01.250 --> 01:01:02.000" in vtt
    assert "<v Animateur>Bonjour -> monde" in vtt   # "-->" neutralisé dans le texte
    assert "<v Expert>Salut" in vtt


def test_ffmetadata_escapes_special_characters_and_newlines():
    meta = render_ffmetadata("Titre = piège; #1\nligne 2", [Chapter(0.0, 2.0, "A=B")])
    assert meta.startswith(";FFMETADATA1\n")
    assert "title=Titre \\= piège\\; \\#1 ligne 2" in meta
    assert "START=0" in meta and "END=2000" in meta and "title=A\\=B" in meta


# ─── Assemblage ffmpeg (ignoré sans ffmpeg) ──────────────────


@needs_ffmpeg
@pytest.mark.asyncio
async def test_assemble_mp3_with_mixed_sample_rates(tmp_path):
    turns = [
        TimedTurn("HOST", "Un.", [make_wav(tmp_path / "a.wav", 1.0, rate=22050)], chapter="Un"),
        TimedTurn("EXPERT", "Deux.", [make_wav(tmp_path / "b.wav", 1.0, rate=16000, freq=660)]),
    ]
    timeline = build_timeline(turns)
    output = tmp_path / "out" / "podcast.mp3"

    result = await assemble_mp3(timeline, title="Test", output_path=output, work_dir=tmp_path / "work")

    assert result == output and output.stat().st_size > 1000
    assert not list(output.parent.glob("*.part.mp3"))
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        probe = json.loads(subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration:format_tags=title", "-of", "json", str(output)],
            capture_output=True, text=True, check=True,
        ).stdout)
        assert float(probe["format"]["duration"]) == pytest.approx(timeline.duration, abs=0.6)
        assert probe["format"]["tags"]["title"] == "Test"


@pytest.mark.asyncio
async def test_assemble_mp3_requires_clips(tmp_path):
    with pytest.raises(AudioAssemblyError):
        await assemble_mp3(build_timeline([]), title="t", output_path=tmp_path / "o.mp3", work_dir=tmp_path / "w")


@pytest.mark.asyncio
async def test_assemble_mp3_reports_missing_ffmpeg(tmp_path):
    timeline = build_timeline([TimedTurn("HOST", "x", [make_wav(tmp_path / "a.wav", 0.2)])])
    with pytest.raises(AudioAssemblyError, match="introuvable"):
        await assemble_mp3(
            timeline, title="t", output_path=tmp_path / "o.mp3", work_dir=tmp_path / "w", ffmpeg="ffmpeg-inexistant-xyz"
        )
