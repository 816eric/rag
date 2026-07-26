import asyncio
import io
import os
import wave

import numpy as np
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

router = APIRouter(prefix="/api/voice", tags=["voice"])

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voice_models")
_KOKORO_MODEL_PATH = os.path.join(_MODELS_DIR, "kokoro-v1.0.int8.onnx")
_KOKORO_VOICES_PATH = os.path.join(_MODELS_DIR, "voices-v1.0.bin")

_whisper_model = None
_kokoro_tts = None


def _get_whisper():
    """Lazy-loaded so the backend doesn't pay Whisper's load cost until voice is actually used."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


def _get_kokoro():
    global _kokoro_tts
    if _kokoro_tts is None:
        from kokoro_onnx import Kokoro

        _kokoro_tts = Kokoro(_KOKORO_MODEL_PATH, _KOKORO_VOICES_PATH)
    return _kokoro_tts


class TranscribeResponse(BaseModel):
    text: str


class SynthesizeRequest(BaseModel):
    text: str


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    model = _get_whisper()

    def _run():
        # faster-whisper decodes whatever format is handed to it (webm/opus from
        # the browser's MediaRecorder, wav, mp3, ...) via PyAV internally, so no
        # manual format conversion is needed here.
        segments, _ = model.transcribe(
            io.BytesIO(audio_bytes),
            language="en",
            # Silero VAD strips any leftover silence/noise before decoding, and
            # disabling previous-text conditioning stops a low-confidence
            # segment from seeding a runaway repeated-phrase loop - both are
            # well-documented Whisper hallucination triggers on quiet/silent
            # audio, which is exactly what a short utterance clip can contain.
            vad_filter=True,
            condition_on_previous_text=False,
            # Single temperature + greedy decoding, matching the reference
            # implementation: real speedup for a small accuracy trade on
            # unclear audio, which matters more for interactive use than
            # batch transcription accuracy.
            temperature=0.0,
            beam_size=1,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    text = await asyncio.to_thread(_run)
    return TranscribeResponse(text=text)


@router.post("/synthesize")
async def synthesize(body: SynthesizeRequest):
    if not body.text.strip():
        return Response(status_code=204)

    kokoro = _get_kokoro()

    def _run():
        return kokoro.create(body.text, voice="af_heart", speed=1.0, lang="en-us")

    samples, sample_rate = await asyncio.to_thread(_run)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)

    # WAV wrapper so the browser/Flutter audio player can decode it without
    # needing to know the raw sample format out of band.
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return Response(content=buf.getvalue(), media_type="audio/wav")
