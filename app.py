# app.py
#
# FastAPI backend for Base44 using a stable wav2vec2 CTC model.
# - Accepts WAV audio via POST /phonemes
# - Resamples to 16kHz
# - Uses facebook/wav2vec2-base-960h to output character sequences
# - Converts that to a space-separated sequence of characters,
#   e.g., "bleesleboos" -> "b l e e s l e b o o s"

import io

import torch
import torchaudio
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

app = FastAPI(title="Base44 wav2vec2 Backend")

TARGET_SR = 16000  # wav2vec2 expects 16kHz audio

# Stable CTC model
MODEL_NAME = "facebook/wav2vec2-base-960h"

# Load processor + model once at startup
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()


def load_and_resample_to_16k(wav_bytes: bytes) -> torch.Tensor:
    """
    Load audio from bytes, convert to mono float32 tensor, resample to 16kHz.
    """
    with io.BytesIO(wav_bytes) as buf:
        audio, sr = sf.read(buf, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # average stereo to mono

    waveform = torch.from_numpy(audio)

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=TARGET_SR
        )

    return waveform


@app.post("/phonemes")
async def phonemes(file: UploadFile = File(...)):
    """
    Accept a WAV file and return a "phoneme-like" character sequence.
    Response example:
    {
      "phonemes": "b l e e s l e b o o s",
      "raw_transcription": "bleesleboos",
      "model": "facebook/wav2vec2-base-960h"
    }
    """
    if file.content_type not in (
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/vnd.wave",
        None,  # some browsers omit it
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content-type {file.content_type}. Please upload a WAV file.",
        )

    try:
        wav_bytes = await file.read()
        waveform = load_and_resample_to_16k(wav_bytes)

        with torch.no_grad():
            inputs = processor(
                waveform,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            )
            input_values = inputs.input_values

            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            decoded = processor.batch_decode(predicted_ids)
            transcription = decoded[0].strip().lower() if decoded else ""

            # Keep only letters and apostrophes
            cleaned = "".join(ch for ch in transcription if ch.isalpha() or ch == "'")

            # "bleesleboos" -> "b l e e s l e b o o s"
            spaced_chars = " ".join(list(cleaned)) if cleaned else ""

        return JSONResponse(
            content={
                "phonemes": spaced_chars,
                "raw_transcription": transcription,
                "model": MODEL_NAME,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phoneme recognition failed: {e}")

