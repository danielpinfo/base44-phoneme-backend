# app.py
#
# FastAPI backend for Base44 using wav2vec2-lv-60-espeak-cv-ft.
# This follows the official Hugging Face usage:
#   - Wav2Vec2Processor.from_pretrained(...)
#   - Wav2Vec2ForCTC.from_pretrained(...)
#   - processor(...).input_values
#   - processor.batch_decode(...)
#
# No custom tokenizer wiring, no phoneme tokenizer class – we let
# the official processor handle it.

import io

import torch
import torchaudio
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

app = FastAPI(title="Base44 wav2vec2 Phoneme Backend")

TARGET_SR = 16000  # wav2vec2 expects 16kHz audio

MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"

# Load processor + model once at startup (as in the Hugging Face example)
# https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft :contentReference[oaicite:1]{index=1}
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()


def load_and_resample_to_16k(wav_bytes: bytes) -> torch.Tensor:
    """
    Load audio from bytes, convert to mono float32 tensor, resample to 16kHz.
    """
    with io.BytesIO(wav_bytes) as buf:
        audio, sr = sf.read(buf, dtype="float32")

    # If stereo, convert to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    waveform = torch.from_numpy(audio)

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=TARGET_SR
        )

    return waveform


@app.post("/phonemes")
async def phonemes(file: UploadFile = File(...)):
    """
    Accept a WAV file and return a phoneme sequence using wav2vec2-lv-60-espeak-cv-ft.
    """
    # Be lenient about content-type – some browsers omit it
    if file.content_type not in (
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/vnd.wave",
        None,
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content-type {file.content_type}. Please upload a WAV file.",
        )

    try:
        wav_bytes = await file.read()
        waveform = load_and_resample_to_16k(wav_bytes)

        # Hugging Face example pattern:
        # input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values
        with torch.no_grad():
            inputs = processor(
                waveform,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            )
            input_values = inputs.input_values

            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            # This returns a list of strings; we take the first one
            # Example from model card:
            # ['m ɪ s t ɚ k w ɪ l t ɚ ...'] :contentReference[oaicite:2]{index=2}
            transcription_list = processor.batch_decode(predicted_ids)
            transcription = transcription_list[0] if transcription_list else ""

        return JSONResponse(
            content={
                "phonemes": transcription,
                "model": MODEL_NAME,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phoneme recognition failed: {e}")
