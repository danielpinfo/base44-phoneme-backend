import io
import numpy as np
import torch
import torchaudio
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Limit PyTorch CPU thread usage (helps on small Railway instances)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

app = FastAPI(title="Base44 wav2vec2 Backend")

TARGET_SR = 16000

# Max voiced audio duration we’ll process (seconds)
# Enough for full sentences, but prevents very long clips.
MAX_SECONDS_SENTENCE = 10
MAX_SAMPLES_SENTENCE = TARGET_SR * MAX_SECONDS_SENTENCE

# Stable CTC model that outputs characters
MODEL_NAME = "facebook/wav2vec2-base-960h"

# Load processor + model once at startup
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.to("cpu")
model.eval()


def trim_silence(waveform: torch.Tensor, sr: int, threshold: float = 0.01) -> torch.Tensor:
    """
    Remove leading and trailing silence using a simple amplitude-based threshold.
    """
    if waveform.ndim != 1:
        waveform = waveform.view(-1)

    audio_np = waveform.numpy()
    energy = np.abs(audio_np)

    # Non-silent indices
    voiced = np.where(energy > threshold)[0]
    if len(voiced) == 0:
        # All silence — return a very short slice so model doesn’t break
        return waveform[: sr // 10]

    start = int(voiced[0])
    end = int(voiced[-1]) + 1

    # Add a small margin (0.1 sec) at both ends
    margin = int(0.1 * sr)
    start = max(0, start - margin)
    end = min(len(audio_np), end + margin)

    trimmed = waveform[start:end]
    return trimmed


def load_and_resample_to_16k(wav_bytes: bytes) -> torch.Tensor:
    """
    Load WAV bytes, ensure mono, trim silence, resample to 16k,
    and cap to max sentence length.
    """
    # Load audio
    with io.BytesIO(wav_bytes) as buf:
        audio, sr = sf.read(buf, dtype="float32")

    # Stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    waveform = torch.from_numpy(audio)

    # 1. Trim silence at original sample rate
    waveform = trim_silence(waveform, sr)

    # 2. Resample to 16k
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=TARGET_SR
        )

    # 3. Cap to max length (10 seconds of audio at 16k)
    if waveform.shape[0] > MAX_SAMPLES_SENTENCE:
        waveform = waveform[:MAX_SAMPLES_SENTENCE]

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

            logits = model(inputs.input_values).logits
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


