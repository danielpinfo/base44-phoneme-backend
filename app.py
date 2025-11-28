# app.py
#
# FastAPI backend for Base44 using wav2vec2 phoneme models.
# - Accepts WAV audio via POST /phonemes
# - Resamples to 16kHz
# - Uses facebook/wav2vec2-lv-60-espeak-cv-ft to output phonetic labels (NOT words)
# - We avoid Wav2Vec2Processor (which is currently buggy) and instead use
#   FeatureExtractor + Phoneme Tokenizer directly.

import io

import torch
import torchaudio
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2ForCTC,
)

app = FastAPI(title="Base44 wav2vec2 Phoneme Backend")

TARGET_SR = 16000  # wav2vec2 expects 16kHz audio

# Default phoneme model (multi-language via espeak labels)
DEFAULT_MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"

MODEL_CONFIGS = {
    "default": DEFAULT_MODEL_NAME,
    "eng": DEFAULT_MODEL_NAME,
    "mul": DEFAULT_MODEL_NAME,
}

_feature_extractors = {}
_tokenizers = {}
_models = {}


def get_components(lang_key: str):
    """
    Get (feature_extractor, tokenizer, model_name, model) for the given language key.
    We explicitly construct the parts instead of using Wav2Vec2Processor to avoid
    the 'bool tokenizer' bug in recent transformers versions.
    """
    if lang_key not in MODEL_CONFIGS:
        lang_key = "default"

    model_name = MODEL_CONFIGS[lang_key]

    # Load feature extractor
    if model_name not in _feature_extractors:
        _feature_extractors[model_name] = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name
        )

    # Load phoneme tokenizer (requires phonemizer + espeak installed)
    if model_name not in _tokenizers:
        _tokenizers[model_name] = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
            model_name
        )

    # Load model
    if model_name not in _models:
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        model.eval()
        _models[model_name] = model

    return (
        _feature_extractors[model_name],
        _tokenizers[model_name],
        model_name,
        _models[model_name],
    )


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
async def phonemes(
    file: UploadFile = File(...),
    lang: str = "eng",  # logical language key: "eng", "mul", etc.
):
    """
    Accept a WAV file and return phoneme sequence using a wav2vec2 phoneme model.
    """
    if file.content_type not in (
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/vnd.wave",
        None,  # some browsers omit content-type, we can be lenient
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content-type {file.content_type}. Please upload a WAV file.",
        )

    try:
        wav_bytes = await file.read()
        waveform = load_and_resample_to_16k(wav_bytes)

        feature_extractor, tokenizer, model_name, model = get_components(lang_key=lang)

        with torch.no_grad():
            inputs = feature_extractor(
                waveform,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            )

            logits = model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            # Decode to a string of phonetic labels (NOT words)
            transcription = tokenizer.batch_decode(predicted_ids)[0]

        return JSONResponse(
            content={
                "phonemes": transcription,  # e.g. "b l iː s l i b uː s"
                "lang": lang,
                "model": model_name,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phoneme recognition failed: {e}")
