import io
import numpy as np
import torch
import torchaudio
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Import IPA helper from phoneme/ipa_lookup.py
from phoneme.ipa_lookup import get_ipa_for_text

# ----------------------------------------------------------------------
# Torch settings (helps on small CPU instances like Railway)
# ----------------------------------------------------------------------
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ----------------------------------------------------------------------
# App bootstrap
# ----------------------------------------------------------------------
app = FastAPI(title="Base44 Multi-Language wav2vec2 Backend")

# CORS so Base44 preview / editor can call this directly from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # you can tighten this later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGET_SR = 16000
MAX_SECONDS_SENTENCE = 10
MAX_SAMPLES_SENTENCE = TARGET_SR * MAX_SECONDS_SENTENCE

# ----------------------------------------------------------------------
# wav2vec2 models per language
# ----------------------------------------------------------------------
LANG_MODELS: dict[str, str] = {
    "en": "facebook/wav2vec2-base-960h",                         # English
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",       # Spanish
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",        # French
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",        # German
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",       # Italian
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",    # Portuguese
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn", # Chinese (Mandarin)
    "ja": "ku-nlp/wav2vec2-large-xlsr-japanese",                 # Japanese
}

# ----------------------------------------------------------------------
# Practice word lists (for future use by the frontend)
# ----------------------------------------------------------------------
PRACTICE_WORDS: dict[str, list[dict]] = {
    "en": [
        {"id": "hello",      "text": "hello",      "translation": "hello",      "hint": "Basic greeting"},
        {"id": "water",      "text": "water",      "translation": "water",      "hint": "Common noun"},
        {"id": "thank_you",  "text": "thank you",  "translation": "thank you",  "hint": "Polite phrase"},
    ],
    "es": [
        {"id": "hola",       "text": "hola",       "translation": "hello",      "hint": "Saludo básico"},
        {"id": "gracias",    "text": "gracias",    "translation": "thank you",  "hint": "Frase de cortesía"},
        {"id": "agua",       "text": "agua",       "translation": "water",      "hint": "Sustantivo común"},
    ],
    "fr": [
        {"id": "bonjour",    "text": "bonjour",    "translation": "hello",      "hint": "Salutation"},
        {"id": "merci",      "text": "merci",      "translation": "thank you",  "hint": "Expression de politesse"},
        {"id": "eau",        "text": "eau",        "translation": "water",      "hint": "Nom courant"},
    ],
    "de": [
        {"id": "hallo",      "text": "hallo",      "translation": "hello",      "hint": "Begrüßung"},
        {"id": "danke",      "text": "danke",      "translation": "thank you",  "hint": "Höfliche Wendung"},
        {"id": "wasser",     "text": "wasser",     "translation": "water",      "hint": "Häufiges Substantiv"},
    ],
    "it": [
        {"id": "ciao",       "text": "ciao",       "translation": "hi / bye",   "hint": "Saluto informale"},
        {"id": "grazie",     "text": "grazie",     "translation": "thank you",  "hint": "Frase di cortesia"},
        {"id": "acqua",      "text": "acqua",      "translation": "water",      "hint": "Sostantivo comune"},
    ],
    "pt": [
        {"id": "ola",        "text": "olá",        "translation": "hello",      "hint": "Saudação básica"},
        {"id": "obrigado",   "text": "obrigado",   "translation": "thank you (m.)", "hint": "Frase de cortesia"},
        {"id": "agua",       "text": "água",       "translation": "water",      "hint": "Substantivo comum"},
    ],
    "zh": [
        {"id": "nihao",      "text": "你好",        "translation": "hello",      "hint": "问候语"},
        {"id": "xiexie",     "text": "谢谢",       "translation": "thank you",  "hint": "礼貌表达"},
        {"id": "shui",       "text": "水",         "translation": "water",      "hint": "常用名词"},
    ],
    "ja": [
        {"id": "konnichiwa", "text": "こんにちは",  "translation": "hello",      "hint": "あいさつ"},
        {"id": "arigatou",   "text": "ありがとう",  "translation": "thank you",  "hint": "ていねいな表現"},
        {"id": "mizu",       "text": "水",         "translation": "water",      "hint": "よく使う名詞"},
    ],
}

# ----------------------------------------------------------------------
# Lazy loading of wav2vec2 models
# ----------------------------------------------------------------------
processors: dict[str, Wav2Vec2Processor] = {}
models: dict[str, Wav2Vec2ForCTC] = {}


def get_model_and_processor(lang: str):
    """
    Return (processor, model) for the given language.
    Models are loaded lazily on first use.
    """
    if lang not in LANG_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported language '{lang}'")

    if lang not in processors:
        model_name = LANG_MODELS[lang]
        proc = Wav2Vec2Processor.from_pretrained(model_name)
        mdl = Wav2Vec2ForCTC.from_pretrained(model_name)
        mdl.to("cpu")
        mdl.eval()

        processors[lang] = proc
        models[lang] = mdl

    return processors[lang], models[lang]


# ----------------------------------------------------------------------
# Audio handling
# ----------------------------------------------------------------------
def trim_silence(
    waveform: torch.Tensor,
    sr: int,
    threshold: float = 0.01,
) -> torch.Tensor:
    """
    Remove leading/trailing silence using a simple amplitude threshold.
    """
    if waveform.ndim != 1:
        waveform = waveform.view(-1)

    audio_np = waveform.numpy()
    energy = np.abs(audio_np)

    voiced = np.where(energy > threshold)[0]
    if len(voiced) == 0:
        # All silence — avoid empty tensors
        return waveform[: sr // 10]

    start = int(voiced[0])
    end = int(voiced[-1]) + 1

    margin = int(0.1 * sr)
    start = max(0, start - margin)
    end = min(len(audio_np), end + margin)

    return waveform[start:end]


def load_and_resample_to_16k(wav_bytes: bytes) -> torch.Tensor:
    """
    Load WAV bytes, mono-ize, trim silence, resample to 16k, cap to max length.
    """
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


# ----------------------------------------------------------------------
# Utility: health check
# ----------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ----------------------------------------------------------------------
# Utility: list languages
# ----------------------------------------------------------------------
@app.get("/languages")
async def list_languages():
    """
    Frontend can call this to build the language picker.
    """
    return [
        {"code": "en", "nativeName": "English",        "englishName": "English",           "flag": "🇺🇸"},
        {"code": "es", "nativeName": "Español",        "englishName": "Spanish",           "flag": "🇪🇸"},
        {"code": "fr", "nativeName": "Français",       "englishName": "French",            "flag": "🇫🇷"},
        {"code": "de", "nativeName": "Deutsch",        "englishName": "German",            "flag": "🇩🇪"},
        {"code": "it", "nativeName": "Italiano",       "englishName": "Italian",           "flag": "🇮🇹"},
        {"code": "pt", "nativeName": "Português",      "englishName": "Portuguese",        "flag": "🇵🇹"},
        {"code": "zh", "nativeName": "中文 (普通话)",   "englishName": "Chinese (Mandarin)", "flag": "🇨🇳"},
        {"code": "ja", "nativeName": "日本語",          "englishName": "Japanese",          "flag": "🇯🇵"},
    ]


# ----------------------------------------------------------------------
# Utility: practice words
# ----------------------------------------------------------------------
@app.get("/practice-words")
async def get_practice_words(
    lang: str = Query("en", description="Language code: en, es, fr, de, it, pt, zh, ja"),
):
    """
    Return practice words for a given language.
    """
    if lang not in PRACTICE_WORDS:
        raise HTTPException(status_code=400, detail=f"Unsupported language '{lang}'")

    return {
        "lang": lang,
        "words": PRACTICE_WORDS[lang],
    }


# ----------------------------------------------------------------------
# Core: /phonemes – wav2vec2 + optional Phonemizer IPA with fallback
# ----------------------------------------------------------------------
@app.post("/phonemes")
async def phonemes(
    file: UploadFile = File(...),
    lang: str = Query("en", description="Language code: en, es, fr, de, it, pt, zh, ja"),
):
    """
    Accept a WAV file + language code and return a phoneme-oriented view.

    DESIGN:
    - wav2vec2 gives us a rough transcription (e.g. "PAH", "POH", "tenna").
    - We *try* to get IPA via phonemizer.
    - If IPA is available:
        * phoneme_list = IPA units
        * phonemes = IPA units joined with "-"
    - If IPA is NOT available:
        * phoneme_list = simple syllable tokens from wav2vec2 transcription
        * phonemes = those tokens joined with "-"
    - No attempt is made to guess dictionary words for scoring.
    """
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

    processor, model = get_model_and_processor(lang)

    try:
        # 1) Read + preprocess audio
        wav_bytes = await file.read()
        waveform = load_and_resample_to_16k(wav_bytes)

        # 2) Run wav2vec2 → rough transcription (characters/letters)
        with torch.no_grad():
            inputs = processor(
                waveform,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            )
            logits = model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            decoded = processor.batch_decode(predicted_ids)
            transcription = decoded[0].strip() if decoded else ""

        # 3) Try Phonemizer: get IPA for the transcription
        ipa_str = get_ipa_for_text(transcription, lang) if transcription else ""
        ipa_units = ipa_str.split() if ipa_str else []

        # 4) Build the phoneme_list with a robust fallback
        if ipa_units:
            # Preferred path: true IPA units from phonemizer
            phoneme_list = ipa_units
        else:
            # Fallback: use wav2vec2's transcription as syllable-like units
            fallback = transcription.lower().strip()
            phoneme_list = fallback.split() if fallback else []

        phonemes_joined = "-".join(phoneme_list) if phoneme_list else ""

        return JSONResponse(
            content={
                "lang": lang,
                "backend": "wav2vec2+phonemizer(fallback)",
                "ipa": ipa_str or None,          # full IPA string (or null)
                "ipa_units": ipa_units,          # may be empty
                "phoneme_list": phoneme_list,    # canonical list of units
                "phonemes": phonemes_joined,     # e.g. "poh" or "poh-ah"
                "model": LANG_MODELS[lang],
                "raw_transcription": transcription,  # wav2vec2's guess (debug)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phoneme recognition failed: {e}")


# ----------------------------------------------------------------------
# IPA lookup endpoint (text-only, no audio)
# ----------------------------------------------------------------------
@app.get("/expected_ipa")
async def expected_ipa(
    text: str = Query(..., description="The target text (letter or word)"),
    lang: str = Query("en", description="Language code: en, es, fr, de, it, pt, zh, ja"),
):
    """
    Return canonical IPA for the given target text + language,
    using the Phonemizer helper (get_ipa_for_text).
    This does NOT do any audio processing – it's just for 'what SHOULD it sound like?'.
    """
    ipa = get_ipa_for_text(text, lang)
    return {
        "text": text,
        "lang": lang,
        "ipa": ipa,
    }
