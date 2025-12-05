from phonemizer import phonemize

# Map our short language codes to phonemizer language tags
LANG_TO_PHONEMIZER = {
    "en": "en-us",
    "en-us": "en-us",
    "en-gb": "en-gb",
    "es": "es",
    "es-es": "es",
    "es-mx": "es",   # Spanish (Mexico) - phonemizer uses generic Spanish
    "fr": "fr-fr",
    "de": "de",
    "pt": "pt",
    "pt-br": "pt-br",
    "it": "it",
    "zh": "cmn",     # Mandarin Chinese
    "ja": "ja",
}


def normalize_lang_for_phonemizer(lang: str) -> str:
    """
    Take things like 'en', 'en-US', 'es-MX' and map them
    to what phonemizer expects, like 'en-us', 'es', etc.
    """
    if not lang:
        return "en-us"

    key = lang.strip().lower()
    # Cut off regional code if needed: "en-us" -> "en"
    short = key.split("-")[0]

    # Try full key first, then short key, then default
    return (
        LANG_TO_PHONEMIZER.get(key)
        or LANG_TO_PHONEMIZER.get(short)
        or "en-us"
    )


def get_ipa_for_text(text: str, lang: str) -> str:
    """
    Return canonical IPA for the target text using phonemizer.
    - text: the letter or word the user is supposed to say (e.g. "M", "hola")
    - lang: a language code like "en", "en-US", "es", "es-MX", etc.
    """
    if not text:
        return ""

    phon_lang = normalize_lang_for_phonemizer(lang)

    try:
        ipa = phonemize(
            text,
            language=phon_lang,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            punctuation_marks=';:,.!?¡¿—…„“”«»()-_',
        )
        return ipa.strip()
    except Exception as e:
        # If phonemizer fails for any reason, we fail gracefully
        # so the rest of the backend still works.
        print(f"[ipa_lookup] Phonemizer error for {text!r} ({phon_lang}): {e}")
        return ""
