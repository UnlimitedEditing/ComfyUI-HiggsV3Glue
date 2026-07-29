import os

VOICE_PRESETS = {
    "jake": (
        "Jake_Rivers_intro.mp3",
        "A male speaker with an energetic and slightly dramatic tone, conveying "
        "passion and enthusiasm, especially about sports.",
    ),
    "chloe": (
        "Chloe_Adams_intro.mp3",
        "A friendly and clear female voice with an engaging, informative tone "
        "and a standard American accent.",
    ),
    "eleanor": (
        "Eleanor_Reed_intro.mp3",
        "A calm, articulate female voice with a clear, professional Standard "
        "American English accent, perfect for educational or explanatory content.",
    ),
    "marcus": (
        "Marcus_Webb_intro.mp3",
        "A male speaker with an enthusiastic, confident, and slightly "
        "professorial American delivery.",
    ),
    "nora": (
        "Nora_Vance_intro.mp3",
        "A female speaker with a calm, clear, and narrative voice, using "
        "standard American English.",
    ),
    "oliver": (
        "Oliver_Grant_intro.mp3",
        "A calm, articulate male voice with a thoughtful American accent, "
        "well-suited for explanatory or reflective content.",
    ),
}

# Boson's public demo clips for their hosted API's named presets (chloe/eleanor/
# jake/marcus/nora/oliver) -- not the actual server-side reference clips, but
# real audio of each named voice, usable locally as HiggsV3VoiceClone reference
# audio to approximate the preset via zero-shot cloning.
VOICE_SAMPLE_BASE_URL = "https://docs.boson.ai/public/audio/higgs-audio-tts-voices"

HIGGS_SAMPLE_RATE = 24000


def _decode_audio_via_ffmpeg(path: str, sample_rate: int = HIGGS_SAMPLE_RATE):
    """
    Decode an audio file to a mono waveform via an ffmpeg subprocess.

    torchaudio.load() routes through torchcodec's AudioDecoder on recent
    torchaudio versions, which is broken on at least some Graydient instances
    (`_AudioDecoder() takes no arguments`) -- confirmed on a live g1_embedded
    run 2026-07-29. ffmpeg is already a hard dependency of this project's
    video pipelines, so shelling out to it directly sidesteps torchcodec
    entirely rather than hoping torchaudio's backend dispatch recovers.

    Raw PCM output (not WAV) is used deliberately: ffmpeg writes a WAV header
    with a placeholder max-int frame count when streaming to a pipe (it can't
    know the final size upfront), which makes Python's `wave` module report
    a bogus multi-hour duration if you trust the header instead of the actual
    byte count.
    """
    import subprocess
    import numpy as np
    import torch

    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {proc.stderr.decode(errors='replace')}")

    arr = np.frombuffer(proc.stdout, dtype="<i2").astype("float32") / 32768.0
    waveform = torch.from_numpy(arr.copy()).unsqueeze(0)  # [channels=1, samples]
    return waveform, sample_rate


class HiggsV3VoicePreset:
    """
    Loads one of six named Higgs v3 demo voices as AUDIO for HiggsV3VoiceClone's
    reference_audio input. Reads the concept_mapping-staged local copy if present;
    falls back to downloading directly, since concept_mapping is not guaranteed
    to persist across Graydient machine classes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice": (list(VOICE_PRESETS.keys()), {"default": "jake"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("reference_audio",)
    FUNCTION = "load"
    CATEGORY = "Higgs v3 TTS"

    def load(self, voice):
        import folder_paths

        filename, _description = VOICE_PRESETS[voice]
        local_path = os.path.join(folder_paths.models_dir, "higgsv3tts", "voices", filename)

        if not os.path.isfile(local_path):
            import urllib.request
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(f"{VOICE_SAMPLE_BASE_URL}/{filename}", local_path)

        waveform, sample_rate = _decode_audio_via_ffmpeg(local_path)
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)


NODE_CLASS_MAPPINGS = {
    "HiggsV3VoicePreset": HiggsV3VoicePreset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiggsV3VoicePreset": "Higgs v3 Voice Preset",
}
