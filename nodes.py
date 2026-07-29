EMOTIONS = [
    "none", "elation", "amusement", "enthusiasm", "determination", "pride",
    "contentment", "affection", "relief", "contemplation", "confusion",
    "surprise", "awe", "longing", "arousal", "anger", "fear", "disgust",
    "bitterness", "sadness", "shame", "helplessness",
]

STYLES = ["none", "singing", "shouting", "whispering"]


class HiggsV3TagBuilder:
    """
    Prepends Higgs Audio v3 <|emotion:x|> / <|style:x|> control tags to text.

    Graydient can only offer real dropdown fields for widgets that are COMBO
    type on the node itself — HiggsV3Generate's `text` is a plain STRING, so
    emotion/style are exposed here as their own COMBO widgets (slot1/slot2)
    and spliced into the text before it reaches HiggsV3Generate.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello! This is Higgs Audio v3."}),
                "emotion": (EMOTIONS, {"default": "none"}),
                "style": (STYLES, {"default": "none"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_text",)
    FUNCTION = "build"
    CATEGORY = "Higgs v3 TTS"

    def build(self, text, emotion, style):
        prefix = ""
        if emotion != "none":
            prefix += f"<|emotion:{emotion}|>"
        if style != "none":
            prefix += f"<|style:{style}|>"
        return (prefix + text,)


NODE_CLASS_MAPPINGS = {
    "HiggsV3TagBuilder": HiggsV3TagBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiggsV3TagBuilder": "Higgs v3 Tag Builder",
}
