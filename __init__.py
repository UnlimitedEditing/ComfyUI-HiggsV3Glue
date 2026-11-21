import os
import sys


def _prefer_canonical_higgs_pack():
    """
    Graydient/ComfyUI-Manager sometimes installs a SECOND, stale copy of the
    Higgs v3 node pack as custom_nodes/Higgs-V3-TTS-ComfyUI (hyphen, capital V)
    next to the one we ask for, custom_nodes/Higgs_v3-TTS-ComfyUI. Both claim
    the same node class names and whichever loads last runs, in an order that
    varies per machine (os.listdir). The stale copy crashes the Whisper step
    under DynamicVRAM ("'_ComfyConv1d' object has no attribute
    '_combined_linear'", seen 2026-11-21). Make the canonical copy always win:
    delete the stale folder if it has not loaded yet, and if it already has,
    put the canonical copy's classes back into ComfyUI's registry.
    Never raises: a failure here must not stop the glue nodes from loading.
    """
    try:
        import shutil

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stale_dir = os.path.join(root, "Higgs-V3-TTS-ComfyUI")
        good_dir = os.path.join(root, "Higgs_v3-TTS-ComfyUI")
        if not os.path.isdir(good_dir) or not os.path.isdir(stale_dir):
            return

        shutil.rmtree(stale_dir, ignore_errors=True)
        print("[HiggsV3Glue] removed stale duplicate node pack Higgs-V3-TTS-ComfyUI")

        def _find(suffix):
            for key, mod in list(sys.modules.items()):
                if mod is not None and key.replace("\\", "/").rstrip("/").endswith(suffix):
                    return mod
            return None

        stale_mod = _find("/Higgs-V3-TTS-ComfyUI") or _find("Higgs-V3-TTS-ComfyUI")
        if stale_mod is None:
            return  # never loaded; deleting the folder was enough

        import nodes as comfy_nodes

        stale_map = getattr(stale_mod, "NODE_CLASS_MAPPINGS", None) or getattr(
            stale_mod, "_NODE_CLASS_MAPPINGS", {}
        )
        good_mod = _find("/Higgs_v3-TTS-ComfyUI") or _find("Higgs_v3-TTS-ComfyUI")
        good_map = None
        if good_mod is not None:
            good_map = getattr(good_mod, "NODE_CLASS_MAPPINGS", None) or getattr(
                good_mod, "_NODE_CLASS_MAPPINGS", None
            )
        if good_map:
            comfy_nodes.NODE_CLASS_MAPPINGS.update(good_map)
            print("[HiggsV3Glue] re-pointed Higgs node classes at Higgs_v3-TTS-ComfyUI")
        else:
            # canonical copy loads later and will register itself; just drop the stale classes
            for name, cls in list(stale_map.items()):
                if comfy_nodes.NODE_CLASS_MAPPINGS.get(name) is cls:
                    del comfy_nodes.NODE_CLASS_MAPPINGS[name]
    except Exception as e:  # noqa: BLE001
        print(f"[HiggsV3Glue] stale-pack guard skipped: {e!r}")


_prefer_canonical_higgs_pack()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
