"""Backend fallback classification for Stage 2 inference."""

from transformers import PreTrainedModel


_FLASH_ATTENTION_CHECK_CODE = (
    PreTrainedModel._check_and_enable_flash_attn_2.__func__.__code__
)
_FLASH_ATTENTION_CHECK_MODULE = "transformers.modeling_utils"


def is_flash_attention_initialization_error(error: Exception) -> bool:
    """Return whether an exception originated in Transformers' FA2 check."""

    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        traceback = current.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if (
                frame.f_code is _FLASH_ATTENTION_CHECK_CODE
                and frame.f_globals.get("__name__")
                == _FLASH_ATTENTION_CHECK_MODULE
            ):
                return True
            traceback = traceback.tb_next

        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)

    return False


__all__ = ["is_flash_attention_initialization_error"]
