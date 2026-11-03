import math
from typing import Literal


DEFAULT_SIZE_THRESHOLD_PX: Literal["inf"] = "inf"


def parse_size_threshold(value=DEFAULT_SIZE_THRESHOLD_PX) -> int | Literal["inf"]:
    """Return a positive pixel count or the JSON-safe disabled value ``"inf"``."""
    if value is None or str(value).strip() == "":
        return DEFAULT_SIZE_THRESHOLD_PX
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Size threshold must be a positive integer or inf, got {value!r}") from exc
    if threshold == math.inf and str(value).strip().lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return DEFAULT_SIZE_THRESHOLD_PX
    if isinstance(value, bool) or not math.isfinite(threshold) or threshold <= 0 or not threshold.is_integer():
        raise ValueError(f"Size threshold must be a positive integer or inf, got {value!r}")
    return int(threshold)


def image_meets_size_threshold(image_bytes: bytes, size_threshold_px: int | Literal["inf"]) -> bool:
    if size_threshold_px == DEFAULT_SIZE_THRESHOLD_PX:
        return True

    import pyvips

    # Opening is lazy: width and height come from the header, without decoding pixels.
    image = pyvips.Image.new_from_buffer(image_bytes, "", access="sequential")
    return image.width >= size_threshold_px and image.height >= size_threshold_px
