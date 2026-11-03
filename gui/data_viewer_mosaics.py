from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

from PIL import Image

from gui.tar_streaming import iter_tar_members_from_blob


IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass
class MosaicProgress:
    saved: int = 0
    seen: int = 0
    included: int = 0
    skipped: int = 0
    tar_offset: int = 0
    last_image: str = ""
    exhausted: bool = False
    cancelled: bool = False


def describe_mosaic_progress(progress, tar_name, blob_size):
    counts = (
        f"{progress.saved:,} mosaic(s); reached image {progress.seen:,} "
        f"({progress.included:,} included, {progress.skipped:,} skipped)"
    )
    if tar_name.lower().endswith(".tar"):
        position = f"TAR byte {progress.tar_offset:,}"
        if blob_size:
            position += (
                f" of {blob_size:,} "
                f"({100 * progress.tar_offset / blob_size:.1f}%)"
            )
    else:
        position = f"uncompressed TAR byte {progress.tar_offset:,}"
    return f"{counts}; {position}; last image: {progress.last_image or '(none)'}"


def export_tar_mosaics(
    blob_client,
    tar_name,
    output_directory,
    count,
    width,
    height,
    skip_every=0,
    blob_size=None,
    log_cb=None,
    progress_cb=None,
    is_cancelled=None,
):
    """Pack streamed images into fixed-size PNGs, dropping every Sth image."""
    if any(type(value) is not int or value < 1 for value in (count, width, height)):
        raise ValueError("Mosaic count, width and height must be positive integers.")
    if type(skip_every) is not int or skip_every < 0:
        raise ValueError("Skip interval must be a non-negative integer.")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    progress = MosaicProgress()
    canvas = None
    x = y = row_height = canvas_images = 0

    def report(message=None):
        if log_cb and message:
            log_cb(message)
        if progress_cb:
            progress_cb(replace(progress))

    def save_canvas():
        nonlocal canvas, canvas_images, x, y, row_height
        filename = output_directory / f"composite_{progress.saved + 1:04d}.png"
        with filename.open("xb") as output:
            canvas.save(output, format="PNG")
        canvas.close()
        canvas = None
        canvas_images = x = y = row_height = 0
        progress.saved += 1
        report(
            f"Saved {filename}: "
            f"{describe_mosaic_progress(progress, tar_name, blob_size)}"
        )

    members = iter_tar_members_from_blob(
        blob_client, tar_name, log_cb=log_cb, raise_on_open_error=True
    )
    try:
        for member, extracted in members:
            try:
                if is_cancelled and is_cancelled():
                    progress.cancelled = True
                    break
                progress.tar_offset = (
                    member.offset_data + ((member.size + 511) // 512) * 512
                )
                if not member.name.lower().endswith(IMAGE_SUFFIXES):
                    continue
                progress.seen += 1
                progress.last_image = member.name
                if skip_every and progress.seen % skip_every == 0:
                    progress.skipped += 1
                    if progress.seen % 1000 == 0:
                        report(describe_mosaic_progress(progress, tar_name, blob_size))
                    continue
                payload = extracted.read()
            finally:
                extracted.close()
            try:               
                with Image.open(BytesIO(payload)) as source:
                    w, h = source.size
                    aspect_ratio = max(w / h, h / w)
                    if aspect_ratio > 10:
                        progress.skipped += 1
                        continue
                    image = source.convert("RGB")                                
            except (OSError, ValueError) as exc:
                raise ValueError(f"Could not decode image {member.name}: {exc}") from exc
            try:
                image.thumbnail((width, height), Image.Resampling.LANCZOS)
                image_width, image_height = image.size
                if x + image_width > width:
                    x = 0
                    y += row_height
                    row_height = 0
                if canvas_images and y + image_height > height:
                    save_canvas()
                    if progress.saved == count:
                        break
                if canvas is None:
                    canvas = Image.new("RGB", (width, height), "white")
                canvas.paste(image, (x, y))
                x += image_width
                row_height = max(row_height, image_height)
                canvas_images += 1
                progress.included += 1
                if x == width and y + row_height == height:
                    save_canvas()
                    if progress.saved == count:
                        break
                elif progress.seen % 1000 == 0:
                    report(describe_mosaic_progress(progress, tar_name, blob_size))
            finally:
                image.close()
        else:
            progress.exhausted = True

        if canvas_images:
            save_canvas()
        reason = (
            "Stopped" if progress.cancelled
            else "End of TAR" if progress.exhausted
            else "Requested mosaic count reached"
        )
        report(
            f"{reason}: {describe_mosaic_progress(progress, tar_name, blob_size)}. "
            f"Requested {count:,} mosaic(s)."
        )
        if not progress.included and log_cb:
            log_cb("WARNING no images were included; no mosaics were written.")
        return progress
    finally:
        members.close()
        if canvas is not None:
            canvas.close()
