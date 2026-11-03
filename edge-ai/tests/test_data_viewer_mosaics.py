from io import BytesIO
from pathlib import Path
import sys
import tarfile

from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUI_DIR = REPO_ROOT / "gui"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GUI_DIR))

from data_viewer_mosaics import export_tar_mosaics


def image_bytes(color, size=(2, 2), mode="RGB"):
    with Image.new(mode, size, color) as image:
        output = BytesIO()
        image.save(output, format="TIFF")
        return output.getvalue()


def tar_bytes(members, mode="w"):
    output = BytesIO()
    with tarfile.open(fileobj=output, mode=mode) as archive:
        directory = tarfile.TarInfo("images/")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, BytesIO(data))
    return output.getvalue()


class FakeDownloader:
    def __init__(self, data):
        self.data = data
        self.closed = False
        self.bytes_yielded = 0

    def chunks(self):
        for start in range(0, len(self.data), 512):
            chunk = self.data[start:start + 512]
            self.bytes_yielded += len(chunk)
            yield chunk

    def close(self):
        self.closed = True


class FakeBlob:
    def __init__(self, data):
        self.downloader = FakeDownloader(data)
        self.calls = []

    def download_blob(self, **kwargs):
        self.calls.append(kwargs)
        return self.downloader


def test_packs_native_size_images_and_stops_at_n_without_decoding_tail(tmp_path):
    payload = tar_bytes(
        [(f"{i}.tif", image_bytes(color)) for i, color in enumerate(
            ("red", "green", "blue", "yellow", "cyan", "magenta", "black", "white")
        )] + [(f"bad_{i}.tif", b"must not be decoded") for i in range(80)]
    )
    blob = FakeBlob(payload)
    messages = []
    snapshots = []

    result = export_tar_mosaics(
        blob, "sample.tar", tmp_path, 2, 4, 4,
        blob_size=len(payload), log_cb=messages.append,
        progress_cb=snapshots.append,
    )

    assert (result.saved, result.seen, result.included, result.skipped) == (2, 8, 8, 0)
    assert not result.exhausted
    assert not result.cancelled
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "composite_0001.png", "composite_0002.png",
    ]
    with Image.open(tmp_path / "composite_0001.png") as image:
        assert image.size == (4, 4)
        assert image.getpixel((0, 0)) == (255, 0, 0)
        assert image.getpixel((2, 0)) == (0, 128, 0)
        assert image.getpixel((0, 2)) == (0, 0, 255)
        assert image.getpixel((2, 2)) == (255, 255, 0)
    with Image.open(tmp_path / "composite_0002.png") as image:
        assert image.size == (4, 4)
        assert image.getpixel((0, 0)) == (0, 255, 255)
    assert blob.downloader.bytes_yielded < len(payload)
    assert blob.downloader.closed
    assert blob.calls == [{"max_concurrency": 1}]
    assert result.last_image == "7.tif"
    assert result.tar_offset == 512 + 8 * 1024
    assert snapshots[0].saved == 1
    assert snapshots[0].included == 4
    assert "Requested mosaic count reached" in messages[-1]
    assert "reached image 8" in messages[-1]
    assert f"TAR byte {result.tar_offset:,} of {len(payload):,}" in messages[-1]
    assert "%" in messages[-1]


@pytest.mark.parametrize("skip_every,kept,skipped", [(0, 6, 0), (2, 3, 3), (3, 4, 2), (1, 0, 6)])
def test_skips_every_sth_image_not_nonimage_members(tmp_path, skip_every, kept, skipped):
    members = [("notes.txt", b"not an image")]
    for index in range(1, 7):
        data = (
            b"skipped TIFFs must not be decoded"
            if skip_every and index % skip_every == 0
            else image_bytes("red")
        )
        members.extend([(f"{index}.TIFF", data), (f"note_{index}.txt", b"metadata")])
    blob = FakeBlob(tar_bytes(members))
    messages = []

    result = export_tar_mosaics(
        blob, "sample.tar", tmp_path, 10, 20, 20,
        skip_every=skip_every, log_cb=messages.append,
    )

    assert result.exhausted
    assert result.seen == 6
    assert result.included == kept
    assert result.skipped == skipped
    assert result.saved == (1 if kept else 0)
    assert "End of TAR" in " ".join(messages)
    assert blob.downloader.closed


def test_partial_canvas_at_eof_has_exact_dimensions_and_white_background(tmp_path):
    blob = FakeBlob(tar_bytes([("../../outside.TIFF", image_bytes(100, mode="L"))]))

    result = export_tar_mosaics(blob, "sample.tar", tmp_path, 5, 7, 5)

    assert result.saved == 1
    assert result.exhausted
    with Image.open(tmp_path / "composite_0001.png") as image:
        assert image.size == (7, 5)
        assert image.getpixel((0, 0)) == (100, 100, 100)
        assert image.getpixel((2, 0)) == (255, 255, 255)
        assert image.getpixel((0, 2)) == (255, 255, 255)
    assert list(tmp_path.iterdir()) == [tmp_path / "composite_0001.png"]


def test_oversized_images_shrink_without_distortion_and_report_lookahead(tmp_path):
    blob = FakeBlob(tar_bytes([
        ("wide.tif", image_bytes("red", size=(20, 10))),
        ("next.tif", image_bytes("blue", size=(4, 4))),
        ("tail.tif", b"must not be decoded"),
    ]))
    result = export_tar_mosaics(blob, "sample.tar", tmp_path, 1, 4, 4)

    assert result.saved == 1
    assert result.seen == 2
    assert result.included == 1
    assert result.last_image == "next.tif"
    with Image.open(tmp_path / "composite_0001.png") as image:
        assert image.size == (4, 4)
        assert image.getpixel((3, 1)) == (255, 0, 0)
        assert image.getpixel((0, 2)) == (255, 255, 255)


def test_overflow_starts_new_canvas_without_losing_lookahead(tmp_path):
    blob = FakeBlob(tar_bytes([
        (f"{i}.tif", image_bytes(color, size=(3, 3)))
        for i, color in enumerate(("red", "blue", "green"))
    ]))
    result = export_tar_mosaics(blob, "sample.tar", tmp_path, 5, 4, 4)

    assert (result.saved, result.seen, result.included) == (3, 3, 3)
    for index, color in enumerate(((255, 0, 0), (0, 0, 255), (0, 128, 0)), 1):
        with Image.open(tmp_path / f"composite_{index:04d}.png") as image:
            assert image.size == (4, 4)
            assert image.getpixel((0, 0)) == color


@pytest.mark.parametrize("mode,suffix", [("w:gz", ".tgz"), ("w:bz2", ".tar.bz2"), ("w:xz", ".tar.xz")])
def test_compressed_tars_report_uncompressed_position_not_false_percentage(tmp_path, mode, suffix):
    payload = tar_bytes([("one.tif", image_bytes("red"))], mode=mode)
    messages = []
    blob = FakeBlob(payload)

    result = export_tar_mosaics(
        blob, "sample" + suffix, tmp_path, 1, 2, 2,
        blob_size=len(payload), log_cb=messages.append,
    )

    assert result.saved == 1
    assert "uncompressed TAR byte" in messages[-1]
    assert "%" not in messages[-1]
    assert blob.downloader.closed


def test_cancellation_saves_current_partial_canvas(tmp_path):
    blob = FakeBlob(tar_bytes([(f"{i}.tif", image_bytes("red")) for i in range(5)]))
    checks = iter([False, True])
    messages = []

    result = export_tar_mosaics(
        blob, "sample.tar", tmp_path, 10, 10, 10,
        is_cancelled=lambda: next(checks), log_cb=messages.append,
    )

    assert result.cancelled
    assert not result.exhausted
    assert (result.saved, result.seen, result.included) == (1, 1, 1)
    assert "Stopped:" in messages[-1]
    assert blob.downloader.closed


def test_no_images_does_not_create_blank_mosaics(tmp_path):
    blob = FakeBlob(tar_bytes([("notes.txt", b"notes")]))
    messages = []

    result = export_tar_mosaics(
        blob, "sample.tar", tmp_path, 1, 2, 2, log_cb=messages.append,
    )

    assert result.exhausted
    assert result.saved == 0
    assert list(tmp_path.iterdir()) == []
    assert "WARNING no images" in messages[-1]


def test_decode_failure_is_explicit_and_closes_stream(tmp_path):
    blob = FakeBlob(tar_bytes([("broken.tif", b"not a TIFF")]))
    with pytest.raises(ValueError, match="Could not decode image broken.tif"):
        export_tar_mosaics(blob, "sample.tar", tmp_path, 1, 2, 2)
    assert blob.downloader.closed
    assert list(tmp_path.iterdir()) == []


def test_download_failure_is_not_reported_as_success(tmp_path):
    class FailingBlob:
        def download_blob(self, **kwargs):
            raise OSError("download failed")

    with pytest.raises(OSError, match="download failed"):
        export_tar_mosaics(FailingBlob(), "sample.tar", tmp_path, 1, 2, 2)


def test_existing_output_is_not_overwritten(tmp_path):
    existing = tmp_path / "composite_0001.png"
    existing.write_bytes(b"previous export")
    blob = FakeBlob(tar_bytes([("one.tif", image_bytes("red"))]))

    with pytest.raises(FileExistsError):
        export_tar_mosaics(blob, "sample.tar", tmp_path, 1, 2, 2)

    assert existing.read_bytes() == b"previous export"
    assert blob.downloader.closed


@pytest.mark.parametrize("overrides", [
    {"count": 0}, {"width": 0}, {"height": -1}, {"skip_every": -1},
    {"width": 1.5}, {"skip_every": True},
])
def test_invalid_export_settings_fail_before_downloading(tmp_path, overrides):
    blob = FakeBlob(b"")
    settings = dict(count=1, width=2, height=2, skip_every=0)
    settings.update(overrides)

    with pytest.raises(ValueError):
        export_tar_mosaics(blob, "sample.tar", tmp_path, **settings)

    assert blob.calls == []
