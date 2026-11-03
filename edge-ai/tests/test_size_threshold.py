import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pyvips
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gui"))

from size_threshold import image_meets_size_threshold, parse_size_threshold


@pytest.mark.parametrize("value,expected", [
    (None, "inf"), ("", "inf"), ("inf", "inf"), (" INF ", "inf"),
    (float("inf"), "inf"), (64, 64), ("64", 64), ("64.0", 64), (64.0, 64),
])
def test_parse_size_threshold(value, expected):
    actual = parse_size_threshold(value)
    assert actual == expected
    assert json.loads(json.dumps(actual, allow_nan=False)) == expected


@pytest.mark.parametrize("value", ["nan", float("nan"), "-inf", "1e309", 0, -1, 0.5, "invalid", True])
def test_parse_size_threshold_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer or inf"):
        parse_size_threshold(value)


def test_disabled_size_threshold_does_not_open_image():
    with patch.object(pyvips.Image, "new_from_buffer") as open_image:
        assert image_meets_size_threshold(b"not even an image", "inf")
    open_image.assert_not_called()


@pytest.mark.parametrize("dimensions,expected", [
    ((63, 100), False),
    ((100, 63), False),
    ((63, 63), False),
    ((64, 64), True),
    ((64, 100), True),
    ((100, 64), True),
])
def test_size_threshold_reads_original_tiff_dimensions(dimensions, expected):
    buffer = BytesIO()
    Image.new("L", dimensions).save(buffer, format="TIFF")
    with patch.object(pyvips.Image, "write_to_memory", side_effect=AssertionError("decoded pixels")):
        assert image_meets_size_threshold(buffer.getvalue(), 64) is expected
