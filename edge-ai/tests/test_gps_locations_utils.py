import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GUI_DIR = os.path.join(REPO_ROOT, "gui")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, GUI_DIR)

from gps_locations_utils import (
    build_gps_plot_html,
    extract_gps_metadata_from_tiff,
    gps_from_exif_tags,
    is_tar_blob,
    merge_processed_summaries,
)


def _tag(*values):
    return SimpleNamespace(values=list(values))


def _processed_summary(**overrides):
    return {
        "container": "raw-images",
        "source_tar_blob": "sample.tar",
        "model_blob": "model-one.pt",
        "sample_ratio_requested": 0.25,
        "sample_percent_requested": 25.0,
        "generated_utc": "2026-10-27T08:00:00Z",
        "class_counts": {"copepod": 7},
        **overrides,
    }


class GPSLocationsUtilsTests(unittest.TestCase):
    def test_is_tar_blob_recognizes_common_case_insensitive_suffixes(self):
        self.assertTrue(is_tar_blob("2025/recording.TAR"))
        self.assertTrue(is_tar_blob("recording.tar.gz"))
        self.assertTrue(is_tar_blob("recording.tgz"))
        self.assertTrue(is_tar_blob("recording.tar.bz2"))
        self.assertFalse(is_tar_blob("recording.tif"))
        self.assertFalse(is_tar_blob("recording.tar.csv"))

    def test_gps_from_exif_tags_includes_signed_coordinates_and_datetime(self):
        tags = {
            "GPS GPSLatitude": _tag(50, 30, 0),
            "GPS GPSLatitudeRef": _tag("S"),
            "GPS GPSLongitude": _tag(1, 15, 30),
            "GPS GPSLongitudeRef": _tag("W"),
            "Image DateTime": _tag("2025:01:02 10:20:30"),
        }

        self.assertEqual(
            gps_from_exif_tags(tags),
            {
                "latitude": -50.5,
                "longitude": -1.2583333333333333,
                "datetime": "2025-01-02T10:20:30",
            },
        )

    def test_gps_from_exif_tags_ignores_missing_or_invalid_gps(self):
        self.assertIsNone(gps_from_exif_tags({}))
        self.assertIsNone(
            gps_from_exif_tags(
                {
                    "GPS GPSLatitude": _tag(91, 0, 0),
                    "GPS GPSLatitudeRef": _tag("N"),
                    "GPS GPSLongitude": _tag(0, 0, 0),
                    "GPS GPSLongitudeRef": _tag("E"),
                }
            )
        )

    def test_extract_gps_metadata_from_tiff_uses_exifread(self):
        tags = {
            "GPS GPSLatitude": _tag(51, 30, 0),
            "GPS GPSLatitudeRef": _tag("N"),
            "GPS GPSLongitude": _tag(0, 10, 0),
            "GPS GPSLongitudeRef": _tag("E"),
            "EXIF DateTimeOriginal": _tag("2025:01:02 10:20:30"),
        }
        fake_exifread = SimpleNamespace(
            process_file=lambda stream, details: tags
        )
        with patch.dict(sys.modules, {"exifread": fake_exifread}):
            result = extract_gps_metadata_from_tiff(b"tiff-bytes")

        self.assertEqual(
            result,
            {
                "latitude": 51.5,
                "longitude": 1 / 6,
                "datetime": "2025-01-02T10:20:30",
            },
        )

    def test_build_gps_plot_html_includes_datetime_filters_and_location_data(self):
        html = build_gps_plot_html(
            [
                {
                    "latitude": 51.5,
                    "longitude": 0.16,
                    "datetime": "2025-01-02T10:20:30",
                    "container": "images",
                    "tar": "sample.tar",
                    "member": "one.tif",
                }
            ]
        )

        self.assertIn('type="datetime-local"', html)
        self.assertIn("Apply filter", html)
        self.assertIn("Clear filter", html)
        self.assertIn("2025-01-02T10:20:30", html)
        self.assertIn("Plotly.restyle", html)

    def test_merge_processed_summaries_matches_raw_tar_and_keeps_models_separate(self):
        location = {
            "container": "raw-images",
            "tar": "2025-07-19/0816.tar",
            "member": "Images/one.tif",
            "latitude": 51.5,
            "longitude": 0.16,
        }
        summaries = [
            (
                "runs/run-a/per_tar/2025-07-19/0816/summary.json",
                {
                    "container": "raw-images",
                    "source_tar_blob": "2025-07-19/0816.tar",
                    "model_blob": "model-one.pt",
                    "sample_percent_requested": 100.0,
                    "class_counts": {"copepod": 7},
                },
            ),
            (
                "runs/run-b/per_tar/2025-07-19/0816/summary.json",
                {
                    "container": "raw-images",
                    "source_tar_blob": "2025-07-19/0816.tar",
                    "model_blob": "model-two.pt",
                    "sample_percent_requested": 25.0,
                    "class_counts": {"fish_larvae": 3},
                },
            ),
            (
                "runs/run-c/per_tar/2025-07-19/0816/summary.json",
                {
                    "container": "different-raw-images",
                    "source_tar_blob": "2025-07-19/0816.tar",
                    "model_blob": "model-one.pt",
                    "class_counts": {"detritus": 4},
                },
            ),
        ]

        merged = merge_processed_summaries([location], summaries)

        self.assertEqual(set(merged), {"model-one.pt", "model-two.pt"})
        self.assertEqual(
            merged["model-one.pt"][0]["summary"]["class_counts"],
            {"copepod": 7},
        )
        self.assertEqual(
            merged["model-two.pt"][0]["summary"]["class_counts"],
            {"fish_larvae": 3},
        )
        self.assertIs(merged["model-one.pt"][0]["location"], location)

    def test_merge_processed_summaries_drops_oldest_runs_before_location_merge(self):
        locations = [
            {"container": "raw-images", "tar": "sample.tar", "member": member}
            for member in ("one.tif", "two.tif")
        ]
        summaries = [
            (
                "runs/z-old/summary.json",
                _processed_summary(
                    generated_utc="2026-10-25T08:00:00Z",
                    class_counts={"copepod": 100},
                ),
            ),
            ("runs/a-new/summary.json", _processed_summary()),
            (
                "runs/m-middle/summary.json",
                _processed_summary(
                    generated_utc="2026-10-26T08:00:00Z",
                    class_counts={"copepod": 50},
                ),
            ),
        ]

        for ordered in (summaries, summaries[::-1], summaries[1:] + summaries[:1]):
            with self.subTest(order=[name for name, _ in ordered]):
                messages = []
                merged = merge_processed_summaries(
                    locations, ordered, log_cb=messages.append
                )

                records = merged["model-one.pt"]
                self.assertEqual(len(records), 2)
                self.assertEqual(
                    [record["summary_blob"] for record in records],
                    ["runs/a-new/summary.json"] * 2,
                )
                self.assertTrue(all(
                    record["summary"]["class_counts"] == {"copepod": 7}
                    for record in records
                ))
                self.assertEqual(len(messages), 2)
                for message in messages:
                    self.assertIn("WARNING", message)
                    self.assertIn("oldest run was dropped before merge", message)
                    self.assertIn("keeping runs/a-new/summary.json", message)
                self.assertTrue(any("z-old/summary.json" in m for m in messages))
                self.assertTrue(any("m-middle/summary.json" in m for m in messages))

    def test_merge_processed_summaries_keeps_each_distinct_deduplication_key(self):
        summaries = [
            _processed_summary(),
            _processed_summary(source_tar_blob="different.tar"),
            _processed_summary(model_blob="model-two.pt"),
            _processed_summary(sample_ratio_requested=0.5),
            _processed_summary(sample_percent_requested=50.0),
            _processed_summary(container="different-raw-images"),
            _processed_summary(size_threshold_px=64),
        ]
        locations = [
            {"container": "raw-images", "tar": "sample.tar"},
            {"container": "raw-images", "tar": "different.tar"},
            {"container": "different-raw-images", "tar": "sample.tar"},
        ]
        messages = []

        merged = merge_processed_summaries(
            locations,
            [(f"runs/{index}/summary.json", summary)
             for index, summary in enumerate(summaries)],
            log_cb=messages.append,
        )

        self.assertEqual(sum(map(len, merged.values())), len(summaries))
        self.assertEqual(messages, [])

    def test_merge_processed_summaries_compares_utc_times_not_timestamp_strings(self):
        summaries = [
            ("runs/a/summary.json", _processed_summary()),
            (
                "runs/z/summary.json",
                _processed_summary(generated_utc="2026-10-27T09:00:00+02:00"),
            ),
        ]

        with patch("builtins.print") as warning:
            merged = merge_processed_summaries(
                [{"container": "raw-images", "tar": "sample.tar"}], summaries
            )

        self.assertEqual(
            merged["model-one.pt"][0]["summary_blob"], "runs/a/summary.json"
        )
        warning.assert_called_once()
        self.assertIn("oldest run was dropped before merge", warning.call_args.args[0])

    def test_merge_processed_summaries_breaks_timestamp_ties_deterministically(self):
        summaries = [
            ("runs/a/summary.json", _processed_summary()),
            (
                "runs/z/summary.json",
                _processed_summary(generated_utc="2026-10-27T08:00:00"),
            ),
        ]
        location = {"container": "raw-images", "tar": "sample.tar"}

        for ordered in (summaries, summaries[::-1]):
            merged = merge_processed_summaries(
                [location], ordered, log_cb=lambda message: None
            )
            self.assertEqual(
                merged["model-one.pt"][0]["summary_blob"], "runs/z/summary.json"
            )

    def test_merge_processed_summaries_warns_about_unknown_duplicate_age(self):
        for timestamp in (None, "invalid"):
            with self.subTest(timestamp=timestamp):
                messages = []
                merged = merge_processed_summaries(
                    [{"container": "raw-images", "tar": "sample.tar"}],
                    [
                        ("runs/a/summary.json", _processed_summary()),
                        (
                            "runs/z/summary.json",
                            _processed_summary(generated_utc=timestamp),
                        ),
                    ],
                    log_cb=messages.append,
                )

                self.assertEqual(
                    merged["model-one.pt"][0]["summary_blob"], "runs/a/summary.json"
                )
                self.assertTrue(any("generated_utc" in m for m in messages))
                self.assertTrue(any("treating it as oldest" in m for m in messages))

    def test_merge_processed_summaries_deduplicates_even_without_a_location_match(self):
        messages = []
        merged = merge_processed_summaries(
            [],
            [
                ("runs/new/summary.json", _processed_summary()),
                (
                    "runs/old/summary.json",
                    _processed_summary(generated_utc="2026-10-26T08:00:00Z"),
                ),
            ],
            log_cb=messages.append,
        )

        self.assertEqual(merged, {})
        self.assertEqual(len(messages), 1)
        self.assertIn("runs/old/summary.json", messages[0])

    def test_build_gps_plot_html_includes_processed_filters_and_class_counts(self):
        location = {
            "container": "raw-images",
            "tar": "sample.tar",
            "member": "one.tif",
            "latitude": 51.5,
            "longitude": 0.16,
        }
        merged = merge_processed_summaries(
            [location],
            [
                (
                    "runs/run/per_tar/sample/summary.json",
                    {
                        "container": "raw-images",
                        "source_tar_blob": "sample.tar",
                        "model_blob": "model-one.pt",
                        "sample_percent_requested": 100.0,
                        "class_counts": {"copepod": 7},
                    },
                )
            ],
        )

        html = build_gps_plot_html([location], merged, processed_summary_count=1)

        self.assertIn('id="model-filter"', html)
        self.assertIn('id="sample-percent-filter"', html)
        self.assertIn('id="size-threshold-filter"', html)
        self.assertIn('<option value="inf">inf</option>', html)
        self.assertIn(
            "result.sample_percent_requested === Number(samplePercent)", html
        )
        self.assertIn('id="container-filter"', html)
        self.assertIn('id="class-filter"', html)
        self.assertIn('value="copepod" checked', html)
        self.assertIn("model-one.pt", html)
        self.assertIn("copepod", html)
        self.assertIn("matched summary record(s)", html)
        self.assertIn("Selected class counts", html)
        self.assertIn('"sizemode":"area"', html)
        self.assertIn('"marker.size": [countPoints.map(item => item.count)]', html)
        self.assertNotIn("<table", html)
        self.assertNotIn('id="processed-results"', html)

    def test_build_gps_plot_html_normalizes_sample_percentages_to_numbers(self):
        location = {
            "container": "raw-images",
            "tar": "sample.tar",
            "latitude": 51.5,
            "longitude": 0.16,
        }
        percentages = (1, 1.0, "1", "1.0", 0, "0.0", 2.5, "2.5", None)
        merged = {
            "model-one.pt": [
                {
                    "summary": _processed_summary(sample_percent_requested=value),
                    "location": location,
                }
                for value in percentages
            ]
        }

        html = build_gps_plot_html([location], merged)
        payload = html.split("const modelResults = ", 1)[1].split(";\n", 1)[0]
        results = json.loads(payload)["model-one.pt"]

        for value, result in zip(percentages, results):
            with self.subTest(percentage=value):
                actual = result["sample_percent_requested"]
                if value is None:
                    self.assertIsNone(actual)
                else:
                    self.assertIsInstance(actual, float)
                    self.assertEqual(actual, float(value))
        for value in (0.0, 1.0, 2.5):
            self.assertEqual(
                html.count(f'<option value="{value}">{value}</option>'), 1
            )
        self.assertNotIn('<option value="None">', html)

    def test_merge_normalizes_disabled_and_numeric_size_thresholds(self):
        location = {"container": "raw-images", "tar": "sample.tar"}
        summaries = [
            ("runs/old/summary.json", _processed_summary()),
            ("runs/new/summary.json", _processed_summary(
                size_threshold_px="inf", generated_utc="2026-10-28T08:00:00Z",
            )),
            ("runs/small-old/summary.json", _processed_summary(size_threshold_px="64.0")),
            ("runs/small-new/summary.json", _processed_summary(
                size_threshold_px=64, generated_utc="2026-10-28T08:00:00Z",
            )),
            ("runs/large/summary.json", _processed_summary(size_threshold_px=128)),
        ]
        merged = merge_processed_summaries([location], summaries, log_cb=lambda message: None)
        self.assertEqual(
            {record["summary_blob"] for record in merged["model-one.pt"]},
            {"runs/new/summary.json", "runs/small-new/summary.json", "runs/large/summary.json"},
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is required to exercise HTML filters")
    def test_html_size_filter_updates_locations_counts_and_clear(self):
        locations = [
            {"container": "raw-images", "tar": tar, "latitude": latitude, "longitude": 0.16}
            for tar, latitude in (("sample.tar", 51.5), ("other.tar", 52.0))
        ]
        merged = merge_processed_summaries(
            locations,
            [
                ("runs/unfiltered", _processed_summary(class_counts={"copepod": 7})),
                ("runs/small", _processed_summary(size_threshold_px="64.0", class_counts={"copepod": 3})),
                ("runs/large", _processed_summary(
                    source_tar_blob="other.tar", size_threshold_px=128,
                    class_counts={"copepod": 1},
                )),
            ],
        )
        html = build_gps_plot_html(locations, merged)
        self.assertEqual(html.count('<option value="64">64</option>'), 1)
        script = html.rsplit("<script>", 1)[1].split("</script>", 1)[0]
        harness = """
          const assert = require("node:assert/strict");
          const updates = [];
          const elements = {};
          for (const id of ["gps-locations-plot", "date-start", "date-end",
            "result-count", "filter-message", "model-filter", "sample-percent-filter",
            "size-threshold-filter", "container-filter", "apply-filter", "clear-filter",
            "class-filter", "class-selection-count"]) {
            elements[id] = {value: "", handlers: {},
              addEventListener(event, callback) { this.handlers[event] = callback; }};
          }
          const checkbox = {value: "copepod", checked: true};
          elements["class-filter"].querySelectorAll = () => [checkbox];
          const document = {getElementById: id => elements[id] || null};
          const Plotly = {restyle: (plot, data, traces) => updates.push({data, traces})};
        """
        assertions = """
          function check(latitudes, counts, matched) {
            assert.deepEqual(updates.at(-2).data.lat[0], latitudes);
            assert.deepEqual(updates.at(-1).data["marker.size"][0], counts);
            assert.ok(elements["result-count"].textContent.includes(
              `${matched} matched summary record(s)`));
          }
          check([51.5, 52.0], [10, 1], 3);
          elements["size-threshold-filter"].value = "64";
          applyFilter();
          check([51.5], [3], 1);
          elements["size-threshold-filter"].value = "inf";
          applyFilter();
          check([51.5], [7], 1);
          elements["size-threshold-filter"].value = "128";
          applyFilter();
          check([52.0], [1], 1);
          elements["sample-percent-filter"].value = "100";
          applyFilter();
          check([], [], 0);
          elements["clear-filter"].handlers.click();
          assert.equal(elements["size-threshold-filter"].value, "");
          check([51.5, 52.0], [10, 1], 3);
        """
        result = subprocess.run(
            ["node", "-e", harness + script + assertions],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_build_gps_plot_html_rejects_empty_locations(self):
        with self.assertRaisesRegex(ValueError, "no GPS locations"):
            build_gps_plot_html([])


if __name__ == "__main__":
    unittest.main()
