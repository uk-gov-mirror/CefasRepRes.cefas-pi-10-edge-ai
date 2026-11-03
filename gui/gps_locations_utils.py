from datetime import datetime, timezone
from io import BytesIO
import html  # DEPRECATED: legacy HTML export.
import json  # DEPRECATED: legacy HTML export.
import math

from gui.size_threshold import parse_size_threshold


TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


def is_tar_blob(blob_name: str) -> bool:
    return str(blob_name or "").lower().endswith(TAR_SUFFIXES)


def _tag_values(tags, name):
    tag = tags.get(name)
    return getattr(tag, "values", None) if tag is not None else None


def _decimal_coordinate(values, reference, positive_ref, negative_ref, limit):
    if values is None or len(values) != 3 or reference is None:
        return None

    try:
        degrees, minutes, seconds = (float(value) for value in values)
        ref = (
            reference.decode("ascii", errors="ignore")
            if isinstance(reference, bytes)
            else str(reference)
        ).strip().upper()
    except (TypeError, ValueError, ZeroDivisionError):
        return None

    if not all(math.isfinite(value) for value in (degrees, minutes, seconds)):
        return None
    if degrees < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return None
    if ref not in (positive_ref, negative_ref):
        return None

    decimal = degrees + minutes / 60 + seconds / 3600
    if decimal > limit:
        return None
    return -decimal if ref == negative_ref else decimal


def _exif_datetime(tags):
    for name in (
        "Image DateTime",
        "EXIF DateTimeOriginal",
        "EXIF DateTimeDigitized",
    ):
        tag = tags.get(name)
        if tag is None:
            continue

        value = str(tag).strip()

        try:
            parsed = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                continue

        return parsed.isoformat(timespec="seconds")

    return None


def gps_from_exif_tags(tags):
    latitude_values = _tag_values(tags, "GPS GPSLatitude")
    longitude_values = _tag_values(tags, "GPS GPSLongitude")
    latitude_ref_values = _tag_values(tags, "GPS GPSLatitudeRef")
    longitude_ref_values = _tag_values(tags, "GPS GPSLongitudeRef")
    if not latitude_ref_values or not longitude_ref_values:
        return None

    latitude = _decimal_coordinate(
        latitude_values, latitude_ref_values[0], "N", "S", 90
    )
    longitude = _decimal_coordinate(
        longitude_values, longitude_ref_values[0], "E", "W", 180
    )
    if latitude is None or longitude is None:
        return None
    if latitude == 0 and longitude == 0:
        return None
    return {
        "latitude": latitude,
        "longitude": longitude,
        "datetime": _exif_datetime(tags),
    }


def extract_gps_metadata_from_tiff(image_bytes: bytes):
    import exifread

    tags = exifread.process_file(BytesIO(image_bytes), details=False)
    return gps_from_exif_tags(tags)


def _summary_generated_utc(blob_name, summary, log_cb):
    value = summary.get("generated_utc")
    try:
        generated = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        log_cb(
            f"WARNING: {blob_name} has missing or invalid generated_utc={value!r}; "
            "cannot determine duplicate run age, treating it as oldest "
            "and using the summary blob name to break ties."
        )
        return datetime.min.replace(tzinfo=timezone.utc)
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    return generated.astimezone(timezone.utc)


def merge_processed_summaries(locations, summaries, log_cb=None):
    """Keep the newest duplicate runs, then join TAR locations by model."""
    if log_cb is None:
        log_cb = print
    locations_by_tar = {}
    for location in locations:
        key = (location.get("container"), location.get("tar"))
        if all(isinstance(value, str) and value for value in key):
            locations_by_tar.setdefault(key, []).append(location)

    summaries_by_key = {}
    for blob_name, summary in summaries:
        if not isinstance(summary, dict):
            continue
        model_blob = str(summary.get("model_blob") or "").strip()
        key = (summary.get("container"), summary.get("source_tar_blob"))
        if not model_blob or not all(
            isinstance(value, str) and value for value in key
        ):
            continue
        duplicate_key = (
            *key,
            model_blob,
            summary.get("sample_ratio_requested"),
            summary.get("sample_percent_requested"),
            parse_size_threshold(summary.get("size_threshold_px")),
        )
        summaries_by_key.setdefault(duplicate_key, []).append((blob_name, summary))

    retained_summaries = []
    for matches in summaries_by_key.values():
        if len(matches) > 1:
            matches.sort(key=lambda item: (
                _summary_generated_utc(item[0], item[1], log_cb), item[0]
            ))
            newest_blob, newest_summary = matches[-1]
            for blob_name, summary in matches[:-1]:
                log_cb(
                    f"WARNING: oldest run was dropped before merge: {blob_name} "
                    f"(generated_utc={summary.get('generated_utc')!r}); "
                    f"keeping {newest_blob} "
                    f"(generated_utc={newest_summary.get('generated_utc')!r})."
                )
        retained_summaries.append(matches[-1])

    merged_by_model = {}
    for blob_name, summary in retained_summaries:
        model_blob = str(summary.get("model_blob") or "").strip()
        key = (summary.get("container"), summary.get("source_tar_blob"))
        for location in locations_by_tar.get(key, ()):
            merged_by_model.setdefault(model_blob, []).append({
                "model_blob": model_blob,
                "summary_blob": blob_name,
                "summary": summary,
                "location": location,
            })

    for records in merged_by_model.values():
        records.sort(key=lambda record: record["summary_blob"])
    return dict(sorted(merged_by_model.items()))


# DEPRECATED: Retained for HTML export compatibility; use GPSPlotWindow in the GUI.
# DEPRECATED: Templates use adjacent literals so every source line can carry this flag.
def _select_options(values):  # DEPRECATED
    options = ['<option value="">All</option>']  # DEPRECATED
    for value in sorted(values, key=str):  # DEPRECATED
        text = str(value)  # DEPRECATED
        options.append(  # DEPRECATED
            f'<option value="{html.escape(text, quote=True)}">'  # DEPRECATED
            f'{html.escape(text)}</option>'  # DEPRECATED
        )  # DEPRECATED
    return "".join(options)  # DEPRECATED
# DEPRECATED
# DEPRECATED
def _class_checkbox_options(values):  # DEPRECATED
    return "".join(  # DEPRECATED
        f'<label class="class-option"><input type="checkbox" '  # DEPRECATED
        f'value="{html.escape(str(value), quote=True)}" checked>'  # DEPRECATED
        f'{html.escape(str(value))}</label>'  # DEPRECATED
        for value in sorted(values, key=str)  # DEPRECATED
    )  # DEPRECATED
# DEPRECATED
# DEPRECATED
def build_gps_plot_html(locations, merged_results=None, processed_summary_count=0):  # DEPRECATED
    import plotly.graph_objects as go  # DEPRECATED
    import plotly.io as pio  # DEPRECATED
# DEPRECATED
    points = [  # DEPRECATED
        {  # DEPRECATED
            "latitude": float(location["latitude"]),  # DEPRECATED
            "longitude": float(location["longitude"]),  # DEPRECATED
            "datetime": location.get("datetime"),  # DEPRECATED
            "container": str(location.get("container") or ""),  # DEPRECATED
            "tar": str(location.get("tar") or ""),  # DEPRECATED
            "member": str(location.get("member") or ""),  # DEPRECATED
            "result_count": 0,  # DEPRECATED
        }  # DEPRECATED
        for location in locations  # DEPRECATED
    ]  # DEPRECATED
    if not points:  # DEPRECATED
        raise ValueError("There are no GPS locations to plot.")  # DEPRECATED
# DEPRECATED
    processed_loaded = merged_results is not None  # DEPRECATED
    results_by_location = {}  # DEPRECATED
    model_results = {}  # DEPRECATED
    if processed_loaded:  # DEPRECATED
        model_names = set()  # DEPRECATED
        sample_percentages = set()  # DEPRECATED
        size_thresholds = set()  # DEPRECATED
        containers = set()  # DEPRECATED
        class_names = set()  # DEPRECATED
        for model_blob, records in sorted(merged_results.items()):  # DEPRECATED
            model_results[model_blob] = []  # DEPRECATED
            for record in records:  # DEPRECATED
                summary = record.get("summary") or {}  # DEPRECATED
                location = record.get("location") or {}  # DEPRECATED
                percentage = summary.get("sample_percent_requested")  # DEPRECATED
                display_record = {  # DEPRECATED
                    "model_blob": model_blob,  # DEPRECATED
                    "summary_blob": record.get("summary_blob") or "",  # DEPRECATED
                    "container": str(location.get("container") or summary.get("container") or ""),  # DEPRECATED
                    "tar": str(location.get("tar") or summary.get("source_tar_blob") or ""),  # DEPRECATED
                    "member": str(location.get("member") or ""),  # DEPRECATED
                    "datetime": location.get("datetime"),  # DEPRECATED
                    "latitude": location.get("latitude"),  # DEPRECATED
                    "longitude": location.get("longitude"),  # DEPRECATED
                    "sample_percent_requested": (  # DEPRECATED
                        float(percentage) if percentage is not None else None  # DEPRECATED
                    ),  # DEPRECATED
                    "size_threshold_px": str(parse_size_threshold(summary.get("size_threshold_px"))),  # DEPRECATED
                    "class_counts": (  # DEPRECATED
                        summary.get("class_counts")  # DEPRECATED
                        if isinstance(summary.get("class_counts"), dict)  # DEPRECATED
                        else {}  # DEPRECATED
                    ),  # DEPRECATED
                }  # DEPRECATED
                model_results[model_blob].append(display_record)  # DEPRECATED
                class_names.update(  # DEPRECATED
                    str(class_name)  # DEPRECATED
                    for class_name in display_record["class_counts"]  # DEPRECATED
                )  # DEPRECATED
                model_names.add(model_blob)  # DEPRECATED
                size_thresholds.add(display_record["size_threshold_px"])  # DEPRECATED
                containers.add(display_record["container"])  # DEPRECATED
                percentage = display_record["sample_percent_requested"]  # DEPRECATED
                if percentage is not None:  # DEPRECATED
                    sample_percentages.add(str(percentage))  # DEPRECATED
                results_by_location.setdefault(  # DEPRECATED
                    (display_record["container"], display_record["tar"]), []  # DEPRECATED
                ).append(display_record)  # DEPRECATED
# DEPRECATED
        for point in points:  # DEPRECATED
            point["result_count"] = len(  # DEPRECATED
                results_by_location.get((point["container"], point["tar"]), [])  # DEPRECATED
            )  # DEPRECATED
# DEPRECATED
    figure = go.Figure([  # DEPRECATED
        go.Scattergeo(  # DEPRECATED
            lat=[point["latitude"] for point in points],  # DEPRECATED
            lon=[point["longitude"] for point in points],  # DEPRECATED
            customdata=[  # DEPRECATED
                [  # DEPRECATED
                    point["datetime"] or "Datetime unavailable",  # DEPRECATED
                    point["container"],  # DEPRECATED
                    point["tar"],  # DEPRECATED
                    point["member"],  # DEPRECATED
                    point["result_count"],  # DEPRECATED
                ]  # DEPRECATED
                for point in points  # DEPRECATED
            ],  # DEPRECATED
            mode="markers",  # DEPRECATED
            marker={"size": 8, "color": "#1677a8", "opacity": 0.8},  # DEPRECATED
            hovertemplate=(  # DEPRECATED
                "Latitude: %{lat:.5f}<br>Longitude: %{lon:.5f}"  # DEPRECATED
                "<br>Datetime: %{customdata[0]}"  # DEPRECATED
                "<br>Container: %{customdata[1]}"  # DEPRECATED
                "<br>TAR: %{customdata[2]}"  # DEPRECATED
                "<br>TIFF: %{customdata[3]}"  # DEPRECATED
                "<br>Matched processed summaries: %{customdata[4]}<extra></extra>"  # DEPRECATED
            ),  # DEPRECATED
        ),  # DEPRECATED
        go.Scattergeo(  # DEPRECATED
            lat=[],  # DEPRECATED
            lon=[],  # DEPRECATED
            customdata=[],  # DEPRECATED
            mode="markers",  # DEPRECATED
            name="Selected class counts",  # DEPRECATED
            marker={  # DEPRECATED
                "size": [],  # DEPRECATED
                "sizemode": "area",  # DEPRECATED
                "sizeref": 1,  # DEPRECATED
                "color": "#e4572e",  # DEPRECATED
                "opacity": 0.65,  # DEPRECATED
                "line": {"width": 1, "color": "#9d301b"},  # DEPRECATED
            },  # DEPRECATED
            hovertemplate=(  # DEPRECATED
                "Latitude: %{lat:.5f}<br>Longitude: %{lon:.5f}"  # DEPRECATED
                "<br>Datetime: %{customdata[0]}"  # DEPRECATED
                "<br>Container: %{customdata[1]}"  # DEPRECATED
                "<br>TAR: %{customdata[2]}"  # DEPRECATED
                "<br>TIFF: %{customdata[3]}"  # DEPRECATED
                "<br>Matched processed summaries: %{customdata[4]}"  # DEPRECATED
                "<br>Selected-class count: %{customdata[5]}<extra></extra>"  # DEPRECATED
            ),  # DEPRECATED
        ),  # DEPRECATED
    ])  # DEPRECATED
    figure.update_layout(  # DEPRECATED
        title="GPS locations",  # DEPRECATED
        template="plotly_white",  # DEPRECATED
        geo={  # DEPRECATED
            "scope": "world",  # DEPRECATED
            "showland": True,  # DEPRECATED
            "landcolor": "#edf3f5",  # DEPRECATED
            "showocean": True,  # DEPRECATED
            "oceancolor": "#dcecf2",  # DEPRECATED
            "showlakes": True,  # DEPRECATED
            "lakecolor": "#dcecf2",  # DEPRECATED
            "showcoastlines": True,  # DEPRECATED
            "coastlinecolor": "#9aadb5",  # DEPRECATED
            "fitbounds": "locations",  # DEPRECATED
        },  # DEPRECATED
        margin={"l": 20, "r": 20, "t": 55, "b": 20},  # DEPRECATED
    )  # DEPRECATED
    plot = pio.to_html(  # DEPRECATED
        figure,  # DEPRECATED
        full_html=False,  # DEPRECATED
        include_plotlyjs=True,  # DEPRECATED
        div_id="gps-locations-plot",  # DEPRECATED
        config={"responsive": True, "displaylogo": False},  # DEPRECATED
    )  # DEPRECATED
    points_json = json.dumps(points, ensure_ascii=True).replace("</", "<\\/")  # DEPRECATED
    model_results_json = json.dumps(model_results, ensure_ascii=True).replace("</", "<\\/")  # DEPRECATED
    processed_filters = ""  # DEPRECATED
    processed_summary_status = ""  # DEPRECATED
    if processed_loaded:  # DEPRECATED
        class_filter = ""  # DEPRECATED
        if class_names:  # DEPRECATED
            class_filter = (  # DEPRECATED
                f"""\n"""  # DEPRECATED
                f"""    <div class="class-control">\n"""  # DEPRECATED
                f"""      <span>Classes to plot</span>\n"""  # DEPRECATED
                f"""      <details>\n"""  # DEPRECATED
                f"""        <summary id="class-selection-count"></summary>\n"""  # DEPRECATED
                f"""        <div id="class-filter" class="class-filter">{_class_checkbox_options(class_names)}</div>\n"""  # DEPRECATED
                f"""      </details>\n"""  # DEPRECATED
                f"""    </div>"""  # DEPRECATED
            )  # DEPRECATED
        processed_filters = (  # DEPRECATED
            f"""\n"""  # DEPRECATED
            f"""    <label>Model <select id="model-filter">{_select_options(model_names)}</select></label>\n"""  # DEPRECATED
            f"""    <label>Sample percent requested\n"""  # DEPRECATED
            f"""      <select id="sample-percent-filter">{_select_options(sample_percentages)}</select>\n"""  # DEPRECATED
            f"""    </label>\n"""  # DEPRECATED
            f"""    <label>Size threshold (px; inf = not applied)\n"""  # DEPRECATED
            f"""      <select id="size-threshold-filter">{_select_options(size_thresholds)}</select>\n"""  # DEPRECATED
            f"""    </label>\n"""  # DEPRECATED
            f"""    <label>Raw container <select id="container-filter">{_select_options(containers)}</select></label>\n"""  # DEPRECATED
            f"""    {class_filter}"""  # DEPRECATED
        )  # DEPRECATED
        processed_summary_status = (  # DEPRECATED
            f"<p>{int(processed_summary_count):,} summary file(s) downloaded. "  # DEPRECATED
            "Class counts are kept separate for each model and matched TAR; circle area "  # DEPRECATED
            "represents the selected-class total.</p>"  # DEPRECATED
        )  # DEPRECATED
    return (  # DEPRECATED
        f"""<!doctype html>\n"""  # DEPRECATED
        f"""<html lang="en">\n"""  # DEPRECATED
        f"""<head>\n"""  # DEPRECATED
        f"""  <meta charset="utf-8">\n"""  # DEPRECATED
        f"""  <meta name="viewport" content="width=device-width, initial-scale=1">\n"""  # DEPRECATED
        f"""  <title>GPS locations</title>\n"""  # DEPRECATED
        f"""  <style>\n"""  # DEPRECATED
        f"""    body {{ font: 15px Arial, sans-serif; color: #20313a; margin: 24px auto; max-width: 1400px; padding: 0 20px; }}\n"""  # DEPRECATED
        f"""    .filters {{ display: flex; flex-wrap: wrap; align-items: end; gap: 12px; margin-bottom: 12px; }}\n"""  # DEPRECATED
        f"""    label {{ display: grid; gap: 5px; }}\n"""  # DEPRECATED
        f"""    input, select, button {{ font: inherit; padding: 7px 9px; }}\n"""  # DEPRECATED
        f"""    button {{ cursor: pointer; }}\n"""  # DEPRECATED
        f"""    .class-control {{ display: grid; gap: 5px; }}\n"""  # DEPRECATED
        f"""    .class-control summary {{ cursor: pointer; }}\n"""  # DEPRECATED
        f"""    .class-filter {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 4px 12px; max-height: 220px; min-width: 260px; overflow: auto; padding: 8px; border: 1px solid #d3dce0; background: white; }}\n"""  # DEPRECATED
        f"""    .class-option {{ display: flex; align-items: center; gap: 6px; }}\n"""  # DEPRECATED
        f"""    .class-option input {{ margin: 0; padding: 0; }}\n"""  # DEPRECATED
        f"""    #result-count {{ margin: 0 0 8px; }}\n"""  # DEPRECATED
        f"""    #filter-message {{ color: #a12622; }}\n"""  # DEPRECATED
        f"""    #gps-locations-plot {{ min-height: 650px; }}\n"""  # DEPRECATED
        f"""  </style>\n"""  # DEPRECATED
        f"""</head>\n"""  # DEPRECATED
        f"""<body>\n"""  # DEPRECATED
        f"""  <h1>GPS locations</h1>\n"""  # DEPRECATED
        f"""  <div class="filters">\n"""  # DEPRECATED
        f"""    <label>From datetime <input id="date-start" type="datetime-local" step="1"></label>\n"""  # DEPRECATED
        f"""    <label>To datetime <input id="date-end" type="datetime-local" step="1"></label>\n"""  # DEPRECATED
        f"""    {processed_filters}\n"""  # DEPRECATED
        f"""    <button id="apply-filter" type="button">Apply filter</button>\n"""  # DEPRECATED
        f"""    <button id="clear-filter" type="button">Clear filter</button>\n"""  # DEPRECATED
        f"""  </div>\n"""  # DEPRECATED
        f"""  <p id="result-count"></p>\n"""  # DEPRECATED
        f"""  <p id="filter-message" role="alert"></p>\n"""  # DEPRECATED
        f"""  <p>Datetime filters use the TIFF capture time. Locations without a capture time are hidden while a filter is active.</p>\n"""  # DEPRECATED
        f"""  {processed_summary_status}\n"""  # DEPRECATED
        f"""  {plot}\n"""  # DEPRECATED
        f"""  <script>\n"""  # DEPRECATED
        f"""    const allPoints = {points_json};\n"""  # DEPRECATED
        f"""    const modelResults = {model_results_json};\n"""  # DEPRECATED
        f"""    const plot = document.getElementById("gps-locations-plot");\n"""  # DEPRECATED
        f"""    const startInput = document.getElementById("date-start");\n"""  # DEPRECATED
        f"""    const endInput = document.getElementById("date-end");\n"""  # DEPRECATED
        f"""    const count = document.getElementById("result-count");\n"""  # DEPRECATED
        f"""    const message = document.getElementById("filter-message");\n"""  # DEPRECATED
        f"""    const classFilter = document.getElementById("class-filter");\n"""  # DEPRECATED
        f"""    const classSelectionCount = document.getElementById("class-selection-count");\n"""  # DEPRECATED
        f"""    const processedLoaded = {str(processed_loaded).lower()};\n"""  # DEPRECATED
        f"""    const normalizedDatetime = value => {{\n"""  # DEPRECATED
        f"""      if (!value) return "";\n"""  # DEPRECATED
        f"""      return value.length === 16 ? value + ":00" : value;\n"""  # DEPRECATED
        f"""    }};\n"""  # DEPRECATED
        f"""\n"""  # DEPRECATED
        f"""    const resultsByLocation = new Map();\n"""  # DEPRECATED
        f"""    Object.values(modelResults).flat().forEach(result => {{\n"""  # DEPRECATED
        f"""      const key = JSON.stringify([result.container, result.tar]);\n"""  # DEPRECATED
        f"""      if (!resultsByLocation.has(key)) resultsByLocation.set(key, []);\n"""  # DEPRECATED
        f"""      resultsByLocation.get(key).push(result);\n"""  # DEPRECATED
        f"""    }});\n"""  # DEPRECATED
        f"""    allPoints.forEach(point => {{\n"""  # DEPRECATED
        f"""      point.results = resultsByLocation.get(JSON.stringify([point.container, point.tar])) || [];\n"""  # DEPRECATED
        f"""    }});\n"""  # DEPRECATED
        f"""\n"""  # DEPRECATED
        f"""    function matchingResults(point) {{\n"""  # DEPRECATED
        f"""      if (!processedLoaded) return [];\n"""  # DEPRECATED
        f"""      const model = document.getElementById("model-filter").value;\n"""  # DEPRECATED
        f"""      const samplePercent = document.getElementById("sample-percent-filter").value;\n"""  # DEPRECATED
        f"""      const sizeThreshold = document.getElementById("size-threshold-filter").value;\n"""  # DEPRECATED
        f"""      const container = document.getElementById("container-filter").value;\n"""  # DEPRECATED
        f"""      return point.results.filter(result =>\n"""  # DEPRECATED
        f"""        (!model || result.model_blob === model) &&\n"""  # DEPRECATED
        f"""        (!samplePercent || result.sample_percent_requested === Number(samplePercent)) &&\n"""  # DEPRECATED
        f"""        (!sizeThreshold || result.size_threshold_px === sizeThreshold) &&\n"""  # DEPRECATED
        f"""        (!container || result.container === container)\n"""  # DEPRECATED
        f"""      );\n"""  # DEPRECATED
        f"""    }}\n"""  # DEPRECATED
        f"""\n"""  # DEPRECATED
        f"""    function selectedClasses() {{\n"""  # DEPRECATED
        f"""      return classFilter\n"""  # DEPRECATED
        f"""        ? [...classFilter.querySelectorAll('input[type="checkbox"]:checked')]\n"""  # DEPRECATED
        f"""          .map(option => option.value)\n"""  # DEPRECATED
        f"""        : [];\n"""  # DEPRECATED
        f"""    }}\n"""  # DEPRECATED
        f"""\n"""  # DEPRECATED
        f"""    function selectedClassCount(results, classes) {{\n"""  # DEPRECATED
        f"""      return results.reduce((total, result) => {{\n"""  # DEPRECATED
        f"""        classes.forEach(className => {{\n"""  # DEPRECATED
        f"""          const value = Number((result.class_counts || {{}})[className]);\n"""  # DEPRECATED
        f"""          if (Number.isFinite(value) && value > 0 && Number.isFinite(total + value)) {{\n"""  # DEPRECATED
        f"""            total += value;\n"""  # DEPRECATED
        f"""          }}\n"""  # DEPRECATED
        f"""        }});\n"""  # DEPRECATED
        f"""        return total;\n"""  # DEPRECATED
        f"""      }}, 0);\n"""  # DEPRECATED
        f"""    }}\n"""  # DEPRECATED
        f"""\n"""  # DEPRECATED
        f"""    function updateClassSelectionCount() {{\n"""  # DEPRECATED
        f"""      if (!classFilter || !classSelectionCount) return;\n"""  # DEPRECATED
        f"""      const options = classFilter.querySelectorAll('input[type="checkbox"]');\n"""  # DEPRECATED
        f"""      const selected = classFilter.querySelectorAll('input[type="checkbox"]:checked');\n"""  # DEPRECATED
        f"""      classSelectionCount.textContent =\n"""  # DEPRECATED
        f"""        `${{selected.length}} of ${{options.length}} classes selected`;\n"""  # DEPRECATED
        f"""    }}\n"""  # DEPRECATED
        f"""\n"""  # DEPRECATED
        f"""    function applyFilter() {{\n"""  # DEPRECATED
        f"""      const start = normalizedDatetime(startInput.value);\n"""  # DEPRECATED
        f"""      const end = normalizedDatetime(endInput.value);\n"""  # DEPRECATED
        f"""      if (start && end && end < start) {{\n"""  # DEPRECATED
        f"""        message.textContent = "End datetime must be on or after start datetime.";\n"""  # DEPRECATED
        f"""        return;\n"""  # DEPRECATED
        f"""      }}\n"""  # DEPRECATED
        f"""      message.textContent = "";\n"""  # DEPRECATED
        f"""      const visible = allPoints.filter(point => {{\n"""  # DEPRECATED
        f"""        if (!start && !end) return true;\n"""  # DEPRECATED
        f"""        if (!point.datetime) return false;\n"""  # DEPRECATED
        f"""        const value = normalizedDatetime(point.datetime);\n"""  # DEPRECATED
        f"""        return (!start || value >= start) && (!end || value <= end);\n"""  # DEPRECATED
        f"""      }}).map(point => ({{...point, visibleResults: matchingResults(point)}}));\n"""  # DEPRECATED
        f"""      const hasProcessedFilter = processedLoaded && (\n"""  # DEPRECATED
        f"""        document.getElementById("model-filter").value ||\n"""  # DEPRECATED
        f"""        document.getElementById("sample-percent-filter").value ||\n"""  # DEPRECATED
        f"""        document.getElementById("size-threshold-filter").value ||\n"""  # DEPRECATED
        f"""        document.getElementById("container-filter").value\n"""  # DEPRECATED
        f"""      );\n"""  # DEPRECATED
        f"""      const filtered = hasProcessedFilter\n"""  # DEPRECATED
        f"""        ? visible.filter(point => point.visibleResults.length)\n"""  # DEPRECATED
        f"""        : visible;\n"""  # DEPRECATED
        f"""      Plotly.restyle(plot, {{\n"""  # DEPRECATED
        f"""        lat: [filtered.map(point => point.latitude)],\n"""  # DEPRECATED
        f"""        lon: [filtered.map(point => point.longitude)],\n"""  # DEPRECATED
        f"""        customdata: [filtered.map(point => [\n"""  # DEPRECATED
        f"""          point.datetime || "Datetime unavailable",\n"""  # DEPRECATED
        f"""          point.container,\n"""  # DEPRECATED
        f"""          point.tar,\n"""  # DEPRECATED
        f"""          point.member,\n"""  # DEPRECATED
        f"""          point.visibleResults.length\n"""  # DEPRECATED
        f"""        ])]\n"""  # DEPRECATED
        f"""      }}, [0]);\n"""  # DEPRECATED
        f"""      const classes = selectedClasses();\n"""  # DEPRECATED
        f"""      const countPoints = filtered.map(point => ({{\n"""  # DEPRECATED
        f"""        point,\n"""  # DEPRECATED
        f"""        count: selectedClassCount(point.visibleResults, classes)\n"""  # DEPRECATED
        f"""      }})).filter(item => item.count > 0);\n"""  # DEPRECATED
        f"""      const maxCount = countPoints.reduce(\n"""  # DEPRECATED
        f"""        (maximum, item) => Math.max(maximum, item.count),\n"""  # DEPRECATED
        f"""        0\n"""  # DEPRECATED
        f"""      );\n"""  # DEPRECATED
        f"""      const sizeReference = maxCount ? 2 * maxCount / (40 * 40) : 1;\n"""  # DEPRECATED
        f"""      Plotly.restyle(plot, {{\n"""  # DEPRECATED
        f"""        lat: [countPoints.map(item => item.point.latitude)],\n"""  # DEPRECATED
        f"""        lon: [countPoints.map(item => item.point.longitude)],\n"""  # DEPRECATED
        f"""        customdata: [countPoints.map(item => [\n"""  # DEPRECATED
        f"""          item.point.datetime || "Datetime unavailable",\n"""  # DEPRECATED
        f"""          item.point.container,\n"""  # DEPRECATED
        f"""          item.point.tar,\n"""  # DEPRECATED
        f"""          item.point.member,\n"""  # DEPRECATED
        f"""          item.point.visibleResults.length,\n"""  # DEPRECATED
        f"""          item.count\n"""  # DEPRECATED
        f"""        ])],\n"""  # DEPRECATED
        f"""        "marker.size": [countPoints.map(item => item.count)],\n"""  # DEPRECATED
        f"""        "marker.sizeref": [sizeReference]\n"""  # DEPRECATED
        f"""      }}, [1]);\n"""  # DEPRECATED
        f"""      const matchedSummaryCount = filtered.reduce(\n"""  # DEPRECATED
        f"""        (total, point) => total + point.visibleResults.length,\n"""  # DEPRECATED
        f"""        0\n"""  # DEPRECATED
        f"""      );\n"""  # DEPRECATED
        f"""      count.textContent = processedLoaded\n"""  # DEPRECATED
        f"""        ? `${{filtered.length.toLocaleString()}} of ${{allPoints.length.toLocaleString()}} locations; ${{matchedSummaryCount.toLocaleString()}} matched summary record(s)`\n"""  # DEPRECATED
        f"""        : `${{filtered.length.toLocaleString()}} of ${{allPoints.length.toLocaleString()}} locations`;\n"""  # DEPRECATED
        f"""    }}\n"""  # DEPRECATED
        f"""\n"""  # DEPRECATED
        f"""    document.getElementById("apply-filter").addEventListener("click", applyFilter);\n"""  # DEPRECATED
        f"""    document.getElementById("clear-filter").addEventListener("click", () => {{\n"""  # DEPRECATED
        f"""      startInput.value = "";\n"""  # DEPRECATED
        f"""      endInput.value = "";\n"""  # DEPRECATED
        f"""      if (processedLoaded) {{\n"""  # DEPRECATED
        f"""        document.getElementById("model-filter").value = "";\n"""  # DEPRECATED
        f"""        document.getElementById("sample-percent-filter").value = "";\n"""  # DEPRECATED
        f"""        document.getElementById("size-threshold-filter").value = "";\n"""  # DEPRECATED
        f"""        document.getElementById("container-filter").value = "";\n"""  # DEPRECATED
        f"""      }}\n"""  # DEPRECATED
        f"""      if (classFilter) {{\n"""  # DEPRECATED
        f"""        classFilter.querySelectorAll('input[type="checkbox"]')\n"""  # DEPRECATED
        f"""          .forEach(option => {{ option.checked = true; }});\n"""  # DEPRECATED
        f"""        updateClassSelectionCount();\n"""  # DEPRECATED
        f"""      }}\n"""  # DEPRECATED
        f"""      applyFilter();\n"""  # DEPRECATED
        f"""    }});\n"""  # DEPRECATED
        f"""    if (classFilter) {{\n"""  # DEPRECATED
        f"""      classFilter.addEventListener("change", () => {{\n"""  # DEPRECATED
        f"""        updateClassSelectionCount();\n"""  # DEPRECATED
        f"""        applyFilter();\n"""  # DEPRECATED
        f"""      }});\n"""  # DEPRECATED
        f"""      updateClassSelectionCount();\n"""  # DEPRECATED
        f"""    }}\n"""  # DEPRECATED
        f"""    applyFilter();\n"""  # DEPRECATED
        f"""  </script>\n"""  # DEPRECATED
        f"""</body>\n"""  # DEPRECATED
        f"""</html>\n"""  # DEPRECATED
    )  # DEPRECATED
