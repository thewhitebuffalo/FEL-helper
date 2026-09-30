#!/usr/bin/env python3
"""Validate private FEL recordings without including them in the repository.

Manifest: {"recordings": [{"fel": "capture.fel", "events_csv": "events.txt",
"transient_csv": "transients.txt", "timezone": "America/Los_Angeles"}]}.
CSV paths are optional. Relative paths resolve beside the manifest. Results may
contain local filenames and must stay outside the public repository.

The independent struct reader checks extraction, not electrical interpretation.
Named tag-70/71 field labels come from the decoder registry, so these checks do
not independently validate those labels. Vendor event/transient CSV comparisons
validate only their corresponding measurements. Use validate_energy_exports.py
for independent comparisons with native energy-trend/demand exports.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from itertools import zip_longest
import math
from pathlib import Path
import re
import statistics
import struct
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

import numpy as np

# Supports an editable installation or execution directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fel_decoder.core import decode_bytes, decode_file  # noqa: E402
from fel_decoder.layouts import ENERGY_TREND_FIELDS, DEMAND_FIELDS  # noqa: E402

EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)
DECIMAL_TOLERANCE = 0.0006
EVENT_TYPES = {
    0: "DIP", 2: "INTERRUPTION", 4: "RVC", 6: "WAVESHAPE",
    7: "RVC_CANCELED", 8: "TRANSIENT",
}
CHANNELS = {1: "-1---", 2: "--2--", 4: "---3-", 14: "P-23-", 15: "P123-"}
SEVERITIES = {0: "NOT_SPECIFIED", 2: "MEDIUM", 3: "LOW"}
WINDOWS = ("wave_start", "wave_end", "rms_start", "rms_end",
           "msv_start", "msv_end", "transient_start", "transient_end")
# Sizes/counts are independent literals. Field names are shared deliberately:
# this is an extraction check, not a second source of electrical semantics.
NAMED_LAYOUTS = {
    70: (744, 180, "energy_trends", "trend_period", ENERGY_TREND_FIELDS),
    71: (104, 20, "demand", "demand_period", DEMAND_FIELDS),
}
GROUP_NAMES = {105: "waveforms", 111: "events", 112: "transients", 119: "trends",
               70: "energy_trends", 71: "demand"}


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def ticks(data, offset):
    """Independent integer reference, deliberately separate from decoder helpers."""
    high = int.from_bytes(data[offset:offset + 4], "little")
    low = int.from_bytes(data[offset + 4:offset + 8], "little")
    return high * 4294967296 + low


def reference_records(data):
    require(len(data) >= 32, "Reference: truncated header")
    require(data[:4] == bytes.fromhex("0bba1400"), "Reference: unsupported signature")
    require(data[24:32] == bytes.fromhex("0000090000000000"), "Reference: unsupported profile")
    require(int.from_bytes(data[4:8], "little") == len(data) - 32, "Reference: wrong declared size")
    rows, offset = [], 32
    while offset < len(data):
        require(offset + 4 <= len(data), f"Reference: short framing at {offset}")
        tag, size = struct.unpack_from("<HH", data, offset)
        require(size >= 4 and offset + size <= len(data), f"Reference: invalid record at {offset}")
        rows.append((offset, tag, size))
        offset += size
    return rows


def equal(actual, expected, description):
    require(np.array_equal(actual, np.asarray(expected), equal_nan=True), description)


def validate_struct(data, decoded):
    reference = reference_records(data)
    actual = [(r.offset, r.tag, r.size) for r in decoded.records]
    require(actual == reference, "Decoder record inventory differs from independent scanner")
    require(decoded.source_bytes == len(data), "Wrong source size")
    require(decoded.source_sha256 == hashlib.sha256(data).hexdigest(), "Wrong source hash")
    require(decoded.header_hex == data[:32].hex(), "Wrong retained header")
    groups = decoded.groups
    seen = Counter()
    checked = Counter()
    cursors = Counter()
    for offset, tag, size in reference:
        if tag not in GROUP_NAMES:
            continue
        name = GROUP_NAMES[tag]
        require(name in groups, f"Missing decoded group: {name}")
        group, row = groups[name], seen[tag]
        seen[tag] += 1
        require(int(group["record_offset"][row]) == offset, f"{name}: wrong record offset")
        for key, rel in (("start_ticks", 4), ("end_ticks", 12)):
            require(int(group[key][row]) == ticks(data, offset + rel), f"{name}: wrong {key}")
        if tag in NAMED_LAYOUTS:
            expected_size, count, _, period_key, fields = NAMED_LAYOUTS[tag]
            require(size == expected_size, f"{name}: unsupported reference record size")
            require(len(fields) == count and len(set(fields)) == count,
                    f"{name}: field registry differs from the reference field count")
            expected_keys = ("record_offset", "start_ticks", "end_ticks", period_key, *fields)
            require(tuple(group) == expected_keys, f"{name}: output fields/order differ from registry")
            expected_period = struct.unpack_from("<I", data, offset + 20)[0]
            require(int(group[period_key][row]) == expected_period, f"{name}: wrong {period_key}")
            # Deliberately unpack each scalar without np.frombuffer or the
            # decoder's layout dispatch. Every float is checked, including NaNs.
            for index, field in enumerate(fields):
                expected = struct.unpack_from("<f", data, offset + 24 + index * 4)[0]
                equal(group[field][row], expected, f"{name}: {field} differs from scalar struct reference")
            checked[name + "_records"] += 1
            checked[name + "_scalar_values"] += count
        elif tag == 111:
            for key, fmt, rel in (("event_id", "<I", 20), ("event_type", "<H", 24),
                                  ("channel_code", "<H", 26), ("value", "<f", 28),
                                  ("depth", "<f", 32), ("severity", "<I", 36)):
                equal(group[key][row], struct.unpack_from(fmt, data, offset + rel)[0], f"Event {key}")
            for index, key in enumerate(WINDOWS):
                require(int(group[key + "_ticks"][row]) == ticks(data, offset + 40 + 8 * index),
                        f"Event window {key}")
            checked["events"] += 1
        elif tag == 119:
            expected = np.array(struct.unpack_from("<18f", data, offset + 24)).reshape(6, 3)
            equal(group["values"][row], expected, "Trend triples differ from struct reader")
            checked["trend_triples"] += 6
        else:
            channels, base, stride = (8, 88, 20) if tag == 105 else (4, 64, 8)
            count = struct.unpack_from("<I", data, offset + base - 4)[0]
            begin = cursors[tag]
            require(int(group["sample_start"][row]) == begin, "Incorrect sample boundary")
            require(int(group["sample_count"][row]) == count, "Incorrect sample count")
            require(count > 0 and base + count * stride <= size, "Sample count exceeds record capacity")
            selection = slice(begin, begin + count)
            coefficients = struct.unpack_from(f"<{2 * channels}f", data, offset + 20)
            equal(group["coefficients"][row], np.array(coefficients).reshape(channels, 2), "Wrong coefficients")
            expected_raw, expected_values, expected_offsets = [], [], []
            for sample in range(count):
                at = offset + base + sample * stride
                if tag == 105:
                    expected_offsets.append(struct.unpack_from("<I", data, at)[0])
                    at += 4
                raw = struct.unpack_from(f"<{channels}h", data, at)
                expected_raw.append(raw)
                # Scalar Python arithmetic is independent of NumPy vectorized scaling.
                expected_values.append([float(raw[ch]) * coefficients[ch * 2] + coefficients[ch * 2 + 1]
                                        for ch in range(channels)])
            equal(group["raw"][selection], expected_raw, "Raw samples differ from scalar struct reference")
            equal(group["values"][selection], expected_values, "Scaled samples differ from scalar reference")
            if tag == 105:
                equal(group["offset_us"][selection], expected_offsets, "Incorrect waveform offsets")
            else:
                rate = struct.unpack_from("<I", data, offset + 56)[0]
                require(int(group["sample_rate_hz"][row]) == rate, "Incorrect transient sample rate")
            checked[name + "_sample_rows"] += count
            checked[name + "_scalar_values"] += count * channels
            cursors[tag] += count
    for tag, name in GROUP_NAMES.items():
        if tag in NAMED_LAYOUTS and not seen[tag]:
            require(name not in groups, f"Unexpected empty {name} group")
            continue
        require(len(groups[name]["record_offset"]) == seen[tag], f"Wrong {name} output length")
        if tag in NAMED_LAYOUTS:
            _, _, _, period_key, fields = NAMED_LAYOUTS[tag]
            for field, values in groups[name].items():
                require(values.shape == (seen[tag],), f"{name}/{field}: wrong array shape")
                expected_dtype = ("<f4" if field in fields else "<u4" if field == period_key else "<u8")
                require(values.dtype == np.dtype(expected_dtype), f"{name}/{field}: wrong dtype")
    for tag, name in ((105, "waveforms"), (112, "transients")):
        require(len(groups[name]["values"]) == cursors[tag], f"Wrong {name} sample length")
    return {"record_count": len(reference), "record_counts": dict(sorted(Counter(t for _, t, _ in reference).items())),
            "checked": dict(checked), "validation_kind": "structural_extraction",
            "coverage": {
                "supported_tags": sorted(GROUP_NAMES),
                "named_field_labels": "Shared decoder registry; not independently validated by this scalar reader.",
                "vendor_energy_trend_or_demand_validation": False,
            }}


def iso_ticks(value, local_zone):
    # Preserve a possible seventh fractional digit instead of rounding through floats.
    match = re.search(r"[.,](\d+)", value)
    remainder = 0
    if match:
        digits = match.group(1)
        require(not any(c != "0" for c in digits[7:]), "CSV timestamp finer than 100 ns")
        remainder = int((digits + "0000000")[6])
    instant = datetime.fromisoformat(value)
    if instant.tzinfo is None:
        # A timezone is required explicitly: the vendor header's 'Standard Time'
        # wording may still contain daylight-saving local timestamps.
        first = instant.replace(tzinfo=local_zone, fold=0)
        second = instant.replace(tzinfo=local_zone, fold=1)
        require(first.utcoffset() == second.utcoffset(), "Ambiguous/nonexistent local CSV timestamp; use an offset")
        instant = first
    delta = instant.astimezone(timezone.utc) - EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds) * 10 + remainder


def named_column(columns, prefix):
    matches = [key for key in columns if key.startswith(prefix + "(") or key == prefix]
    require(len(matches) == 1, f"Expected one CSV {prefix} column")
    return matches[0]


def validate_events(path, group, local_zone):
    ids = [int(x) for x in group["event_id"]]
    require(len(set(ids)) == len(ids), "Duplicate decoded event IDs")
    indexed = {ident: row for row, ident in enumerate(ids)}
    seen, errors, timestamps = set(), {"value": 0.0, "depth": 0.0}, 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        start_key, stop_key = (named_column(reader.fieldnames or [], key) for key in ("Start", "Stop"))
        for vendor in reader:
            ident = int(vendor["ID"])
            require(ident not in seen and ident in indexed, f"Missing/duplicate vendor event {ident}")
            seen.add(ident)
            row = indexed[ident]
            for csv_key, array_key in ((start_key, "start_ticks"), (stop_key, "end_ticks")):
                require(iso_ticks(vendor[csv_key], local_zone) == int(group[array_key][row]), f"Event {ident}: {csv_key}")
                timestamps += 1
            for key in WINDOWS:
                expected = iso_ticks(vendor[key], local_zone) if vendor[key].strip() else 0
                require(expected == int(group[key + "_ticks"][row]), f"Event {ident}: {key}")
                timestamps += bool(vendor[key].strip())
            for csv_key, array_key in (("absolute_value", "value"), ("depth", "depth")):
                error = abs(float(vendor[csv_key]) - float(group[array_key][row]))
                require(error <= DECIMAL_TOLERANCE, f"Event {ident}: {array_key} error {error}")
                errors[array_key] = max(errors[array_key], error)
            for csv_key, array_key, mapping in (("type", "event_type", EVENT_TYPES),
                                                ("channel", "channel_code", CHANNELS),
                                                ("severity", "severity", SEVERITIES)):
                code = int(group[array_key][row])
                require(code in mapping, f"Unvalidated {array_key} code {code}; extend the harness with evidence")
                require(vendor[csv_key] == mapping[code], f"Event {ident}: {array_key} mismatch")
    require(seen == set(ids), "Vendor export does not cover every decoded event")
    return {"rows_checked": len(seen), "timestamps_checked": timestamps, "max_timestamp_error_ticks": 0,
            "max_value_error": errors["value"], "max_depth_error": errors["depth"],
            "value_tolerance": DECIMAL_TOLERANCE}


def validate_transients(path, group, local_zone):
    seen, count, row, in_record = Counter(), 0, 0, 0
    max_errors = [0.0, 0.0, 0.0]
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        time_key = named_column(reader.fieldnames or [], "Time")
        for vendor in reader:
            while row < len(group["sample_count"]) and in_record == int(group["sample_count"][row]):
                row += 1
                in_record = 0
            require(row < len(group["sample_count"]), "Vendor transient export has extra samples")
            table_index = int(vendor["TableIndex"])
            require(table_index == row, "Transient TableIndex does not match record boundary")
            rate = int(group["sample_rate_hz"][row])
            numerator = in_record * 10000000
            require(numerator % rate == 0, "Transient sampling cannot be represented in exact 100 ns ticks")
            expected_time = int(group["start_ticks"][row]) + numerator // rate
            nanos = int(vendor.get("nanos", "0") or 0)
            require(0 <= nanos < 1000 and nanos % 100 == 0, "Unsupported CSV sub-microsecond remainder")
            actual_time = iso_ticks(vendor[time_key], local_zone) + nanos // 100
            require(actual_time == expected_time, f"Transient timestamp mismatch at sample {count}")
            actual_index = int(group["sample_start"][row]) + in_record
            for channel, key in enumerate(("Voltage_A", "Voltage_B", "Voltage_C")):
                error = abs(float(vendor[key]) - float(group["values"][actual_index, channel]))
                require(error <= DECIMAL_TOLERANCE, f"Transient value error {error} at sample {count}, channel {channel}")
                max_errors[channel] = max(max_errors[channel], error)
            seen[row] += 1
            in_record += 1
            count += 1
    require(count == len(group["values"]), "Vendor export does not cover every transient sample")
    require([seen[i] for i in range(len(group["sample_count"]))] == list(group["sample_count"]),
            "Transient export record boundaries differ")
    return {"sample_rows_checked": count, "scalar_values_checked": count * 3,
            "record_boundaries_checked": len(seen), "max_timestamp_error_ticks": 0,
            "max_value_error_by_channel": max_errors, "value_tolerance": DECIMAL_TOLERANCE}


def resolve_path(value, base):
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def check_export_arrays(destination, decoded):
    manifest = json.loads((destination / "manifest.json").read_text())
    require(manifest["source_sha256"] == decoded.source_sha256, "Export manifest source hash differs")
    require(manifest["record_count"] == len(decoded.records), "Export manifest record count differs")
    checked = 0
    for name, arrays in decoded.groups.items():
        with np.load(destination / (name + ".npz"), allow_pickle=False) as actual:
            require(set(actual.files) == set(arrays), "NPZ array keys differ")
            for key, expected in arrays.items():
                require(actual[key].dtype == expected.dtype, f"{name}/{key}: NPZ dtype differs")
                equal(actual[key], expected, f"{name}/{key}: NPZ round trip differs")
                checked += 1
    return checked


def check_csv(path, headers, expected_rows):
    missing, count = object(), 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        require(next(reader) == headers, f"{path.name}: CSV header differs")
        for actual, expected in zip_longest(reader, expected_rows, fillvalue=missing):
            require(actual is not missing and expected is not missing, f"{path.name}: CSV row count differs")
            require(len(actual) == len(expected), f"{path.name}: CSV column count differs")
            for text, value in zip(actual, expected):
                if isinstance(value, (float, np.floating)):
                    parsed = float(text)
                    require(parsed == value or (math.isnan(parsed) and math.isnan(value)),
                            f"{path.name}: float round trip differs at row {count}")
                else:
                    # Parsing directly to int confirms timestamps did not go through
                    # float strings, which would lose precision at this magnitude.
                    require(int(text) == int(value), f"{path.name}: integer differs at row {count}")
            count += 1
    return count


def check_export_csv(destination, decoded, samples):
    record_indices = {record.offset: index for index, record in enumerate(decoded.records)}
    metadata = ["record_index", "record_offset", "start_ticks", "end_ticks"]

    def metadata_values(group, row):
        offset = int(group["record_offset"][row])
        return [record_indices[offset], offset, int(group["start_ticks"][row]), int(group["end_ticks"][row])]

    counts = {}
    counts["records"] = check_csv(destination / "records.csv", ["record_index", "offset", "tag", "size"],
                                  ([index, record.offset, record.tag, record.size]
                                   for index, record in enumerate(decoded.records)))
    events = decoded.groups["events"]
    fields = [key for key in events if key not in metadata]
    counts["events"] = check_csv(destination / "events.csv", metadata + fields,
                                 (metadata_values(events, row) + [events[key][row] for key in fields]
                                  for row in range(len(events["record_offset"]))))
    if not samples:
        return counts
    for name, channels in (("waveforms", 8), ("transients", 4)):
        group = decoded.groups[name]
        rate_key = "offset_us" if name == "waveforms" else "sample_rate_hz"
        header = metadata + ["sample_index", rate_key]
        header += [f"channel_{channel}_raw" for channel in range(1, channels + 1)]
        header += [f"channel_{channel}_value" for channel in range(1, channels + 1)]

        def rows(group=group, name=name):
            for record_row in range(len(group["record_offset"])):
                values = metadata_values(group, record_row)
                begin = int(group["sample_start"][record_row])
                for local in range(int(group["sample_count"][record_row])):
                    sample = begin + local
                    relative = (group["offset_us"][sample] if name == "waveforms"
                                else group["sample_rate_hz"][record_row])
                    yield values + [local, relative] + list(group["raw"][sample]) + list(group["values"][sample])

        counts[name] = check_csv(destination / (name + ".csv"), header, rows())
    trends = decoded.groups["trends"]
    counts["trends"] = check_csv(
        destination / "trends.csv", metadata + [f"channel_{channel}_field_{field}"
                                                  for channel in range(1, 7) for field in range(1, 4)],
        (metadata_values(trends, row) + list(trends["values"][row].flat)
         for row in range(len(trends["record_offset"]))),
    )
    for name in ("energy_trends", "demand"):
        if name not in decoded.groups:
            continue
        group = decoded.groups[name]
        fields = [key for key in group if key not in metadata]
        counts[name] = check_csv(
            destination / (name + ".csv"), metadata + fields,
            (metadata_values(group, row) + [group[key][row] for key in fields]
             for row in range(len(group["record_offset"]))),
        )
    return counts


def validate_exports(path, reference, scratch_parent, check_full_csv):
    from fel_decoder.export import export_decoded

    result = {"source": str(path), "source_sha256": reference.source_sha256, "variants": {}}
    variants = [("default", False, False), ("compressed", True, False)]
    if check_full_csv:
        variants.append(("csv", False, True))
    for name, compressed, csv_samples in variants:
        with tempfile.TemporaryDirectory(prefix="fel-validation-", dir=scratch_parent) as scratch:
            destination = Path(scratch) / "export"
            started = time.perf_counter()
            decoded = decode_file(path)
            export_decoded(decoded, destination, source_name=path.name,
                           compress=compressed, csv_samples=csv_samples)
            duration = time.perf_counter() - started
            require(decoded.source_sha256 == reference.source_sha256, "Input changed during export validation")
            result["variants"][name] = {
                "read_decode_write_seconds": duration,
                "npz_arrays_checked": check_export_arrays(destination, reference),
                "csv_rows_checked": check_export_csv(destination, reference, csv_samples),
                "output_bytes": sum(file.stat().st_size for file in destination.iterdir()),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3, help="Repeated decode/read/hash timing runs (default: 3)")
    parser.add_argument("--export-output", type=Path,
                        help="Also validate default/compressed exports for every file and full CSVs for the first file containing transients and each previously unchecked named group; write a separate private JSON report")
    args = parser.parse_args()
    require(args.runs >= 3, "Use at least three benchmark runs")
    report = {"status": "running", "validation_scope": [
        "All extracted raw samples, scaled values, event fields, selected trend triples, tag-70/71 float fields, periods, timestamps and record boundaries checked against an independent scalar struct reader.",
        "Vendor event and transient CSVs, when supplied, independently validate their supported values and timestamps.",
        "Waveform and trend checks are structural/extraction checks; tag-70/71 labels are shared from the decoder registry. None establish independent vendor-value validation or channel electrical meaning.",
        "Event/transient vendor CSVs do not validate tag-70/71 energy trends or demand. Run validate_energy_exports.py against native tables for that separate evidence.",
        "Tolerance 0.0006 covers decimal rounding of the available vendor exports; this is not an instrument accuracy specification.",
        "Benchmark includes file reading, framing, decoding and SHA-256; repeated reads can use the OS cache. CSV validation and output writing are excluded.",
    ], "recordings": []}
    export_report = {"status": "running", "recordings": [], "timing_scope":
                     "Single end-to-end read/decode/write per variant, warm OS cache; validation/reloading excluded. No fsync durability claim. Temporary exports are removed after checking."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_checked = False
    named_csv_checked = set()
    try:
        manifest = json.loads(args.manifest.read_text())
        require(bool(manifest.get("recordings")), "Manifest must list recordings")
        for item in manifest["recordings"]:
            path = resolve_path(item["fel"], args.manifest.parent)
            data = path.read_bytes()
            decoded = decode_bytes(data)
            result = {"source": str(path), "source_bytes": len(data), "source_sha256": decoded.source_sha256,
                      "struct_reference": validate_struct(data, decoded)}
            if item.get("events_csv") or item.get("transient_csv"):
                require(bool(item.get("timezone")), "Specify the vendor export timezone explicitly")
                local_zone = ZoneInfo(item["timezone"])
                if item.get("events_csv"):
                    result["vendor_events"] = validate_events(resolve_path(item["events_csv"], args.manifest.parent),
                                                               decoded.groups["events"], local_zone)
                if item.get("transient_csv"):
                    result["vendor_transients"] = validate_transients(resolve_path(item["transient_csv"], args.manifest.parent),
                                                                       decoded.groups["transients"], local_zone)
            if args.export_output:
                has_transients = len(decoded.groups["transients"]["values"]) > 0
                named_present = {name for name in ("energy_trends", "demand") if name in decoded.groups}
                csv_this_file = ((not csv_checked and has_transients)
                                 or bool(named_present - named_csv_checked))
                export_report["recordings"].append(validate_exports(path, decoded, args.output.parent, csv_this_file))
                csv_checked = csv_checked or (csv_this_file and has_transients)
                if csv_this_file:
                    named_csv_checked.update(named_present)
            del decoded, data
            durations = []
            for _ in range(args.runs):
                started = time.perf_counter()
                decoded = decode_file(path)
                durations.append(time.perf_counter() - started)
                require(decoded.source_sha256 == result["source_sha256"], "Input changed during validation")
                del decoded
            median = statistics.median(durations)
            result["benchmark"] = {"runs_seconds": durations, "median_seconds": median,
                                   "median_megabytes_per_second": result["source_bytes"] / 1000000 / median}
            report["recordings"].append(result)
            print(f"PASS {path.name}: {result['struct_reference']['record_count']} records; median {median:.4f} s", flush=True)
        total_bytes = sum(r["source_bytes"] for r in report["recordings"])
        total_seconds = sum(r["benchmark"]["median_seconds"] for r in report["recordings"])
        report["summary"] = {
            "files": len(report["recordings"]), "bytes": total_bytes,
            "records": sum(r["struct_reference"]["record_count"] for r in report["recordings"]),
            "vendor_event_rows": sum(r.get("vendor_events", {}).get("rows_checked", 0) for r in report["recordings"]),
            "vendor_transient_sample_rows": sum(r.get("vendor_transients", {}).get("sample_rows_checked", 0) for r in report["recordings"]),
            "named_layout_structural_checks": {
                key: sum(r["struct_reference"]["checked"].get(key, 0) for r in report["recordings"])
                for key in ("energy_trends_records", "energy_trends_scalar_values", "demand_records", "demand_scalar_values")
            },
            "sum_median_decode_seconds": total_seconds,
            "aggregate_megabytes_per_second": total_bytes / 1000000 / total_seconds,
        }
        report["status"] = "passed"
        export_report["status"] = "passed"
        export_report["full_sample_csv_checked"] = csv_checked
        export_report["named_csv_groups_checked"] = sorted(named_csv_checked)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        export_report["status"] = "failed"
        export_report["error"] = report["error"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.export_output:
        args.export_output.parent.mkdir(parents=True, exist_ok=True)
        args.export_output.write_text(json.dumps(export_report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output), "summary": report.get("summary"),
                      "error": report.get("error")}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
