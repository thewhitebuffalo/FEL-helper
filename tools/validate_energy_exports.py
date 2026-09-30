#!/usr/bin/env python3
"""Compare decoded energy/demand fields with explicitly mapped native Parquet.

Inputs and mappings stay private. The exclusive-create JSON result contains only
code provenance, aggregate counts/errors, and decoder field names. Native input
provenance is a declaration by the operator; synthetic results are marked as such.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fel_decoder  # noqa: E402
from fel_decoder import decode_file  # noqa: E402
from fel_decoder.layouts import DEMAND_FIELDS, ENERGY_TREND_FIELDS  # noqa: E402

GROUPS = {"energy_trends": (ENERGY_TREND_FIELDS, "trend_period"),
          "demand": (DEMAND_FIELDS, "demand_period")}
UNIX_EPOCH_TICKS = 116444736000000000


class ValidationFailure(Exception):
    """Only allow controlled diagnostics into the publishable report."""

    def __init__(self, code, group=None, field=None):
        self.diagnostic = {"code": code}
        if group in GROUPS:
            self.diagnostic["group"] = group
            permitted = set(GROUPS[group][0]) | {GROUPS[group][1], "start_ticks", "end_ticks"}
            if field in permitted:
                self.diagnostic["field"] = field
        super().__init__(code)


def require(condition, code, group=None, field=None):
    if not condition:
        raise ValidationFailure(code, group, field)


def _git(*arguments):
    try:
        return subprocess.run(["git", *arguments], cwd=ROOT, check=True,
                              capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        raise ValidationFailure("git_provenance_unavailable") from None


def source_provenance():
    """Verify the running module paths/content, not an installed package version."""
    require(Path(_git("rev-parse", "--show-toplevel").decode().strip()).resolve() == ROOT,
            "unexpected_repository_root")
    commit = _git("rev-parse", "--verify", "HEAD").decode().strip()
    require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is not None, "invalid_commit_id")
    dirty = bool(_git("status", "--porcelain=v1", "--untracked-files=all"))
    actual = {"tools/validate_energy_exports.py": Path(__file__).resolve()}
    for module_name, relative in (("fel_decoder", "src/fel_decoder/__init__.py"),
                                  ("fel_decoder._version", "src/fel_decoder/_version.py"),
                                  ("fel_decoder.core", "src/fel_decoder/core.py"),
                                  ("fel_decoder.layouts", "src/fel_decoder/layouts.py")):
        module = importlib.import_module(module_name)
        actual[relative] = Path(module.__file__).resolve()
    for relative, module_path in actual.items():
        require(module_path == (ROOT / relative).resolve(), "decoder_imported_outside_checkout")
        # Verifies tracked source and catches assume-unchanged/skip-worktree cases.
        try:
            expected = _git("show", f"{commit}:{relative}")
        except ValidationFailure:
            raise ValidationFailure("validator_or_decoder_not_in_commit") from None
        if module_path.read_bytes() != expected:
            dirty = True
    return {"git_commit": commit, "dirty": dirty, "package_version": fel_decoder.__version__,
            "import_verified_against_checkout": True}


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ValidationFailure("pyarrow_not_installed") from None
    return pa, pq


def _integers(column, group, field):
    pa, _ = _arrow()
    require(pa.types.is_integer(column.type), "integer_column_required", group, field)
    require(column.null_count == 0, "null_not_a_nan_or_integer", group, field)
    return column.to_pylist()  # Python ints: no float conversion or uint64 overflow.


def timestamp_ticks(column, encoding, group, field):
    """Explicit epoch/unit conversions using integer arithmetic only."""
    pa, _ = _arrow()
    require(isinstance(encoding, dict), "timestamp_encoding_required", group, field)
    kind = encoding.get("encoding")
    if kind == "filetime_100ns":
        require(set(encoding) == {"column", "encoding"}, "invalid_timestamp_mapping", group, field)
        values = _integers(column, group, field)
    elif kind in ("unix_integer", "arrow_timestamp_utc"):
        if kind == "unix_integer":
            require(set(encoding) == {"column", "encoding", "unit"}, "invalid_timestamp_mapping", group, field)
            values = _integers(column, group, field)
            unit = encoding["unit"]
        else:
            require(set(encoding) == {"column", "encoding"}, "invalid_timestamp_mapping", group, field)
            require(pa.types.is_timestamp(column.type), "arrow_timestamp_required", group, field)
            require(column.null_count == 0, "null_not_a_nan_or_integer", group, field)
            # Arrow timestamps are integers since Unix epoch; tz-naive input is
            # explicitly asserted UTC by this encoding. No local-zone inference.
            unit = column.type.unit
            values = column.cast(pa.int64()).to_pylist()
        require(unit in ("s", "ms", "us", "ns", "100ns"), "unsupported_time_unit", group, field)
        if unit == "ns":
            require(all(value % 100 == 0 for value in values), "timestamp_finer_than_100ns", group, field)
            values = [value // 100 for value in values]
        else:
            factor = {"s": 10000000, "ms": 10000, "us": 10, "100ns": 1}[unit]
            values = [value * factor for value in values]
        values = [value + UNIX_EPOCH_TICKS for value in values]
    else:
        raise ValidationFailure("unsupported_timestamp_encoding", group, field)
    require(all(0 <= value <= 2**64 - 1 for value in values), "timestamp_out_of_range", group, field)
    return values


def _compare(expected, actual, *, integers=False):
    """Exact numeric comparison; never narrow vendor float64 to decoder float32."""
    mismatch, nan_count, nonfinite, maximum = 0, 0, 0, 0
    for decoded, vendor in zip(expected, actual):
        decoded = int(decoded) if integers else float(decoded)
        vendor_nan = not integers and isinstance(vendor, float) and math.isnan(vendor)
        decoded_nan = not integers and math.isnan(decoded)
        if vendor_nan or decoded_nan:
            if vendor_nan and decoded_nan:
                nan_count += 1
            else:
                mismatch += 1
                nonfinite += 1
            continue
        if decoded != vendor:
            mismatch += 1
            if integers:
                maximum = max(maximum, abs(decoded - vendor))
            elif math.isfinite(decoded) and math.isfinite(vendor):
                difference = abs(decoded - vendor)
                if math.isfinite(difference):
                    maximum = max(maximum, difference)
                else:
                    nonfinite += 1
            else:
                nonfinite += 1
    return mismatch, nan_count, nonfinite, maximum


def compare_group(group_name, arrays, table, mapping):
    """Check every row/field, including ordered timestamps and nominal periods."""
    pa, _ = _arrow()
    fields, period = GROUPS[group_name]
    require(isinstance(mapping, dict), "invalid_group_mapping", group_name)
    require(set(mapping) <= {"parquet", "parquet_stream", "fields", "start_ticks", "end_ticks", "period", "ignored_columns"},
            "invalid_group_mapping", group_name)
    require({"fields", "start_ticks", "end_ticks", "period"} <= set(mapping)
            and len(set(mapping) & {"parquet", "parquet_stream"}) == 1,
            "incomplete_group_mapping", group_name)
    require(isinstance(mapping["fields"], dict) and set(mapping["fields"]) == set(fields),
            "all_named_fields_must_be_mapped", group_name)
    require(set(arrays) == set(fields) | {"record_offset", "start_ticks", "end_ticks", period},
            "unexpected_decoder_group_schema", group_name)
    rows = len(arrays["start_ticks"])
    require(rows > 0 and table.num_rows == rows, "row_count_mismatch", group_name)
    require(all(len(values) == rows for values in arrays.values()), "decoder_array_length_mismatch", group_name)
    require(len(set(table.column_names)) == len(table.column_names), "duplicate_parquet_columns", group_name)
    selected = dict(mapping["fields"])
    for field in ("start_ticks", "end_ticks"):
        require(isinstance(mapping[field], dict), "timestamp_encoding_required", group_name, field)
        selected[field] = mapping[field].get("column")
    period_mapping = mapping["period"]
    require(isinstance(period_mapping, dict) and set(period_mapping) == {"column", "encoding"}
            and period_mapping["encoding"] == "seconds_integer", "invalid_period_mapping", group_name, period)
    selected[period] = period_mapping["column"]
    require(all(isinstance(column, str) and column for column in selected.values()),
            "invalid_column_mapping", group_name)
    require(len(set(selected.values())) == len(selected), "column_mapped_more_than_once", group_name)
    ignored = mapping.get("ignored_columns", [])
    require(isinstance(ignored, list) and all(isinstance(name, str) for name in ignored)
            and len(set(ignored)) == len(ignored), "invalid_ignored_columns", group_name)
    require(not set(ignored) & set(selected.values()), "mapped_column_cannot_be_ignored", group_name)
    require(set(selected.values()) | set(ignored) == set(table.column_names), "parquet_column_set_mismatch", group_name)
    result = {"rows": rows, "named_fields": len(fields), "named_value_comparisons": rows * len(fields),
              "timestamp_comparisons": rows * 2, "period_comparisons": rows,
              "matched_nan_cells": 0, "mismatched_cells": 0, "nonfinite_mismatches": 0,
              "max_absolute_value_error": 0.0, "max_timestamp_error_ticks": 0,
              "max_period_error_seconds": 0, "diagnostics": []}
    for field, column_name in selected.items():
        column = table.column(column_name)
        integer = field in ("start_ticks", "end_ticks", period)
        if field in ("start_ticks", "end_ticks"):
            values = timestamp_ticks(column, mapping[field], group_name, field)
            error_key = "max_timestamp_error_ticks"
        elif field == period:
            values = _integers(column, group_name, field)
            require(all(0 <= value <= 2**32 - 1 for value in values), "period_out_of_range", group_name, field)
            error_key = "max_period_error_seconds"
        else:
            require(pa.types.is_floating(column.type) or pa.types.is_integer(column.type),
                    "numeric_measurement_column_required", group_name, field)
            require(column.null_count == 0, "null_not_a_nan_or_integer", group_name, field)
            values = column.to_pylist()
            error_key = "max_absolute_value_error"
        mismatch, nans, nonfinite, maximum = _compare(arrays[field], values, integers=integer)
        result["matched_nan_cells"] += nans
        result["mismatched_cells"] += mismatch
        result["nonfinite_mismatches"] += nonfinite
        result[error_key] = max(result[error_key], maximum)
        if mismatch:
            result["diagnostics"].append({"field": field, "code": "exact_comparison_failed", "count": mismatch})
    return result


def _input_path(value, base):
    require(isinstance(value, str) and bool(value), "invalid_input_path")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _stream_path(value):
    require(isinstance(value, list) and bool(value)
            and all(isinstance(part, str) and bool(part) for part in value), "invalid_native_stream_path")
    return value


def _read_tables(entry, manifest_path, kind, decoded):
    """Native mode reads the original OLE import and verifies its embedded FEL."""
    pa, parquet = _arrow()
    if "native_container" not in entry:
        require(kind == "synthetic", "native_container_required")
        tables = {}
        for name in GROUPS:
            mapping = entry[name]
            require(isinstance(mapping, dict) and "parquet" in mapping and "parquet_stream" not in mapping,
                    "parquet_path_required", name)
            vendor_path = _input_path(mapping["parquet"], manifest_path.parent)
            try:
                tables[name] = parquet.read_table(vendor_path)
            except Exception:
                raise ValidationFailure("parquet_unreadable", name) from None
        return tables, False
    config = entry["native_container"]
    require(isinstance(config, dict) and set(config) == {"path", "embedded_fel_stream"}, "invalid_native_container")
    try:
        import olefile
    except ImportError:
        raise ValidationFailure("olefile_not_installed") from None
    container_path = _input_path(config["path"], manifest_path.parent)
    embedded_stream = _stream_path(config["embedded_fel_stream"])
    try:
        with olefile.OleFileIO(str(container_path)) as container:
            embedded = container.openstream(embedded_stream).read()
            # The source hash stays private; only the identity-check outcome is reported.
            require(hashlib.sha256(embedded).hexdigest() == decoded.source_sha256,
                    "embedded_fel_does_not_match_decoded_source")
            source_path = _input_path(entry["fel"], manifest_path.parent)
            require(embedded == source_path.read_bytes(), "embedded_fel_bytes_differ")
            tables = {}
            for name in GROUPS:
                mapping = entry[name]
                require(isinstance(mapping, dict) and "parquet_stream" in mapping and "parquet" not in mapping,
                        "parquet_stream_required", name)
                stream = _stream_path(mapping["parquet_stream"])
                tables[name] = parquet.read_table(pa.BufferReader(container.openstream(stream).read()))
            return tables, True
    except ValidationFailure:
        raise
    except Exception:
        raise ValidationFailure("native_container_unreadable") from None


def validate_manifest(manifest_path):
    """Return a sanitized report; fail closed on dirty/stale/untracked source."""
    report = {"report_schema_version": 1, "status": "failed", "evidence_kind": "unconfirmed",
              "comparison": "Exact equality; no tolerance, unit inference, clock correction, row sorting, or float narrowing.",
              "provenance_scope": "Operator declares native-export origin. Native mode verifies byte identity with the FEL embedded in the same OLE container as both tables; it does not authenticate the vendor or calibration.",
              "recordings": [], "diagnostics": []}
    try:
        source = source_provenance()
        report["source"] = source
        require(not source["dirty"], "checkout_not_clean")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValidationFailure("manifest_unreadable") from None
        require(isinstance(manifest, dict) and set(manifest) == {"schema_version", "evidence_kind", "recordings"}
                and manifest["schema_version"] == 1, "invalid_manifest_schema")
        require(manifest["evidence_kind"] in ("native_vendor_export", "synthetic"), "invalid_evidence_kind")
        report["evidence_kind"] = manifest["evidence_kind"]
        recordings = manifest["recordings"]
        require(isinstance(recordings, list) and bool(recordings), "no_recordings")
        _arrow()
        seen, schemas = set(), set()
        for index, entry in enumerate(recordings):
            require(isinstance(entry, dict) and {"fel", *GROUPS} <= set(entry)
                    and set(entry) <= {"fel", "native_container", *GROUPS}, "both_groups_required")
            path = _input_path(entry["fel"], manifest_path.parent)
            require(path not in seen, "duplicate_recording_path")
            seen.add(path)
            try:
                decoded = decode_file(path)
            except (OSError, ValueError):
                raise ValidationFailure("fel_decode_failed") from None
            schemas.add(decoded.manifest()["schema_version"])
            tables, identity_checked = _read_tables(entry, manifest_path, manifest["evidence_kind"], decoded)
            result = {"recording_index": index, "embedded_source_byte_identity_checked": identity_checked, "groups": {}}
            report["recordings"].append(result)
            for name in GROUPS:
                require(name in decoded.groups, "required_group_absent", name)
                mapping = entry[name]
                result["groups"][name] = compare_group(name, decoded.groups[name], tables[name], mapping)
        report["source"]["manifest_schema_versions"] = sorted(schemas)
        totals = {"recordings": len(recordings), "group_tables": 0, "rows": 0,
                  "named_value_comparisons": 0, "timestamp_comparisons": 0, "period_comparisons": 0,
                  "matched_nan_cells": 0, "mismatched_cells": 0, "nonfinite_mismatches": 0,
                  "max_absolute_value_error": 0.0, "max_timestamp_error_ticks": 0,
                  "max_period_error_seconds": 0}
        for result in report["recordings"]:
            for group in result["groups"].values():
                totals["group_tables"] += 1
                for key in totals:
                    if key in group:
                        totals[key] = max(totals[key], group[key]) if key.startswith("max_") else totals[key] + group[key]
        report["summary"] = totals
        final_source = source_provenance()
        require(not final_source["dirty"] and final_source["git_commit"] == source["git_commit"]
                and final_source["package_version"] == source["package_version"], "source_changed_during_validation")
        require(totals["mismatched_cells"] == 0, "native_table_comparison_failed")
        report["status"] = "passed" if manifest["evidence_kind"] == "native_vendor_export" else "synthetic_passed"
    except ValidationFailure as error:
        report["diagnostics"].append(error.diagnostic)
    except Exception:
        # Never propagate exception text from client filenames or Parquet metadata.
        report["diagnostics"].append({"code": "unexpected_validation_error"})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Private JSON manifest with explicit column mappings")
    parser.add_argument("--output", type=Path, required=True, help="New sanitized report; existing paths are refused")
    args = parser.parse_args(argv)
    if args.output.exists():
        print("Validation report output already exists; no files changed.", file=sys.stderr)
        return 2
    report = validate_manifest(args.manifest.resolve())
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except OSError:
        print("Could not exclusively create the report; inputs were not modified.", file=sys.stderr)
        return 2
    print(f"Validation status: {report['status']}. Report contains aggregate results only.")
    return 0 if report["status"] in ("passed", "synthetic_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
