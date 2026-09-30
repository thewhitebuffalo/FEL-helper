"""Fault-injection tests for the local scalar validation harness.

The generated field labels use the registry; this establishes structural
coverage, not independent vendor verification of electrical meanings.
"""

import copy
import csv
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from fel_decoder import decode_bytes
from fel_decoder.export import export_decoded
from tests.test_decoder import START, event, fel, record, transient, trend, wave
from tools.validate_local import check_export_csv, validate_struct


def named_record(tag, *, start=START):
    size, count = (744, 180) if tag == 70 else (104, 20)
    data = record(tag, size, start=start, end=start + 6_000_000_001)
    struct.pack_into("<I", data, 20, 600)
    for index in range(count):
        struct.pack_into("<f", data, 24 + 4 * index, (-1 if index % 2 else 1) * (index + 0.25))
    # Missing values must remain missing even at the last registered field.
    struct.pack_into("<f", data, size - 4, float("nan"))
    return data


def mixed_data():
    # Partial-capacity waveform/transient records exercise valid padding too.
    return fel(event(), named_record(70), wave(), named_record(71), trend(), transient(),
               named_record(70, start=START + 100_000_000_000))


def test_scalar_reference_checks_every_registered_named_value_and_boundaries():
    data = mixed_data()
    result = validate_struct(data, decode_bytes(data))
    assert result["checked"]["energy_trends_records"] == 2
    assert result["checked"]["energy_trends_scalar_values"] == 360
    assert result["checked"]["demand_records"] == 1
    assert result["checked"]["demand_scalar_values"] == 20
    assert result["checked"]["events"] == 1
    assert result["checked"]["waveforms_sample_rows"] == 1
    assert result["checked"]["transients_sample_rows"] == 2
    assert result["validation_kind"] == "structural_extraction"
    assert result["coverage"]["supported_tags"] == [70, 71, 105, 111, 112, 119]
    assert result["coverage"]["vendor_energy_trend_or_demand_validation"] is False
    assert "not independently validated" in result["coverage"]["named_field_labels"]


@pytest.mark.parametrize("tag,name,count", [(70, "energy_trends", 180), (71, "demand", 20)])
def test_each_named_scalar_is_actually_checked(tag, name, count):
    data = fel(named_record(tag))
    original = decode_bytes(data)
    fields = [key for key, values in original.groups[name].items() if values.dtype == np.dtype("float32")]
    assert len(fields) == count
    for field in fields:
        decoded = copy.deepcopy(original)
        value = decoded.groups[name][field][0]
        decoded.groups[name][field][0] = 17 if np.isnan(value) else value + 0.5
        with pytest.raises(AssertionError, match="differs from scalar struct reference"):
            validate_struct(data, decoded)


@pytest.mark.parametrize("tag,name,period", [(70, "energy_trends", "trend_period"), (71, "demand", "demand_period")])
@pytest.mark.parametrize("field", ["record_offset", "start_ticks", "end_ticks", "period"])
def test_named_metadata_corruption_fails(tag, name, period, field):
    data = fel(named_record(tag))
    decoded = decode_bytes(data)
    key = period if field == "period" else field
    decoded.groups[name][key][0] += 1
    with pytest.raises(AssertionError, match="wrong"):
        validate_struct(data, decoded)


@pytest.mark.parametrize("mutation", ["missing", "order", "dtype", "extra_row", "extra_field"])
def test_named_schema_mistakes_fail(mutation):
    data = fel(named_record(70))
    decoded = decode_bytes(data)
    group = decoded.groups["energy_trends"]
    field = "current_rms_a_max"
    if mutation == "missing":
        del group[field]
    elif mutation == "order":
        group[field] = group.pop(field)
    elif mutation == "dtype":
        group[field] = group[field].astype(np.float64)
    elif mutation == "extra_row":
        group[field] = np.append(group[field], group[field])
    else:
        group["unexpected"] = np.array([1], dtype=np.float32)
    with pytest.raises(AssertionError):
        validate_struct(data, decoded)


def test_named_groups_absent_when_no_matching_records_exist():
    data = fel(event())
    decoded = decode_bytes(data)
    assert validate_struct(data, decoded)["checked"] == {"events": 1}
    decoded.groups["demand"] = {"record_offset": np.array([], dtype=np.uint64)}
    with pytest.raises(AssertionError, match="Unexpected empty demand group"):
        validate_struct(data, decoded)


def test_named_group_missing_or_extra_records_fail():
    data = fel(named_record(71))
    decoded = decode_bytes(data)
    del decoded.groups["demand"]
    with pytest.raises(AssertionError, match="Missing decoded group"):
        validate_struct(data, decoded)
    decoded = decode_bytes(data)
    group = decoded.groups["demand"]
    for key, values in list(group.items()):
        group[key] = np.concatenate([values, values])
    with pytest.raises(AssertionError, match="Wrong demand output length"):
        validate_struct(data, decoded)


def test_named_csv_roundtrip_is_checked_and_corruption_rejected(tmp_path):
    decoded = decode_bytes(mixed_data())
    output = export_decoded(decoded, tmp_path / "output", source_name="synthetic.fel", csv_samples=True)
    checked = check_export_csv(output, decoded, True)
    assert checked["energy_trends"] == 2
    assert checked["demand"] == 1
    assert set(check_export_csv(output, decoded, False)) == {"records", "events"}
    path = output / "demand.csv"
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    rows[1][rows[0].index("current_rms_c")] = "999"
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(AssertionError, match="float round trip differs"):
        check_export_csv(output, decoded, True)


def test_energy_only_cli_report_distinguishes_structural_from_vendor_evidence(tmp_path):
    source = tmp_path / "synthetic.fel"
    source.write_bytes(fel(named_record(70), named_record(71)))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"recordings": [{"fel": source.name}]}))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "tools" / "validate_local.py"),
         "--manifest", str(manifest), "--output", str(tmp_path / "report.json"),
         "--export-output", str(tmp_path / "exports.json"), "--runs", "3"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "passed"
    assert report["summary"]["vendor_event_rows"] == 0
    assert report["summary"]["vendor_transient_sample_rows"] == 0
    assert report["summary"]["named_layout_structural_checks"] == {
        "energy_trends_records": 1, "energy_trends_scalar_values": 180,
        "demand_records": 1, "demand_scalar_values": 20,
    }
    assert any("do not validate tag-70/71" in scope for scope in report["validation_scope"])
    exports = json.loads((tmp_path / "exports.json").read_text())
    assert exports["named_csv_groups_checked"] == ["demand", "energy_trends"]
    assert exports["full_sample_csv_checked"] is False
    assert exports["recordings"][0]["variants"]["csv"]["csv_rows_checked"]["energy_trends"] == 1
