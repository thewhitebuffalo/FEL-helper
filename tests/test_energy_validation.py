"""Independent synthetic oracle tables test the validator, not vendor evidence."""

import copy
import json
import struct

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fel_decoder import decode_file
from tools import validate_energy_exports as check


# Independent wire-order fixture definitions; deliberately not imported from
# production layouts. Unique values identify every permutation of these fields.
STATS = ("min", "max", "avg")
ENERGY_FIELDS = tuple(
    [f"voltage_rms_{phase}_{stat}" for phase in ("an", "bn", "cn", "ab", "bc", "ca") for stat in STATS]
    + [f"current_rms_{phase}_{stat}" for phase in "abc" for stat in STATS]
    + [f"{quantity}_thd_{phase}_{stat}" for quantity in ("voltage", "current") for phase in "abc" for stat in STATS]
    + [f"frequency_{stat}" for stat in STATS]
    + [f"power_{basis}_{kind}_{phase}_{stat}" for basis in ("rms", "fund")
       for kind in ("active", "apparent", "nonactive") for phase in ("a", "b", "c", "total") for stat in STATS]
    + [f"powfac_{kind}_{phase}_{stat}" for kind in ("pf", "dpf") for phase in ("a", "b", "c", "total") for stat in STATS]
    + [f"energy_{kind}_{phase}" for kind in ("active", "apparent", "nonactive") for phase in ("a", "b", "c", "total")]
    + ["energy_consumed_total", "energy_supplied_total"]
    + [f"aux{aux}_{stat}" for aux in (1, 2) for stat in STATS]
    + [f"current_rms_n_{stat}" for stat in STATS]
    + [f"voltage_symm_comp_{kind}" for kind in ("negative", "positive", "zero")]
    + ["voltage_effective", "current_effective"]
    + [f"{quantity}_fund_{phase}{'n' if quantity == 'voltage' else ''}"
       for quantity in ("voltage", "current") for phase in "abc"]
    + ["voltage_fund_effective", "current_fund_effective"]
)
DEMAND_FIELDS = tuple(
    [f"voltage_rms_{phase}n" for phase in "abc"] + [f"current_rms_{phase}" for phase in "abc"]
    + [f"energy_{kind}_{phase}" for kind in ("active", "apparent", "nonactive") for phase in ("a", "b", "c", "total")]
    + ["energy_consumed_total", "energy_supplied_total"]
)
START = 133000000123456789


@pytest.fixture
def fixture(tmp_path):
    records, tables, maps = [], {}, {}
    for name, tag, fields, period_name in (("energy_trends", 70, ENERGY_FIELDS, "trend_period"),
                                           ("demand", 71, DEMAND_FIELDS, "demand_period")):
        assert len(fields) == (180 if tag == 70 else 20)
        data = {field: [] for field in fields}
        starts, ends, periods = [], [], []
        for row in range(2):
            start, period = START + row * 90_000_000_007, 300 + row * 300
            end = start + period * 10_000_000 - 1
            raw = bytearray(24 + len(fields) * 4)
            struct.pack_into("<HH", raw, 0, tag, len(raw))
            struct.pack_into("<II", raw, 4, start >> 32, start & 0xffffffff)
            struct.pack_into("<II", raw, 12, end >> 32, end & 0xffffffff)
            struct.pack_into("<I", raw, 20, period)
            for index, field in enumerate(fields):
                value = index + row * 1024 + 0.25
                if field == "power_rms_active_total_max":
                    value = -45000.5 - row
                if field == "aux1_avg" and row == 1:
                    value = float("nan")
                struct.pack_into("<f", raw, 24 + 4 * index, value)
                data[field].append(value)
            # Independent known offsets guard fixture field-order mistakes.
            if tag == 70:
                assert struct.unpack_from("<f", raw, 100)[0] == data["current_rms_a_max"][-1]
                assert struct.unpack_from("<f", raw, 256)[0] == -45000.5 - row
            records.append(raw)
            starts.append(start)
            ends.append(end)
            periods.append(period)
        columns = {"vendor_start": pa.array(starts, type=pa.int64()),
                   "vendor_end": pa.array(ends, type=pa.int64()),
                   "vendor_period": pa.array(periods, type=pa.uint32())}
        columns.update({"vendor_" + field: pa.array(values, type=pa.float64()) for field, values in data.items()})
        tables[name] = pa.table(columns)
        maps[name] = {"parquet": str(tmp_path / (name + ".parquet")),
                      "fields": {field: "vendor_" + field for field in fields},
                      "start_ticks": {"column": "vendor_start", "encoding": "filetime_100ns"},
                      "end_ticks": {"column": "vendor_end", "encoding": "filetime_100ns"},
                      "period": {"column": "vendor_period", "encoding": "seconds_integer"}}
    body = b"".join(records)
    header = bytearray(bytes.fromhex("0bba1400") + bytes(20) + bytes.fromhex("0000090000000000"))
    struct.pack_into("<I", header, 4, len(body))
    fel = tmp_path / "synthetic.fel"
    fel.write_bytes(header + body)
    return fel, decode_file(fel), tables, maps


def changed_column(table, name, values, dtype=pa.float64()):
    return table.set_column(table.column_names.index(name), name, pa.array(values, type=dtype))


def compare(fixture, group="energy_trends", table=None, mapping=None):
    _, decoded, tables, maps = fixture
    return check.compare_group(group, decoded.groups[group], tables[group] if table is None else table,
                               maps[group] if mapping is None else mapping)


def test_every_named_field_and_nan_is_checked(fixture):
    a, b = compare(fixture), compare(fixture, "demand")
    assert a["named_fields"] == 180 and b["named_fields"] == 20
    assert a["named_value_comparisons"] + b["named_value_comparisons"] == 400
    assert a["matched_nan_cells"] == 1
    assert a["mismatched_cells"] == b["mismatched_cells"] == 0


def test_float64_difference_that_float32_would_hide_fails(fixture):
    table = fixture[2]["energy_trends"]
    key = "vendor_current_rms_c_max"
    values = table[key].to_pylist()
    values[0] = np.nextafter(values[0], float("inf"))
    assert np.float32(values[0]) == np.float32(table[key][0].as_py())
    result = compare(fixture, table=changed_column(table, key, values))
    assert result["mismatched_cells"] == 1 and result["max_absolute_value_error"] > 0


def test_nan_is_preserved_and_cannot_match_a_number(fixture):
    table = fixture[2]["energy_trends"]
    values = table["vendor_aux1_avg"].to_pylist()
    values[1] = 0.0
    result = compare(fixture, table=changed_column(table, "vendor_aux1_avg", values))
    assert result["mismatched_cells"] == result["nonfinite_mismatches"] == 1


def test_swapped_columns_and_changed_period_fail(fixture):
    mapping = copy.deepcopy(fixture[3]["energy_trends"])
    x, y = "current_rms_a_max", "current_rms_c_max"
    mapping["fields"][x], mapping["fields"][y] = mapping["fields"][y], mapping["fields"][x]
    assert compare(fixture, mapping=mapping)["mismatched_cells"] == 4
    table = changed_column(fixture[2]["demand"], "vendor_period", [301, 600], pa.uint32())
    assert compare(fixture, "demand", table=table)["max_period_error_seconds"] == 1


def test_timestamp_one_tick_and_float_precision(fixture):
    table = fixture[2]["energy_trends"]
    values = table["vendor_start"].to_pylist()
    values[0] += 1
    result = compare(fixture, table=changed_column(table, "vendor_start", values, pa.int64()))
    assert result["mismatched_cells"] == result["max_timestamp_error_ticks"] == 1
    with pytest.raises(check.ValidationFailure, match="integer_column_required"):
        compare(fixture, table=changed_column(table, "vendor_start", [float(value) for value in values]))


@pytest.mark.parametrize("encoding", ["unix_integer", "arrow_timestamp_utc"])
def test_explicit_integer_and_arrow_timestamp_conversion(fixture, encoding):
    mapping = copy.deepcopy(fixture[3]["energy_trends"])
    table = fixture[2]["energy_trends"]
    for name, key in (("start_ticks", "vendor_start"), ("end_ticks", "vendor_end")):
        values = [(value - check.UNIX_EPOCH_TICKS) * 100 for value in table[key].to_pylist()]
        kind = pa.int64() if encoding == "unix_integer" else pa.timestamp("ns", tz="UTC")
        table = changed_column(table, key, values, kind)
        mapping[name] = {"column": key, "encoding": encoding}
        if encoding == "unix_integer":
            mapping[name]["unit"] = "ns"
    assert compare(fixture, table=table, mapping=mapping)["mismatched_cells"] == 0
    values[0] += 1
    table = changed_column(table, "vendor_end", values, kind)
    with pytest.raises(check.ValidationFailure, match="timestamp_finer_than_100ns"):
        compare(fixture, table=table, mapping=mapping)


@pytest.mark.parametrize("mutation", ["missing_row", "extra_row", "missing_field", "duplicate_map", "extra_column", "null"])
def test_incomplete_or_ambiguous_oracles_fail(fixture, mutation):
    table, mapping = fixture[2]["energy_trends"], copy.deepcopy(fixture[3]["energy_trends"])
    if mutation == "missing_row": table = table.slice(0, 1)
    if mutation == "extra_row": table = pa.concat_tables([table, table.slice(0, 1)])
    if mutation == "missing_field": mapping["fields"].pop("frequency_avg")
    if mutation == "duplicate_map": mapping["fields"]["frequency_avg"] = mapping["fields"]["frequency_min"]
    if mutation == "extra_column": table = table.append_column("private_extra", pa.array([1, 2]))
    if mutation == "null": table = changed_column(table, "vendor_aux1_avg", [0.25, None])
    with pytest.raises(check.ValidationFailure): compare(fixture, table=table, mapping=mapping)


def write_manifest(fixture, tmp_path, kind="synthetic"):
    fel, _, tables, mappings = fixture
    for name, table in tables.items(): pq.write_table(table, mappings[name]["parquet"])
    path = tmp_path / "private-manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "evidence_kind": kind,
                                "recordings": [{"fel": str(fel), **mappings}]}))
    return path


def clean_provenance():
    return {"git_commit": "b" * 40, "dirty": False, "package_version": "synthetic-test-version",
            "import_verified_against_checkout": True}


def test_report_is_explicitly_synthetic_sanitized_and_exclusive(fixture, tmp_path, monkeypatch):
    monkeypatch.setattr(check, "source_provenance", clean_provenance)
    manifest = write_manifest(fixture, tmp_path)
    output = tmp_path / "aggregate.json"
    assert check.main(["--manifest", str(manifest), "--output", str(output)]) == 0
    text, report = output.read_text(), json.loads(output.read_text())
    assert report["status"] == "synthetic_passed" and report["evidence_kind"] == "synthetic"
    assert report["summary"]["named_value_comparisons"] == 400
    assert report["source"]["manifest_schema_versions"] == [2]
    assert str(tmp_path) not in text and "synthetic.fel" not in text and "vendor_" not in text
    assert "45000.5" not in text and fixture[1].source_sha256 not in text
    assert check.main(["--manifest", str(manifest), "--output", str(output)]) == 2
    assert output.read_text() == text


def test_dirty_or_source_changes_never_pass(fixture, tmp_path, monkeypatch):
    manifest = write_manifest(fixture, tmp_path)
    dirty = dict(clean_provenance(), dirty=True)
    monkeypatch.setattr(check, "source_provenance", lambda: dirty)
    assert check.validate_manifest(manifest)["status"] == "failed"
    sequence = iter([clean_provenance(), dirty])
    monkeypatch.setattr(check, "source_provenance", lambda: next(sequence))
    report = check.validate_manifest(manifest)
    assert report["status"] == "failed"
    assert report["diagnostics"][0]["code"] == "source_changed_during_validation"


def test_native_claim_requires_container_and_failure_sanitizes(fixture, tmp_path, monkeypatch):
    monkeypatch.setattr(check, "source_provenance", clean_provenance)
    manifest = write_manifest(fixture, tmp_path, kind="native_vendor_export")
    report = check.validate_manifest(manifest)
    assert report["status"] == "failed" and report["diagnostics"][0]["code"] == "native_container_required"
    def bad_decode(path): raise ValueError("SECRET_FILE_AND_MEASUREMENTS")
    monkeypatch.setattr(check, "decode_file", bad_decode)
    report = check.validate_manifest(manifest)
    assert "SECRET" not in json.dumps(report) and report["status"] == "failed"


def test_mismatches_cannot_produce_pass(fixture, tmp_path, monkeypatch):
    monkeypatch.setattr(check, "source_provenance", clean_provenance)
    manifest = write_manifest(fixture, tmp_path)
    table = changed_column(fixture[2]["demand"], "vendor_current_rms_c", [0.0, 0.0])
    pq.write_table(table, fixture[3]["demand"]["parquet"])
    report = check.validate_manifest(manifest)
    assert report["status"] == "failed" and report["summary"]["mismatched_cells"] == 2


def test_wrong_import_and_untracked_validator_are_rejected(monkeypatch):
    from types import SimpleNamespace

    # This test models a clean checkout explicitly. CI imports a non-editable
    # installed wheel before this module; its real module paths must not serve as
    # the mocked clean-checkout case (nor be changed for other tests).
    modules = {
        name: SimpleNamespace(__file__=str(check.ROOT / relative))
        for name, relative in (
            ("fel_decoder", "src/fel_decoder/__init__.py"),
            ("fel_decoder._version", "src/fel_decoder/_version.py"),
            ("fel_decoder.core", "src/fel_decoder/core.py"),
            ("fel_decoder.layouts", "src/fel_decoder/layouts.py"),
        )
    }
    monkeypatch.setattr(check, "importlib", SimpleNamespace(import_module=modules.__getitem__))
    validator_untracked = False

    def git(*args):
        if args[0] == "rev-parse" and args[1] == "--show-toplevel": return str(check.ROOT).encode()
        if args[0] == "rev-parse": return b"b" * 40
        if args[0] == "status": return b""
        if args[0] == "show":
            relative = args[1].split(":", 1)[1]
            if validator_untracked and relative == "tools/validate_energy_exports.py":
                raise check.ValidationFailure("git_provenance_unavailable")
            return (check.ROOT / relative).read_bytes()
        raise AssertionError(args)

    monkeypatch.setattr(check, "_git", git)
    assert check.source_provenance()["dirty"] is False
    core = modules["fel_decoder.core"]
    original_path = core.__file__
    core.__file__ = str(check.ROOT.parent / "stale-installed" / "core.py")
    with pytest.raises(check.ValidationFailure, match="decoder_imported_outside_checkout"):
        check.source_provenance()
    core.__file__ = original_path
    validator_untracked = True
    with pytest.raises(check.ValidationFailure, match="validator_or_decoder_not_in_commit"):
        check.source_provenance()


def test_ole_embedded_source_is_byte_checked(fixture, tmp_path, monkeypatch):
    import io
    import olefile
    fel, decoded, tables, mappings = fixture
    streams = {("session", "raw"): fel.read_bytes()}
    entry = {"fel": str(fel), "native_container": {"path": "import.fca2", "embedded_fel_stream": ["session", "raw"]}}
    for name, table in tables.items():
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        streams[("session", name)] = sink.getvalue().to_pybytes()
        entry[name] = dict(mappings[name])
        entry[name].pop("parquet")
        entry[name]["parquet_stream"] = ["session", name]
    class FakeOle:
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def openstream(self, path): return io.BytesIO(streams[tuple(path)])
    monkeypatch.setattr(olefile, "OleFileIO", FakeOle)
    actual, identical = check._read_tables(entry, tmp_path / "manifest.json", "synthetic", decoded)
    assert identical
    assert check.compare_group("energy_trends", decoded.groups["energy_trends"],
                               actual["energy_trends"], entry["energy_trends"])["mismatched_cells"] == 0
    streams[("session", "raw")] += b"wrong"
    with pytest.raises(check.ValidationFailure, match="embedded_fel_does_not_match"):
        check._read_tables(entry, tmp_path / "manifest.json", "synthetic", decoded)
