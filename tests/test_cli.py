"""Synthetic end-to-end exports; no recordings or instrument identifiers."""

import csv
import json
import struct
import zipfile

import numpy as np
import pytest

from fel_decoder.cli import main
from fel_decoder.core import decode_bytes
from fel_decoder.export import export_decoded


START = 133000000000000013  # Deliberately not exactly representable as float64.


def _record(tag, size, start=START):
    data = bytearray(size)
    struct.pack_into("<HH", data, 0, tag, size)
    for offset, ticks in ((4, start), (12, start + 1000)):
        struct.pack_into("<II", data, offset, ticks >> 32, ticks & 0xFFFFFFFF)
    return data


def _sample_record(tag, start=START):
    transient = tag == 112
    channels, base, size = (4, 64, 32064) if transient else (8, 88, 5208)
    data = _record(tag, size, start)
    for channel in range(channels):
        struct.pack_into("<ff", data, 20 + 8 * channel, 0.5, -1.0)
    struct.pack_into("<I", data, base - 4, 2)
    if transient:
        struct.pack_into("<I", data, 56, 1000000)
    for sample in range(2):
        offset = base + sample * (channels * 2 + (0 if transient else 4))
        if not transient:
            struct.pack_into("<I", data, offset, sample * 17)
            offset += 4
        struct.pack_into("<" + "h" * channels, data, offset, *(sample * 10 + c for c in range(channels)))
    return data


def _wrap_records(records):
    header = bytearray(32)
    header[:4] = bytes.fromhex("0bba1400")
    header[24:32] = bytes.fromhex("0000090000000000")
    struct.pack_into("<I", header, 4, len(records))
    return bytes(header + records)


def _fixture_bytes():
    unknown = struct.pack("<HHI", 999, 8, 42)
    event = _record(111, 104)
    struct.pack_into("<IHHf", event, 20, 4294967295, 7, 3, 1.25)
    struct.pack_into("<fI", event, 32, 2.5, 4294967294)
    for index in range(8):
        ticks = START + index
        struct.pack_into("<II", event, 40 + index * 8, ticks >> 32, ticks & 0xFFFFFFFF)
    trend = _record(119, 1724)
    struct.pack_into("<18f", trend, 24, *range(18))
    return _wrap_records(unknown + event + _sample_record(105) + _sample_record(112) + trend + _sample_record(105, START + 5000))


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "synthetic.fel"
    path.write_bytes(_fixture_bytes())
    return path


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_help_and_version(capsys):
    assert main([]) == 0
    assert "fel-helper" in capsys.readouterr().out
    with pytest.raises(SystemExit) as result:
        main(["--version"])
    assert result.value.code == 0
    from fel_decoder import __version__
    assert f"FEL-helper {__version__}" in capsys.readouterr().out


def test_inspect_counts_only(recording, capsys):
    assert main(["inspect", str(recording), "--json"]) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert inventory["record_count"] == 6
    assert inventory["record_counts"] == {"105": 2, "111": 1, "112": 1, "119": 1, "999": 1}
    assert inventory["uninterpreted_record_counts"] == {"999": 1}
    assert inventory["source_filename"] == "synthetic.fel"
    assert "values" not in inventory


def test_inspect_human_readable(recording, capsys):
    assert main(["inspect", str(recording)]) == 0
    output = capsys.readouterr().out
    assert "tag 999: 1 (uninterpreted)" in output
    assert "framing only" in output


@pytest.mark.parametrize("compressed", [False, True])
def test_decode_arrays_and_default_csv(recording, tmp_path, compressed):
    destination = tmp_path / "decoded"
    args = ["decode", str(recording), "--output", str(destination)]
    if compressed:
        args.append("--compress")
    assert main(args) == 0
    assert {p.name for p in destination.iterdir()} == {
        "manifest.json", "records.csv", "events.csv",
        "events.npz", "waveforms.npz", "transients.npz", "trends.npz",
    }
    decoded = decode_bytes(recording.read_bytes())
    for name, arrays in decoded.groups.items():
        with np.load(destination / f"{name}.npz", allow_pickle=False) as exported:
            assert set(exported.files) == set(arrays)
            for key, expected in arrays.items():
                np.testing.assert_array_equal(exported[key], expected)
                assert exported[key].dtype == expected.dtype
        with zipfile.ZipFile(destination / f"{name}.npz") as archive:
            assert all(item.compress_type == (zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED) for item in archive.infolist())
    event = _read_csv(destination / "events.csv")[0]
    assert event["start_ticks"] == str(START)
    assert event["event_id"] == "4294967295"
    assert event["record_index"] == "1"
    assert event["value"] == "1.25"
    assert event["depth"] == "2.5"
    assert event["severity"] == "4294967294"
    for index, key in enumerate((
        "wave_start_ticks", "wave_end_ticks", "rms_start_ticks", "rms_end_ticks",
        "msv_start_ticks", "msv_end_ticks", "transient_start_ticks", "transient_end_ticks",
    )):
        assert event[key] == str(START + index)
    inventory = _read_csv(destination / "records.csv")
    assert inventory[0] == {"record_index": "0", "offset": "32", "tag": "999", "size": "8"}
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["source_filename"] == "synthetic.fel"
    assert manifest["export"]["npz_compressed"] is compressed
    assert str(tmp_path) not in json.dumps(manifest)


def test_optional_csv_keeps_capture_boundaries(recording, tmp_path):
    destination = tmp_path / "csv"
    assert main(["decode", str(recording), "--output", str(destination), "--csv"]) == 0
    waveforms = _read_csv(destination / "waveforms.csv")
    assert [row["record_index"] for row in waveforms] == ["2", "2", "5", "5"]
    assert [row["sample_index"] for row in waveforms] == ["0", "1", "0", "1"]
    assert [row["offset_us"] for row in waveforms] == ["0", "17", "0", "17"]
    assert [row["start_ticks"] for row in waveforms] == [str(START)] * 2 + [str(START + 5000)] * 2
    assert waveforms[1]["channel_8_raw"] == "17"
    assert float(waveforms[1]["channel_8_value"]) == 7.5
    transient = _read_csv(destination / "transients.csv")
    assert [row["sample_index"] for row in transient] == ["0", "1"]
    assert transient[0]["sample_rate_hz"] == "1000000"
    assert transient[0]["start_ticks"] == str(START)
    trends = _read_csv(destination / "trends.csv")
    assert trends[0]["channel_6_field_3"] == "17.0"


def test_csv_preserves_missing_values_and_source_bytes(recording, tmp_path):
    data = bytearray(recording.read_bytes())
    decoded = decode_bytes(data)
    waveform = next(record for record in decoded.records if record.tag == 105)
    struct.pack_into("<f", data, waveform.offset + 20, float("nan"))
    recording.write_bytes(data)
    destination = tmp_path / "missing_values"
    assert main(["decode", str(recording), "--output", str(destination), "--csv"]) == 0
    rows = _read_csv(destination / "waveforms.csv")
    assert rows[0]["channel_1_value"] == "nan"
    assert rows[0]["channel_1_raw"] == "0"
    with np.load(destination / "waveforms.npz", allow_pickle=False) as arrays:
        assert np.isnan(arrays["values"][0, 0])
    assert recording.read_bytes() == data


@pytest.mark.parametrize("existing_file", [False, True])
def test_refuses_existing_output(recording, tmp_path, capsys, existing_file):
    destination = tmp_path / "existing"
    if existing_file:
        destination.write_bytes(b"untouched")
    else:
        destination.mkdir()
    assert main(["decode", str(recording), "--output", str(destination)]) == 1
    assert "fel-helper:" in capsys.readouterr().err
    if existing_file:
        assert destination.read_bytes() == b"untouched"
    else:
        assert list(destination.iterdir()) == []


@pytest.mark.parametrize("bad_file", [False, True])
def test_missing_or_invalid_input_never_creates_output(tmp_path, capsys, bad_file):
    source = tmp_path / "bad.fel"
    if bad_file:
        source.write_bytes(b"not a FEL file")
    destination = tmp_path / "absent"
    assert main(["decode", str(source), "--output", str(destination)]) == 1
    assert "fel-helper:" in capsys.readouterr().err
    assert not destination.exists()


def test_invalid_known_record_is_not_exported(recording, tmp_path):
    recording.write_bytes(_wrap_records(struct.pack("<HH", 105, 4)))
    destination = tmp_path / "absent"
    assert main(["decode", str(recording), "--output", str(destination)]) == 1
    assert not destination.exists()


def test_empty_measurement_groups_export_without_pickle(tmp_path):
    # A valid event-only recording still exports correctly shaped empty groups.
    decoded = decode_bytes(_wrap_records(_record(111, 104)))
    destination = export_decoded(decoded, tmp_path / "empty", source_name="/private/path/empty.fel", csv_samples=True)
    for name in ("waveforms", "transients", "trends"):
        with np.load(destination / f"{name}.npz", allow_pickle=False) as arrays:
            assert arrays["record_offset"].size == 0
    assert _read_csv(destination / "waveforms.csv") == []
    assert json.loads((destination / "manifest.json").read_text())["source_filename"] == "empty.fel"


def test_profile_without_measurements_is_inspectable_not_decodable(tmp_path, capsys):
    source = tmp_path / "empty.fel"
    source.write_bytes(_wrap_records(b""))
    assert main(["inspect", str(source), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["record_count"] == 0
    assert main(["decode", str(source), "--output", str(tmp_path / "absent")]) == 1
    assert "No supported measurement records" in capsys.readouterr().err
    assert not (tmp_path / "absent").exists()


def test_failed_export_has_no_completion_marker(recording, tmp_path, monkeypatch):
    import fel_decoder.export as exports
    destination = tmp_path / "partial"
    actual_write_csv = exports._write_csv

    def interrupted(path, header, rows):
        if path.name == "events.csv":
            raise OSError("simulated storage failure")
        actual_write_csv(path, header, rows)

    monkeypatch.setattr(exports, "_write_csv", interrupted)
    assert main(["decode", str(recording), "--output", str(destination)]) == 1
    assert destination.exists()
    assert not (destination / "manifest.json").exists()


def test_failed_manifest_write_has_no_completion_marker(recording, tmp_path, monkeypatch):
    import fel_decoder.export as exports
    destination = tmp_path / "partial_manifest"

    def interrupted_dump(manifest, stream, **kwargs):
        stream.write("{")
        raise OSError("simulated storage failure during manifest")

    monkeypatch.setattr(exports.json, "dump", interrupted_dump)
    assert main(["decode", str(recording), "--output", str(destination)]) == 1
    assert not (destination / "manifest.json").exists()
    assert (destination / ".manifest.json.tmp").read_text() == "{"
