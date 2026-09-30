"""Synthetic wire fixtures; no client recordings, names, hashes or measurements."""
import json
import struct

import numpy as np
import pytest

from fel_decoder import decode_bytes, FelError
from fel_decoder.cli import main
from fel_decoder.export import export_decoded
from fel_decoder.layouts import resolve_layout
from tests.test_decoder import fel, record, trend, START


def energy(tag=70, start=START):
    size, count = (744, 180) if tag == 70 else (104, 20)
    out = record(tag, size, start=start, end=start + 6_000_000_001)
    struct.pack_into('<I', out, 20, 600)
    struct.pack_into('<' + 'f' * count, out, 24, *(i + 0.25 for i in range(count)))
    return out


def test_named_measurements_and_wire_offsets():
    data = energy()
    # Independently specified byte offsets, not generated from the registry.
    for offset, value in [(100, 101.5), (112, 202.5), (124, 303.5),
                          (256, -45000.5), (304, 55000.25)]:
        struct.pack_into('<f', data, offset, value)
    decoded = decode_bytes(fel(data))
    g = decoded.groups['energy_trends']
    assert len(g) == 184  # 180 floats, period, three provenance arrays.
    assert g['trend_period'].dtype == np.dtype('uint32')
    assert g['start_ticks'][0] == START
    assert int(g['end_ticks'][0]) == START + 6_000_000_001
    assert g['current_rms_a_max'][0] == 101.5
    assert g['current_rms_b_max'][0] == 202.5
    assert decoded.measurement('current_rms_c_max')[0] == 303.5
    assert decoded.measurement('power_rms_active_total_max')[0] == -45000.5
    assert decoded.measurement('power_rms_apparent_total_max')[0] == 55000.25
    assert g['current_fund_effective'][0] == 179.25
    assert g['current_rms_c_max'].dtype == np.dtype('float32')
    m = decoded.manifest()
    assert m['named_fields']['energy_trends']['current_rms_c_max']['unit'] == 'A'
    assert m['named_fields']['energy_trends']['power_rms_active_total_max']['unit'] == 'W'
    assert m['named_fields']['energy_trends']['aux1_avg']['unit'] is None
    assert m['schema_version'] == 2


def test_demand_and_mixed_layouts_keep_boundaries():
    decoded = decode_bytes(fel(energy(), trend(), energy(71), energy(start=START + 90_000_000_000)))
    assert decoded.groups['trends']['values'].shape == (1, 6, 3)
    g = decoded.groups['energy_trends']
    np.testing.assert_array_equal(g['record_offset'], [32, 32 + 744 + 1724 + 104])
    np.testing.assert_array_equal(g['start_ticks'], [START, START + 90_000_000_000])
    d = decoded.groups['demand']
    assert d['current_rms_a'][0] == 3.25
    assert d['current_rms_c'][0] == 5.25
    assert d['energy_supplied_total'][0] == 19.25
    assert d['demand_period'][0] == 600


def test_no_electrical_meaning_is_inferred_for_old_slots():
    decoded = decode_bytes(fel(trend()))
    with pytest.raises(KeyError, match='unavailable'):
        decoded.measurement('current_rms_c_max')
    with pytest.raises(KeyError):
        decode_bytes(fel(energy())).measurement('start_ticks')
    assert resolve_layout(70, 744, profile='unverified') is None
    assert resolve_layout(60001, 744) is None
    assert resolve_layout(70, 1724) is None


@pytest.mark.parametrize('tag,size', [(70, 744), (71, 104)])
@pytest.mark.parametrize('adjustment', [-1, 1])
def test_unsupported_variants_are_inspectable_but_fail_decoding(tmp_path, capsys, tag, size, adjustment):
    source = tmp_path / 'variant.fel'
    source.write_bytes(fel(record(tag, size + adjustment)))
    assert main(['inspect', str(source), '--json']) == 0
    inv = json.loads(capsys.readouterr().out)
    assert inv['uninterpreted_record_counts'] == {str(tag): 1}
    assert inv['record_layouts'][0]['status'] == 'uninterpreted'
    with pytest.raises(FelError, match='Unsupported length'):
        decode_bytes(source.read_bytes())


@pytest.mark.parametrize('tag', [70, 71])
def test_new_layouts_reject_reversed_times_and_truncation(tag):
    data = energy(tag)
    struct.pack_into('<II', data, 12, (START - 1) >> 32, (START - 1) & 0xffffffff)
    with pytest.raises(FelError, match='End time'):
        decode_bytes(fel(data))
    with pytest.raises(FelError):
        decode_bytes(fel(energy(tag)[:-1]))


@pytest.mark.parametrize('compress', [False, True])
def test_named_csv_npz_roundtrip_missing_and_integer_precision(tmp_path, compress):
    data = energy()
    struct.pack_into('<f', data, 124, float('nan'))
    decoded = decode_bytes(fel(data, energy(71)))
    out = export_decoded(decoded, tmp_path / 'export', source_name='synthetic.fel',
                         csv_samples=True, compress=compress)
    import csv
    for name in ('energy_trends', 'demand'):
        with np.load(out / f'{name}.npz', allow_pickle=False) as arrays:
            for field, expected in decoded.groups[name].items():
                np.testing.assert_array_equal(arrays[field], expected)
        with (out / f'{name}.csv').open() as stream:
            row = next(csv.DictReader(stream))
        assert row['start_ticks'] == str(START)
        assert row['end_ticks'] == str(START + 6_000_000_001)
    assert np.isnan(decoded.measurement('current_rms_c_max')[0])
    assert 'nan' in (out / 'energy_trends.csv').read_text()
    manifest = json.loads((out / 'manifest.json').read_text())
    assert 'energy_trends.csv' in manifest['export']['files']
    assert 'demand.npz' in manifest['export']['files']


def test_unknown_same_sized_payload_is_never_decoded():
    data = energy()
    struct.pack_into('<H', data, 0, 60001)
    decoded = decode_bytes(fel(data, trend()))
    assert 'energy_trends' not in decoded.groups
    assert decoded.manifest()['uninterpreted_record_counts'] == {'60001': 1}


def test_cli_decodes_energy_only_recording(tmp_path, capsys):
    source = tmp_path / 'energy.fel'
    source.write_bytes(fel(energy(), energy(71)))
    out = tmp_path / 'out'
    assert main(['decode', str(source), '--output', str(out), '--csv']) == 0
    assert (out / 'energy_trends.csv').exists()
    assert source.read_bytes() == fel(energy(), energy(71))
