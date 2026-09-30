"""Independent, synthetic checks of the supported FEL binary layouts.

These fixtures are written with struct rather than a production encoder, so a
shared encoder/decoder implementation cannot conceal a layout or endian error.
"""

from pathlib import Path
import hashlib
import random
import struct
import tempfile
import unittest

import numpy as np

from fel_decoder import FelError, decode_bytes, decode_file, scan_bytes


HEADER = bytes.fromhex("0bba1400") + bytes(20) + bytes.fromhex("0000090000000000")
START = 133_000_000_123_456_789


def put_ticks(data, offset, ticks):
    """FEL timestamps store high word before low word, each little endian."""
    struct.pack_into("<II", data, offset, ticks >> 32, ticks & 0xFFFFFFFF)


def record(tag, size, *, start=START, end=None):
    out = bytearray(size)
    struct.pack_into("<HH", out, 0, tag, size)
    if size >= 20:
        put_ticks(out, 4, start)
        put_ticks(out, 12, start + 1_000 if end is None else end)
    return out


def event(*, event_id=17, event_type=6, channel=2, value=3.25,
          start=START, end=None):
    out = record(111, 104, start=start, end=end)
    struct.pack_into("<IHHf", out, 20, event_id, event_type, channel, value)
    return out


def wave(raw=None, offsets=None, coefficients=None, *, start=START, end=None):
    if raw is None:
        raw = np.array([[-32768, -1, 0, 1, 32767, 20, -30, 40]], dtype=np.int16)
    raw = np.asarray(raw, dtype=np.int16)
    if offsets is None:
        offsets = np.arange(len(raw), dtype=np.uint32) * 37
    if coefficients is None:
        coefficients = [(1.0, 0.0)] * 8
    out = record(105, 5208, start=start, end=end)
    for channel, (scale, bias) in enumerate(coefficients):
        struct.pack_into("<ff", out, 20 + channel * 8, scale, bias)
    struct.pack_into("<I", out, 84, len(raw))
    for row, (offset, samples) in enumerate(zip(offsets, raw)):
        struct.pack_into("<I8h", out, 88 + row * 20, int(offset), *samples)
    return out


def transient(raw=None, coefficients=None, *, sample_rate=1_000_000,
              start=START, end=None):
    if raw is None:
        raw = [[-32768, -1, 0, 32767], [1, -2, 3, -4]]
    raw = np.asarray(raw, dtype=np.int16)
    if coefficients is None:
        coefficients = [(1.0, 0.0)] * 4
    out = record(112, 32064, start=start, end=end)
    for channel, (scale, bias) in enumerate(coefficients):
        struct.pack_into("<ff", out, 20 + channel * 8, scale, bias)
    struct.pack_into("<II", out, 56, sample_rate, len(raw))
    out[64:64 + raw.size * 2] = raw.astype("<i2").tobytes()
    return out


def trend(values=None, *, start=START, end=None):
    if values is None:
        values = [[channel + 0.25, channel + 9.75, channel + 4.5]
                  for channel in range(6)]
    out = record(119, 1724, start=start, end=end)
    out[24:96] = np.asarray(values, dtype="<f4").reshape(6, 3).tobytes()
    return out


def fel(*records):
    body = b"".join(records)
    header = bytearray(HEADER)
    struct.pack_into("<I", header, 4, len(body))
    return bytes(header) + body


class InventoryTests(unittest.TestCase):
    def test_exact_record_boundaries_and_unknown_inventory(self):
        unknown = record(60001, 7)
        data = fel(unknown, event(), wave())
        records = scan_bytes(data)
        self.assertIsInstance(records, tuple)
        self.assertEqual([(r.offset, r.tag, r.size) for r in records],
                         [(32, 60001, 7), (39, 111, 104), (143, 105, 5208)])
        decoded = decode_bytes(data)
        self.assertEqual(decoded.records, records)
        np.testing.assert_array_equal(decoded.groups["events"]["record_offset"], [39])
        np.testing.assert_array_equal(decoded.groups["waveforms"]["record_offset"], [143])

    def test_unknown_record_does_not_trigger_payload_scanning(self):
        unknown = record(60001, 120)
        # A structurally plausible event inside an unknown payload must stay opaque.
        unknown[8:112] = event(event_id=999)
        decoded = decode_bytes(fel(unknown, event(event_id=123)))
        self.assertEqual(len(decoded.records), 2)
        np.testing.assert_array_equal(decoded.groups["events"]["event_id"], [123])

    def test_unknown_only_is_inventoryable_but_not_decodable(self):
        data = fel(record(60001, 4))
        self.assertEqual(len(scan_bytes(data)), 1)
        with self.assertRaises(FelError):
            decode_bytes(data)

    def test_header_reserved_bytes_are_not_interpreted(self):
        data = bytearray(fel(event()))
        data[8:24] = bytes(range(16))
        decoded = decode_bytes(data)
        self.assertEqual(len(decoded.groups["events"]["value"]), 1)

    def test_payload_record_order_is_preserved(self):
        data = fel(event(start=START + 10_000), event(start=START))
        decoded = decode_bytes(data)
        np.testing.assert_array_equal(decoded.groups["events"]["start_ticks"],
                                      [START + 10_000, START])


class EventTests(unittest.TestCase):
    def test_event_depth_severity_and_auxiliary_capture_times(self):
        data = event()
        struct.pack_into("<fI", data, 32, -7.125, 0xFEDCBA98)
        names = ("wave_start_ticks", "wave_end_ticks", "rms_start_ticks", "rms_end_ticks",
                 "msv_start_ticks", "msv_end_ticks", "transient_start_ticks", "transient_end_ticks")
        times = [0, 0, START - 29, START + 31, START - 11, START + 43,
                 0xEEDDCCBBAA998800, 0xEEDDCCBBAA9988FF]
        for index, ticks in enumerate(times):
            put_ticks(data, 40 + 8 * index, ticks)
        group = decode_bytes(fel(data)).groups["events"]
        self.assertEqual(float(group["depth"][0]), -7.125)
        self.assertEqual(int(group["severity"][0]), 0xFEDCBA98)
        for name, ticks in zip(names, times):
            with self.subTest(field=name):
                self.assertEqual(int(group[name][0]), ticks)

    def test_unsigned_ticks_and_event_fields_remain_exact(self):
        start = 0xFEDCBA9876543210
        decoded = decode_bytes(fel(event(event_id=0xFEDCBA98, event_type=0xFFFD,
                                        channel=0xFFFE, value=-123.25,
                                        start=start, end=start + 31)))
        events = decoded.groups["events"]
        self.assertEqual(int(events["start_ticks"][0]), start)
        self.assertEqual(int(events["end_ticks"][0]), start + 31)
        self.assertEqual(int(events["event_id"][0]), 0xFEDCBA98)
        self.assertEqual(int(events["event_type"][0]), 0xFFFD)
        self.assertEqual(int(events["channel_code"][0]), 0xFFFE)
        self.assertEqual(float(events["value"][0]), -123.25)

    def test_duplicate_event_ids_are_not_deduplicated(self):
        events = decode_bytes(fel(event(value=1), event(value=2))).groups["events"]
        np.testing.assert_array_equal(events["event_id"], [17, 17])
        np.testing.assert_array_equal(events["value"], [1, 2])

    def test_sub_microsecond_tick_distinction_is_retained(self):
        events = decode_bytes(fel(event(start=START), event(start=START + 1))).groups["events"]
        self.assertEqual(int(events["start_ticks"][1]) - int(events["start_ticks"][0]), 1)

    def test_zero_duration_event_is_valid(self):
        events = decode_bytes(fel(event(end=START))).groups["events"]
        self.assertEqual(int(events["start_ticks"][0]), int(events["end_ticks"][0]))


class WaveformTests(unittest.TestCase):
    def test_signed_samples_scaling_bias_and_nan_channel(self):
        raw = np.array([[-32768, -1, 0, 1, 32767, -21, 22, 23],
                        [32767, 2, -3, 4, -32768, 25, -26, -27]], dtype=np.int16)
        coefficients = [(0.25, 10.5), (-2.0, -5.25), (3.0, 7.0),
                        (0.0, -2.0), (1.25, 0.0), (0.5, 100.0),
                        (float("nan"), 0.0), (1.0, float("nan"))]
        group = decode_bytes(fel(wave(raw, [17, 83], coefficients))).groups["waveforms"]
        np.testing.assert_array_equal(group["raw"], raw)
        self.assertEqual(group["raw"].dtype, np.dtype("int16"))
        self.assertEqual(group["values"].dtype, np.dtype("float64"))
        # Independent expected arithmetic, including negative scale and constant channel.
        expected = np.array([[-8181.5, -3.25, 7.0, -2.0, 40958.75, 89.5, np.nan, np.nan],
                             [8202.25, -9.25, -2.0, -2.0, -40960.0, 112.5, np.nan, np.nan]])
        np.testing.assert_array_equal(group["values"], expected)
        np.testing.assert_array_equal(group["offset_us"], [17, 83])
        np.testing.assert_array_equal(group["coefficients"][0], coefficients)
        np.testing.assert_array_equal(group["sample_start"], [0])
        np.testing.assert_array_equal(group["sample_count"], [2])

    def test_record_specific_scaling_and_large_time_gaps(self):
        raw = [[2] * 8, [-3] * 8]
        data = fel(wave(raw, [0, 37]),
                   wave([[4] * 8], [91], [(2, 10)] * 8, start=START + 9_000_000_000))
        group = decode_bytes(data).groups["waveforms"]
        np.testing.assert_array_equal(group["sample_start"], [0, 2])
        np.testing.assert_array_equal(group["sample_count"], [2, 1])
        np.testing.assert_array_equal(group["offset_us"], [0, 37, 91])
        np.testing.assert_array_equal(group["values"], [[2] * 8, [-3] * 8, [18] * 8])
        np.testing.assert_array_equal(group["start_ticks"], [START, START + 9_000_000_000])
        self.assertEqual(group["coefficients"].shape, (2, 8, 2))

    def test_partial_record_padding_is_never_decoded(self):
        data = wave([[7] * 8], [3])
        data[108:] = b"\xff" * (len(data) - 108)
        group = decode_bytes(fel(data)).groups["waveforms"]
        self.assertEqual(group["raw"].shape, (1, 8))
        np.testing.assert_array_equal(group["values"], [[7] * 8])

    def test_full_capacity_waveform(self):
        raw = np.arange(256 * 8, dtype=np.int16).reshape(256, 8) - 1000
        group = decode_bytes(fel(wave(raw))).groups["waveforms"]
        self.assertEqual(group["raw"].shape, (256, 8))
        np.testing.assert_array_equal(group["raw"], raw)

    def test_unsigned_sample_offsets(self):
        group = decode_bytes(fel(wave([[1] * 8], [0xFEDCBA98]))).groups["waveforms"]
        self.assertEqual(int(group["offset_us"][0]), 0xFEDCBA98)

    def test_equal_offsets_are_not_mistaken_for_decrease(self):
        group = decode_bytes(fel(wave([[1] * 8, [2] * 8], [7, 7]))).groups["waveforms"]
        np.testing.assert_array_equal(group["offset_us"], [7, 7])


class TransientTests(unittest.TestCase):
    def test_sample_rate_raw_values_and_scaling(self):
        coefficients = [(0.5, 1.25), (-2.0, 4.0), (3.0, -5.0), (0.0, 2.5)]
        group = decode_bytes(fel(transient(coefficients=coefficients,
                                          sample_rate=2_000_000))).groups["transients"]
        np.testing.assert_array_equal(group["raw"], [[-32768, -1, 0, 32767], [1, -2, 3, -4]])
        np.testing.assert_array_equal(group["values"], [[-16382.75, 6, -5, 2.5], [1.75, 8, 4, 2.5]])
        np.testing.assert_array_equal(group["sample_rate_hz"], [2_000_000])
        self.assertEqual(group["values"].dtype, np.dtype("float64"))
        self.assertEqual(group["coefficients"].shape, (1, 4, 2))
        self.assertNotIn("offset_us", group)

    def test_records_keep_their_own_rate_and_coefficient(self):
        data = fel(transient([[1] * 4], sample_rate=100),
                   transient([[3] * 4, [-4] * 4], [(2, -1)] * 4,
                             sample_rate=200, start=START + 123456789))
        group = decode_bytes(data).groups["transients"]
        np.testing.assert_array_equal(group["sample_start"], [0, 1])
        np.testing.assert_array_equal(group["sample_count"], [1, 2])
        np.testing.assert_array_equal(group["sample_rate_hz"], [100, 200])
        np.testing.assert_array_equal(group["values"], [[1] * 4, [5] * 4, [-9] * 4])

    def test_partial_record_ignores_padding(self):
        data = transient([[12] * 4])
        data[72:] = b"\xff" * (len(data) - 72)
        group = decode_bytes(fel(data)).groups["transients"]
        self.assertEqual(group["raw"].shape, (1, 4))
        np.testing.assert_array_equal(group["values"], [[12] * 4])

    def test_full_capacity_transient(self):
        raw = (np.arange(4000 * 4, dtype=np.int16) - 8000).reshape(4000, 4)
        group = decode_bytes(fel(transient(raw))).groups["transients"]
        np.testing.assert_array_equal(group["raw"], raw)
        self.assertEqual(group["raw"].shape, (4000, 4))

    def test_nan_coefficients_preserve_unused_channels(self):
        coefficients = [(np.nan, np.nan), (1, 0), (2, 1), (0, np.nan)]
        group = decode_bytes(fel(transient([[1, 2, 3, 4]], coefficients))).groups["transients"]
        np.testing.assert_array_equal(group["values"], [[np.nan, 2, 7, np.nan]])


class TrendAndFileTests(unittest.TestCase):
    def test_six_generic_channel_triples_remain_min_max_average(self):
        values = [[1 + i, 10 + i, 3 + i] for i in range(6)]
        data = trend(values)
        # A seventh triple exists in observed files but is deliberately unsupported.
        struct.pack_into("<3f", data, 96, 999, 998, 997)
        group = decode_bytes(fel(data)).groups["trends"]
        self.assertEqual(group["values"].shape, (1, 6, 3))
        np.testing.assert_array_equal(group["values"], [values])

    def test_trend_nan_is_preserved(self):
        values = np.arange(18, dtype=float).reshape(6, 3)
        values[1, 2] = np.nan
        group = decode_bytes(fel(trend(values))).groups["trends"]
        np.testing.assert_array_equal(group["values"][0], values)

    def test_absent_groups_have_consistent_empty_shapes(self):
        groups = decode_bytes(fel(event())).groups
        self.assertEqual(set(groups), {"events", "waveforms", "transients", "trends"})
        for name, channels in (("waveforms", 8), ("transients", 4)):
            with self.subTest(group=name):
                self.assertEqual(groups[name]["raw"].shape, (0, channels))
                self.assertEqual(groups[name]["values"].shape, (0, channels))
                self.assertEqual(groups[name]["coefficients"].shape, (0, channels, 2))
                self.assertEqual(groups[name]["sample_count"].shape, (0,))
        self.assertEqual(groups["trends"]["values"].shape, (0, 6, 3))
        groups = decode_bytes(fel(wave())).groups
        self.assertEqual(groups["events"]["value"].shape, (0,))

    def test_file_and_byte_apis_agree_for_all_groups(self):
        data = fel(event(), wave(), transient(), trend(), record(60001, 4))
        expected = decode_bytes(data)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.fel"
            path.write_bytes(data)
            actual = decode_file(path)
            self.assertEqual(actual.records, expected.records)
            for group_name, expected_group in expected.groups.items():
                self.assertEqual(set(actual.groups[group_name]), set(expected_group))
                for array_name, expected_array in expected_group.items():
                    with self.subTest(group=group_name, array=array_name):
                        np.testing.assert_array_equal(actual.groups[group_name][array_name], expected_array)

    def test_source_provenance_uses_exact_input_bytes(self):
        data = fel(event(), record(60001, 4))
        decoded = decode_bytes(data)
        self.assertEqual(decoded.source_bytes, len(data))
        self.assertEqual(decoded.source_sha256, hashlib.sha256(data).hexdigest())
        self.assertEqual(decoded.header_hex, data[:32].hex())

    def test_deterministic_random_sample_records_against_scalar_arithmetic(self):
        rng = random.Random(417)
        for case in range(20):
            factory, channels, group_name = ((wave, 8, "waveforms") if case % 2
                                              else (transient, 4, "transients"))
            count = rng.randrange(1, 30)
            raw = [[rng.randrange(-32768, 32768) for _ in range(channels)]
                   for _ in range(count)]
            coefficients = [(rng.randrange(-16, 17) / 8, rng.randrange(-100, 101) / 4)
                            for _ in range(channels)]
            expected = [[sample * coefficients[c][0] + coefficients[c][1]
                         for c, sample in enumerate(row)] for row in raw]
            group = decode_bytes(fel(factory(raw=raw, coefficients=coefficients))).groups[group_name]
            with self.subTest(case=case, group=group_name):
                np.testing.assert_array_equal(group["raw"], raw)
                np.testing.assert_array_equal(group["values"], expected)


class MalformedInputTests(unittest.TestCase):
    def test_short_and_wrong_headers(self):
        cases = [b"", HEADER[:4], HEADER[:31], b"WRNG" + HEADER[4:],
                 HEADER[:24] + bytes(8)]
        for data in cases:
            with self.subTest(length=len(data), prefix=data[:4]):
                for function in (scan_bytes, decode_bytes):
                    with self.assertRaises(FelError):
                        function(data)

    def test_header_without_records_cannot_be_decoded(self):
        with self.assertRaises(FelError):
            decode_bytes(HEADER)

    def test_trailing_partial_record_header_is_rejected(self):
        for tail in (b"\x01", b"\x01\x00", b"\x01\x00\x04"):
            with self.subTest(tail=tail):
                for function in (scan_bytes, decode_bytes):
                    with self.assertRaises(FelError):
                        function(fel(event(), tail))

    def test_record_lengths_below_header_size(self):
        for size in range(4):
            for function in (scan_bytes, decode_bytes):
                with self.subTest(size=size, function=function.__name__):
                    with self.assertRaises(FelError):
                        function(fel(struct.pack("<HH", 60001, size)))

    def test_claimed_size_past_eof(self):
        for function in (scan_bytes, decode_bytes):
            with self.subTest(function=function.__name__):
                with self.assertRaises(FelError):
                    function(fel(struct.pack("<HH", 60001, 65000) + bytes(100)))

    def test_file_payload_length_mismatch_is_rejected(self):
        for adjustment in (-1, 1, 2**24):
            data = bytearray(fel(event()))
            struct.pack_into("<I", data, 4, len(data) - 32 + adjustment)
            for function in (scan_bytes, decode_bytes):
                with self.subTest(adjustment=adjustment, function=function.__name__):
                    with self.assertRaisesRegex(FelError, "[Hh]eader|payload"):
                        function(data)

    def test_truncated_supported_records(self):
        for payload in (event(), wave(), transient(), trend()):
            for removed in (1, 4, len(payload) - 4):
                with self.subTest(tag=struct.unpack_from("<H", payload)[0], removed=removed):
                    with self.assertRaises(FelError):
                        decode_bytes(fel(payload[:-removed]))

    def test_known_tags_with_different_complete_lengths(self):
        for tag, size in ((105, 5208), (111, 104), (112, 32064), (119, 1724)):
            for wrong_size in (4, size - 1, size + 1):
                with self.subTest(tag=tag, size=wrong_size):
                    with self.assertRaises(FelError):
                        decode_bytes(fel(record(tag, wrong_size)))

    def test_counts_zero_and_beyond_capacity_are_rejected(self):
        for factory, field, capacity in ((wave, 84, 256), (transient, 60, 4000)):
            for count in (0, capacity + 1, 0xFFFFFFFF):
                payload = factory()
                struct.pack_into("<I", payload, field, count)
                with self.subTest(factory=factory.__name__, count=count):
                    with self.assertRaises(FelError):
                        decode_bytes(fel(payload))

    def test_zero_transient_sample_rate_is_rejected(self):
        with self.assertRaises(FelError):
            decode_bytes(fel(transient(sample_rate=0)))

    def test_reversed_times_are_rejected_in_every_supported_group(self):
        for factory in (event, wave, transient, trend):
            with self.subTest(factory=factory.__name__):
                with self.assertRaises(FelError):
                    decode_bytes(fel(factory(start=START, end=START - 1)))

    def test_decreasing_wave_offsets_are_rejected(self):
        with self.assertRaises(FelError):
            decode_bytes(fel(wave([[1] * 8, [2] * 8], [100, 99])))

    def test_infinite_scale_and_bias_are_rejected(self):
        for factory, channels in ((wave, 8), (transient, 4)):
            for bad in (float("inf"), float("-inf")):
                for position in (0, 1):
                    coefficients = [[1.0, 0.0] for _ in range(channels)]
                    coefficients[channels - 1][position] = bad
                    with self.subTest(factory=factory.__name__, bad=bad, position=position):
                        with self.assertRaises(FelError):
                            decode_bytes(fel(factory(coefficients=coefficients)))

    def test_error_does_not_return_partial_decoding(self):
        bad = wave()
        struct.pack_into("<I", bad, 84, 257)
        with self.assertRaises(FelError):
            decode_bytes(fel(event(), bad))


if __name__ == "__main__":
    unittest.main()
