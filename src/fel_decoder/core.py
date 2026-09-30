"""Bounded, vectorized decoding of an empirically observed FEL record profile.

This module extracts numbers, not electrical meaning. Raw integer timestamps,
channel slots, and capture boundaries are retained. No clock alignment is applied.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct

import numpy as np

HEADER_SIZE = 32
MAGIC = bytes.fromhex("0bba1400")
PROFILE_BYTES = bytes.fromhex("0000090000000000")
KNOWN_SIZES = {105: 5208, 111: 104, 112: 32064, 119: 1724}
FILETIME_UNIX_EPOCH = 116444736000000000
WAVE_DTYPE = np.dtype([("offset_us", "<u4"), ("raw", "<i2", (8,))])


class FelError(ValueError):
    """The input is damaged or outside the supported FEL layout."""


@dataclass(frozen=True)
class Record:
    offset: int
    tag: int
    size: int


@dataclass
class DecodedFile:
    records: tuple[Record, ...]
    groups: dict[str, dict[str, np.ndarray]]
    source_bytes: int
    source_sha256: str
    header_hex: str

    def manifest(self) -> dict:
        counts = Counter(r.tag for r in self.records)
        return {
            "schema_version": 1,
            "decoder_version": "0.1.0",
            "profile": "observed-fel-0bba1400",
            "source_bytes": self.source_bytes,
            "source_sha256": self.source_sha256,
            "header_hex": self.header_hex,
            "record_count": len(self.records),
            "record_counts": {str(k): v for k, v in sorted(counts.items())},
            "uninterpreted_record_counts": {
                str(k): v for k, v in sorted(counts.items()) if k not in KNOWN_SIZES
            },
            "coverage": "Selected fields in tags 105, 111, 112 and 119 only; not a complete FEL conversion.",
            "time_basis": "Unadjusted instrument ticks: 100 ns since 1601-01-01, high word then low word. Clock accuracy and synchronization are not verified.",
            "channel_basis": "Numbered slots only. No wiring, voltage reference, engineering unit or phase-angle interpretation is inferred.",
            "missing_values": "NaN is preserved. It must not be treated as a zero measurement.",
            "groups": {
                name: {key: {"shape": list(a.shape), "dtype": str(a.dtype)} for key, a in arrays.items()}
                for name, arrays in self.groups.items()
            },
        }


def scan_bytes(data: bytes) -> tuple[Record, ...]:
    """Inventory every record; reject bad framing instead of returning partial data."""
    if len(data) < HEADER_SIZE:
        raise FelError("Truncated FEL header: expected at least 32 bytes")
    if data[:4] != MAGIC or data[24:32] != PROFILE_BYTES:
        raise FelError("Unsupported FEL header/profile; this decoder accepts only the observed layout")
    declared = struct.unpack_from("<I", data, 4)[0]
    if declared != len(data) - HEADER_SIZE:
        raise FelError(f"Header declares {declared} payload bytes, but file contains {len(data) - HEADER_SIZE}")
    records = []
    offset = HEADER_SIZE
    while offset < len(data):
        if len(data) - offset < 4:
            raise FelError(f"Truncated record header at byte {offset}")
        tag, size = struct.unpack_from("<HH", data, offset)
        if size < 4:
            raise FelError(f"Invalid record length {size} at byte {offset}")
        if size > len(data) - offset:
            raise FelError(f"Record at byte {offset} extends beyond end of file")
        records.append(Record(offset, tag, size))
        offset += size
    return tuple(records)


def _ticks(data: bytes, offset: int) -> int:
    high, low = struct.unpack_from("<II", data, offset)
    return (high << 32) | low


def _times(data: bytes, record: Record) -> tuple[int, int]:
    start, end = _ticks(data, record.offset + 4), _ticks(data, record.offset + 12)
    if end < start:
        raise FelError(f"End time precedes start time at byte {record.offset}")
    return start, end


def _array(values, dtype, shape=None):
    a = np.asarray(values, dtype=dtype)
    return a if shape is None else a.reshape(shape)


def _metadata(rows):
    return {
        "record_offset": _array([r[0] for r in rows], "<u8"),
        "start_ticks": _array([r[1] for r in rows], "<u8"),
        "end_ticks": _array([r[2] for r in rows], "<u8"),
    }


def _samples(data: bytes, records: list[Record], transient: bool) -> dict[str, np.ndarray]:
    channels = 4 if transient else 8
    base = 64 if transient else 88
    stride = 8 if transient else WAVE_DTYPE.itemsize
    rows, counts, coefficients, rates = [], [], [], []
    # Validate all sizes/counts before allocation. Counts cannot allocate beyond input.
    for r in records:
        start, end = _times(data, r)
        count = struct.unpack_from("<I", data, r.offset + base - 4)[0]
        if count == 0 or count > (r.size - base) // stride:
            raise FelError(f"Invalid sample count {count} at byte {r.offset}")
        co = np.frombuffer(data, dtype="<f4", count=channels * 2, offset=r.offset + 20).reshape(channels, 2)
        if np.isinf(co).any():
            raise FelError(f"Infinite scaling coefficient at byte {r.offset}")
        if transient:
            rate = struct.unpack_from("<I", data, r.offset + 56)[0]
            if rate == 0:
                raise FelError(f"Zero transient sample rate at byte {r.offset}")
            rates.append(rate)
        rows.append((r.offset, start, end))
        counts.append(count)
        coefficients.append(co)
    count_array = _array(counts, "<u4")
    starts = np.empty(len(records), dtype="<u8")
    if records:
        starts[0] = 0
        np.cumsum(count_array[:-1], dtype="<u8", out=starts[1:])
    total = sum(counts)
    raw = np.empty((total, channels), dtype="<i2")
    values = np.empty((total, channels), dtype="<f8")
    offsets = np.empty(total, dtype="<u4") if not transient else None
    for r, begin, count, co in zip(records, starts, counts, coefficients):
        begin = int(begin)
        selection = slice(begin, begin + count)
        if transient:
            samples = np.frombuffer(data, dtype="<i2", count=count * channels, offset=r.offset + base).reshape(count, channels)
        else:
            block = np.frombuffer(data, dtype=WAVE_DTYPE, count=count, offset=r.offset + base)
            dt = block["offset_us"]
            if np.any(dt[1:] < dt[:-1]):
                raise FelError(f"Waveform sample offsets go backwards at byte {r.offset}")
            offsets[selection] = dt
            samples = block["raw"]
        raw[selection] = samples
        np.multiply(samples, co[:, 0], out=values[selection], dtype=np.float64)
        values[selection] += co[:, 1]
    result = _metadata(rows)
    result.update({
        "sample_start": starts,
        "sample_count": count_array,
        "coefficients": _array(coefficients, "<f4", (-1, channels, 2)),
        "raw": raw,
        "values": values,
    })
    if transient:
        result["sample_rate_hz"] = _array(rates, "<u4")
    else:
        result["offset_us"] = offsets
    return result


def decode_bytes(data: bytes) -> DecodedFile:
    """Decode supported fields, preserving all samples and record boundaries.

    Unknown record tags are inventoried. Unsupported lengths of known tags fail
    explicitly, because silently interpreting a different layout is unsafe.
    """
    records = scan_bytes(data)
    selected = {tag: [] for tag in KNOWN_SIZES}
    for r in records:
        if r.tag in KNOWN_SIZES:
            if r.size != KNOWN_SIZES[r.tag]:
                raise FelError(f"Unsupported length {r.size} for tag {r.tag} at byte {r.offset}; expected {KNOWN_SIZES[r.tag]}")
            selected[r.tag].append(r)
    if not any(selected.values()):
        raise FelError("No supported measurement records found; use inspect to inventory this file")
    event_rows, event_fields = [], []
    for r in selected[111]:
        event_rows.append((r.offset, *_times(data, r)))
        event_fields.append(struct.unpack_from("<IHHf", data, r.offset + 20))
    events = _metadata(event_rows)
    for index, (name, dtype) in enumerate((
        ("event_id", "<u4"), ("event_type", "<u2"), ("channel_code", "<u2"), ("value", "<f4")
    )):
        events[name] = _array([r[index] for r in event_fields], dtype)
    events["depth"] = _array([struct.unpack_from("<f", data, r.offset + 32)[0] for r in selected[111]], "<f4")
    events["severity"] = _array([struct.unpack_from("<I", data, r.offset + 36)[0] for r in selected[111]], "<u4")
    for index, name in enumerate((
        "wave_start_ticks", "wave_end_ticks", "rms_start_ticks", "rms_end_ticks",
        "msv_start_ticks", "msv_end_ticks", "transient_start_ticks", "transient_end_ticks",
    )):
        events[name] = _array([_ticks(data, r.offset + 40 + 8 * index) for r in selected[111]], "<u8")
    trend_rows, trend_values = [], []
    for r in selected[119]:
        trend_rows.append((r.offset, *_times(data, r)))
        trend_values.append(np.frombuffer(data, dtype="<f4", count=18, offset=r.offset + 24))
    trends = _metadata(trend_rows)
    trends["values"] = _array(trend_values, "<f4", (-1, 6, 3))
    groups = {
        "events": events,
        "waveforms": _samples(data, selected[105], transient=False),
        "transients": _samples(data, selected[112], transient=True),
        "trends": trends,
    }
    return DecodedFile(records, groups, len(data), hashlib.sha256(data).hexdigest(), data[:32].hex())


def decode_file(path: str | Path) -> DecodedFile:
    """Read one file into memory and decode. The input is never modified."""
    return decode_bytes(Path(path).read_bytes())
