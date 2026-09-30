"""Lossless array export and optional, record-aware CSV tables.

Integer timestamps are never converted to floating-point epoch seconds. Channel
numbers describe slots in the observed layout, not phases or electrical units.
"""

import csv
import json
from pathlib import Path

import numpy as np

from .core import DecodedFile


def _native(value):
    """Give the CSV writer Python numbers without changing integer precision."""
    return value.item() if isinstance(value, np.generic) else value


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows((_native(value) for value in row) for row in rows)


def _record_rows(arrays, record_indices):
    for i, offset in enumerate(arrays["record_offset"]):
        yield i, (
            record_indices[int(offset)], int(offset),
            int(arrays["start_ticks"][i]), int(arrays["end_ticks"][i]),
        )


def _event_rows(arrays, record_indices):
    fields = [key for key in arrays if key not in ("record_offset", "start_ticks", "end_ticks")]
    for i, metadata in _record_rows(arrays, record_indices):
        yield (*metadata, *(arrays[key][i] for key in fields))


def _sample_rows(arrays, record_indices, transient):
    for i, metadata in _record_rows(arrays, record_indices):
        begin = int(arrays["sample_start"][i])
        count = int(arrays["sample_count"][i])
        for local_index in range(count):
            sample = begin + local_index
            relative_time = (
                arrays["sample_rate_hz"][i] if transient
                else arrays["offset_us"][sample]
            )
            yield (
                *metadata, local_index, relative_time,
                *arrays["raw"][sample], *arrays["values"][sample],
            )


def _trend_rows(arrays, record_indices):
    for i, metadata in _record_rows(arrays, record_indices):
        yield (*metadata, *arrays["values"][i].flat)


def export_decoded(
    decoded: DecodedFile,
    output: str | Path,
    *,
    source_name: str,
    csv_samples: bool = False,
    compress: bool = False,
) -> Path:
    """Write a new export directory; refuse every existing path.

    ``manifest.json`` is written last and marks successful completion. If a disk
    or permission error interrupts export, a partial directory can remain without
    that marker. No existing directory or source recording is modified.
    """
    destination = Path(output)
    destination.mkdir(parents=False, exist_ok=False)
    save = np.savez_compressed if compress else np.savez
    files = []
    for name, arrays in decoded.groups.items():
        filename = f"{name}.npz"
        save(destination / filename, **arrays)
        files.append(filename)

    record_indices = {record.offset: i for i, record in enumerate(decoded.records)}
    _write_csv(
        destination / "records.csv",
        ["record_index", "offset", "tag", "size"],
        ((i, record.offset, record.tag, record.size) for i, record in enumerate(decoded.records)),
    )
    metadata = ["record_index", "record_offset", "start_ticks", "end_ticks"]
    event_fields = [key for key in decoded.groups["events"] if key not in metadata]
    _write_csv(
        destination / "events.csv",
        [*metadata, *event_fields],
        _event_rows(decoded.groups["events"], record_indices),
    )
    files.extend(["records.csv", "events.csv"])

    if csv_samples:
        for name, channels, transient in (("waveforms", 8, False), ("transients", 4, True)):
            _write_csv(
                destination / f"{name}.csv",
                [
                    *metadata, "sample_index", "sample_rate_hz" if transient else "offset_us",
                    *(f"channel_{i}_raw" for i in range(1, channels + 1)),
                    *(f"channel_{i}_value" for i in range(1, channels + 1)),
                ],
                _sample_rows(decoded.groups[name], record_indices, transient),
            )
            files.append(f"{name}.csv")
        _write_csv(
            destination / "trends.csv",
            [*metadata, *(f"channel_{i}_field_{j}" for i in range(1, 7) for j in range(1, 4))],
            _trend_rows(decoded.groups["trends"], record_indices),
        )
        files.append("trends.csv")

    manifest = decoded.manifest()
    manifest.update({
        "source_filename": Path(source_name).name,
        "export": {
            "npz_compressed": compress,
            "csv_samples": csv_samples,
            "files": files,
            "record_index": "Zero-based index in records.csv, including uninterpreted records.",
            "sample_index": "Zero-based within each capture record, never a continuous stream index.",
            "transient_relative_time": "sample_index / sample_rate_hz seconds from the record start_ticks.",
            "trend_fields": "Three stored fields per numbered channel slot. Minimum, maximum, average ordering is inferred from the validation corpus, not independently vendor-validated.",
        },
    })
    # Publish the completion marker only after every data file and the complete
    # manifest have closed. A failed manifest write must not look complete.
    temporary_manifest = destination / ".manifest.json.tmp"
    with temporary_manifest.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary_manifest.replace(destination / "manifest.json")
    return destination
