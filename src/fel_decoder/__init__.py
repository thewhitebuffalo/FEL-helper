"""Decode the observed FEL record profile without assuming wiring or clock accuracy."""

from ._version import __version__
from .core import DecodedFile, FelError, Record, decode_bytes, decode_file, scan_bytes

__all__ = ["DecodedFile", "FelError", "Record", "decode_bytes", "decode_file", "scan_bytes"]
