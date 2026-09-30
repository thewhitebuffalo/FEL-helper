"""Release identity agrees across the package, CLI, exports and distribution."""

from importlib import metadata
import struct

import pytest

from fel_decoder import __version__, decode_bytes
from fel_decoder._version import __version__ as release_version
from fel_decoder.cli import main


def _event_recording():
    """One synthetic event is sufficient to obtain an export manifest."""
    data = bytearray(32 + 104)
    data[:4] = bytes.fromhex("0bba1400")
    struct.pack_into("<I", data, 4, 104)
    data[24:32] = bytes.fromhex("0000090000000000")
    struct.pack_into("<HH", data, 32, 111, 104)
    return bytes(data)


def test_package_and_manifest_share_release_version():
    assert release_version == "0.2.0"
    assert __version__ == release_version
    assert decode_bytes(_event_recording()).manifest()["decoder_version"] == release_version


def test_cli_reports_release_version(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--version"])
    assert result.value.code == 0
    assert capsys.readouterr().out.strip() == f"FEL-helper {release_version}"


def test_installed_distribution_matches_runtime_release():
    try:
        installed_version = metadata.version("FEL-helper")
    except metadata.PackageNotFoundError:
        pytest.skip("Install the package to verify built distribution metadata")
    assert installed_version == release_version
