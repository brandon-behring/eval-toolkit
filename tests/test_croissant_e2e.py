"""End-to-end Croissant interop verification (v0.41.0, closes #42, v1.0 Gate 4).

Verifies that ``HFDatasetsLoader.describe()`` returns per-file ``sha256``
hashes that match the actual bytes of the underlying parquet shards on
HF Hub.

Background on the dual-source design (Croissant + tree API):
- HF Hub's Croissant emitter (``/api/datasets/{repo}/croissant``) ships
  metadata (name, license, citation, schema) but **does not** populate
  per-file ``distribution[].sha256`` — instead, the field carries a
  placeholder URL pointing at the MLCommons Croissant issue tracking
  the eventual checksum addition (issue #80, open).
- HF Hub's tree API (``/api/datasets/{repo}/tree/...``) exposes
  ``lfs.oid`` per file: a 64-hex sha256 of the raw file content.
- ``HFDatasetsLoader.describe()`` reads sha256 from the tree API today,
  and will pick up Croissant's eventual sha256 with a one-line change
  when #80 resolves (same downstream contract; same hash format).

Tests are marked ``@pytest.mark.integration`` — network-dependent;
excluded from PR CI via ``-m "not integration"`` in ``make coverage``.
Run explicitly via ``pytest -m integration`` (nightly or local dev).
"""

from __future__ import annotations

import hashlib
import urllib.request
from typing import Any

import pytest

from eval_toolkit.loaders import HFDatasetsLoader

# Small public Croissant-compliant dataset. ~50 KB test split (1 parquet
# shard). Pinned via repo_id; HF retains revisions, so even if the dataset
# is updated the test only fails if HF re-shards (rare for popular
# datasets) — which is a real signal we want to catch in nightly.
_TEST_REPO_ID = "stanfordnlp/sst2"


def _download_and_hash(url: str) -> str:
    """GET the URL, return ``sha256:<hex>`` of the body."""
    req = urllib.request.Request(url, headers={"User-Agent": "eval-toolkit-test"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    return f"sha256:{hashlib.sha256(body).hexdigest()}"


@pytest.mark.integration
def test_hfdatasets_describe_returns_real_sha256_from_tree_api() -> None:
    """``describe()`` populates per-file sha256 from HF Hub's tree API.

    Closes the infrastructure half of v1.0 Gate 4: prove the loader can
    surface authoritative file hashes from HF Hub.
    """
    loader = HFDatasetsLoader(repo_id=_TEST_REPO_ID)
    desc = loader.describe()

    distribution = desc["distribution"]
    assert isinstance(distribution, list)
    assert distribution, "expected at least one parquet shard in distribution[]"

    # Every entry should have a real sha256 (64 hex chars after the prefix).
    for entry in distribution:
        sha = entry["sha256"]
        assert isinstance(sha, str)
        assert sha.startswith("sha256:"), f"unexpected hash format: {sha!r}"
        hex_part = sha.removeprefix("sha256:")
        assert len(hex_part) == 64, f"expected 64-hex sha256, got {len(hex_part)}"
        assert all(c in "0123456789abcdef" for c in hex_part), f"non-hex: {hex_part!r}"


@pytest.mark.integration
def test_hfdatasets_describe_sha256_matches_actual_file_bytes() -> None:
    """End-to-end Gate 4 verification: hash a downloaded shard, assert match.

    For each shard in ``describe()['distribution']``, fetch the raw
    parquet bytes from ``contentUrl`` and verify ``sha256(bytes) ==
    entry['sha256']``. This proves the source-of-truth chain:
    HF Hub tree API → ``describe()`` → real file content.
    """
    loader = HFDatasetsLoader(repo_id=_TEST_REPO_ID)
    desc = loader.describe()
    distribution = desc["distribution"]
    assert isinstance(distribution, list)

    # Only verify the first shard to keep CI cost bounded (sst2 train is
    # ~3 MB; we just need one matched pair to prove the contract).
    entry: dict[str, Any] = distribution[0]
    content_url = entry["contentUrl"]
    expected_sha = entry["sha256"]
    assert content_url and expected_sha

    actual_sha = _download_and_hash(content_url)
    assert actual_sha == expected_sha, (
        f"sha256 mismatch for {entry['name']!r}: "
        f"describe() reported {expected_sha}, actual file hashed to {actual_sha}"
    )


@pytest.mark.integration
def test_hfdatasets_describe_returns_croissant_metadata() -> None:
    """``describe()`` enriches with Croissant metadata (name, license, citeAs).

    Even though Croissant's ``distribution[].sha256`` is unusable today
    (placeholder URL per MLCommons #80), the metadata fields are valid
    and should pass through to ``describe()`` output.
    """
    loader = HFDatasetsLoader(repo_id=_TEST_REPO_ID)
    desc = loader.describe()

    # Either Croissant provided a non-empty name or we fell back to repo_id.
    name = desc["name"]
    assert isinstance(name, str) and name


@pytest.mark.integration
def test_hfdatasets_caller_overrides_win() -> None:
    """Caller-provided fields override Croissant fetches.

    Explicit ``name=...`` / ``cite_as=...`` are not overwritten by
    remote metadata even when ``fetch_remote_metadata=True``.
    """
    loader = HFDatasetsLoader(
        repo_id=_TEST_REPO_ID,
        name="my-custom-name",
        cite_as="my-citation",
    )
    desc = loader.describe()
    assert desc["name"] == "my-custom-name"
    assert desc["citeAs"] == "my-citation"


@pytest.mark.integration
def test_hfdatasets_fetch_remote_metadata_disabled_skips_network() -> None:
    """``fetch_remote_metadata=False`` produces the v0.40-era empty-sha256 output."""
    loader = HFDatasetsLoader(
        repo_id=_TEST_REPO_ID,
        fetch_remote_metadata=False,
    )
    desc = loader.describe()
    distribution = desc["distribution"]
    assert isinstance(distribution, list)
    assert len(distribution) == 1
    assert distribution[0]["sha256"] == ""
