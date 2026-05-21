"""Coverage-focused tests for small modules: evidence, config, __init__, operating_points.

Each module has a handful of missing-line branches (validation errors,
optional-feature paths, lazy-import behavior) too small to warrant a
dedicated test file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import eval_toolkit
from eval_toolkit.config import from_yaml, frozen_config
from eval_toolkit.evidence import AggregateEvidence, EvidenceAxis, PairingMetadata
from eval_toolkit.operating_points import OperatingPointSpec
from eval_toolkit.thresholds import MaxF1Selector

# ---------------------------------------------------------------------------
# evidence.py: validation + to_dict-with-metadata branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_evidence_axis_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name must be non-empty"):
        EvidenceAxis(name="", value="v")


@pytest.mark.unit
def test_evidence_axis_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="value must be non-empty"):
        EvidenceAxis(name="n", value="")


@pytest.mark.unit
def test_aggregate_evidence_rejects_empty_method() -> None:
    with pytest.raises(ValueError, match="method must be non-empty"):
        AggregateEvidence(status="inferential", method="")


@pytest.mark.unit
def test_aggregate_evidence_to_dict_includes_metadata_when_set() -> None:
    """Metadata dict appears in to_dict() output only when non-empty (line 89)."""
    bare = AggregateEvidence(status="inferential", method="bootstrap")
    assert "metadata" not in bare.to_dict()

    enriched = AggregateEvidence(
        status="inferential",
        method="bootstrap",
        metadata={"n_resamples": 1000},
    )
    assert enriched.to_dict()["metadata"] == {"n_resamples": 1000}


@pytest.mark.unit
def test_pairing_metadata_round_trip() -> None:
    pm = PairingMetadata(paired=True, unit="row", valid_scope="dev", notes="rationale")
    assert pm.to_dict() == {
        "paired": True,
        "unit": "row",
        "valid_scope": "dev",
        "notes": "rationale",
    }


# ---------------------------------------------------------------------------
# config.py: from_yaml validation branches
# ---------------------------------------------------------------------------


@frozen_config
class _Cfg:
    lr: float
    batch_size: int = 16

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")


@pytest.mark.unit
def test_from_yaml_rejects_non_mapping_root(tmp_path: Path) -> None:
    """YAML root must be a mapping (line 105)."""
    pytest.importorskip("yaml")
    bad = tmp_path / "scalar.yaml"
    bad.write_text("just-a-string\n")
    with pytest.raises(TypeError, match="YAML root must be a mapping"):
        from_yaml(bad, _Cfg)


@pytest.mark.unit
def test_from_yaml_raises_when_yaml_missing(tmp_path: Path, monkeypatch) -> None:
    """When pyyaml is not importable, from_yaml raises ImportError with hint."""
    import sys

    monkeypatch.setitem(sys.modules, "yaml", None)  # forces re-import to fail
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("lr: 0.001\n")
    with pytest.raises(ImportError, match="from_yaml requires pyyaml"):
        from_yaml(cfg_file, _Cfg)


# ---------------------------------------------------------------------------
# __init__.py: lazy import + __dir__
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dunder_version_accessible_via_getattr() -> None:
    """__getattr__('__version__') returns the constant (line 215)."""
    # Direct attribute access goes through __getattr__ when the name isn't
    # already bound in globals().
    v = eval_toolkit.__version__
    assert isinstance(v, str)
    assert v.count(".") >= 2  # e.g. "0.10.0"


@pytest.mark.unit
def test_dunder_dir_returns_sorted_all() -> None:
    """__dir__ returns the sorted lazy export list.

    Updated at v0.46: `pr_auc` is no longer in `__all__` per Decision L
    (soft-deprecated at v0.46; hard-removed at v0.47). The deprecated names
    are still resolvable at the top level via the transitional
    `__getattr__` branch — they just aren't advertised in `__dir__`.
    """
    listing = dir(eval_toolkit)
    assert listing == sorted(listing)
    # Non-deprecated top-level symbols still appear:
    assert "evaluate" in listing
    assert "scorecard" in listing  # v0.46 new
    assert "bootstrap_ci" in listing
    # Deprecated scalars are NO LONGER in __dir__ (removed from _EXPORTS at v0.46):
    assert "pr_auc" not in listing
    assert "brier_score" not in listing


@pytest.mark.unit
def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'definitely_not_a_symbol'"):
        eval_toolkit.definitely_not_a_symbol  # noqa: B018


# ---------------------------------------------------------------------------
# operating_points.py: OperatingPointSpec __post_init__ validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_op_spec_rejects_empty_fit_slice() -> None:
    with pytest.raises(ValueError, match="fit_slice must be non-empty"):
        OperatingPointSpec(fit_slice="", apply_slices=("a",), selectors=(MaxF1Selector(),))


@pytest.mark.unit
def test_op_spec_rejects_empty_apply_slices() -> None:
    with pytest.raises(ValueError, match="apply_slices must be non-empty"):
        OperatingPointSpec(fit_slice="A", apply_slices=(), selectors=(MaxF1Selector(),))


@pytest.mark.unit
def test_op_spec_rejects_empty_selectors() -> None:
    with pytest.raises(ValueError, match="selectors must be non-empty"):
        OperatingPointSpec(fit_slice="A", apply_slices=("a",), selectors=())


@pytest.mark.unit
def test_op_spec_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name must be non-empty"):
        OperatingPointSpec(
            fit_slice="A",
            apply_slices=("a",),
            selectors=(MaxF1Selector(),),
            name="",
        )


@pytest.mark.unit
def test_op_spec_rejects_empty_apply_slice_name() -> None:
    with pytest.raises(ValueError, match="apply_slices must not contain empty names"):
        OperatingPointSpec(
            fit_slice="A",
            apply_slices=("a", ""),
            selectors=(MaxF1Selector(),),
        )


@pytest.mark.unit
def test_op_spec_rejects_empty_scorer_name() -> None:
    with pytest.raises(ValueError, match="scorer_names must not contain empty names"):
        OperatingPointSpec(
            fit_slice="A",
            apply_slices=("a",),
            selectors=(MaxF1Selector(),),
            scorer_names=("m", ""),
        )
