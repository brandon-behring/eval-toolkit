"""v0.9 decoupling and evidence-contract roundtrip tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from eval_toolkit.analysis import (
    bootstrap_metric_from_predictions,
    paired_diff_from_prediction_refs,
)
from eval_toolkit.artifacts import (
    PredictionArtifactRef,
    PredictionColumns,
    validate_manifest,
    validate_payload,
)
from eval_toolkit.claims import (
    ClaimSpec,
    evaluate_claims,
    external_diagnostic_gate,
    source_role_gate,
    strict_artifact_gate,
)
from eval_toolkit.evidence import AggregateEvidence, EvidenceAxis, PairingMetadata
from eval_toolkit.harness import RunResult
from eval_toolkit.manifest import SourceRoleRecord, build_manifest, write_manifest
from eval_toolkit.provenance import file_sha256
from eval_toolkit.text_dedup import ExactNormalizedHashStrategy, audit_source_label_similarity
from eval_toolkit.thresholds import CISafeThresholdSelector, ThresholdPolicyMetadata


def _write_prediction_csv(
    path: Path,
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    scorer: str,
) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["row_id", "content_hash", "label", "score", "scorer", "slice", "source"],
        )
        writer.writeheader()
        for i, (label, score) in enumerate(zip(labels, scores, strict=True)):
            writer.writerow(
                {
                    "row_id": f"row-{i}",
                    "content_hash": f"sha256:text-{i}",
                    "label": int(label),
                    "score": float(score),
                    "scorer": scorer,
                    "slice": "locked_eval",
                    "source": "locked_eval_fixture",
                }
            )


def _prediction_ref(path: Path, *, scorer: str, n_rows: int) -> PredictionArtifactRef:
    digest = file_sha256(path)
    assert digest is not None
    return PredictionArtifactRef(
        uri=str(path),
        media_type="text/csv",
        sha256=f"sha256:{digest}",
        n_rows=n_rows,
        columns=PredictionColumns(
            row_id="row_id",
            content_hash="content_hash",
            label="label",
            score="score",
            scorer="scorer",
            slice="slice",
            provenance={"source": "source"},
        ),
        metadata={"scorer": scorer},
    )


@pytest.mark.unit
def test_v09_generic_contract_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    labels = np.array([0, 1] * 50, dtype=int)
    baseline_scores = np.clip(0.42 + 0.16 * labels + rng.normal(0, 0.18, size=len(labels)), 0, 1)
    candidate_scores = np.clip(0.36 + 0.32 * labels + rng.normal(0, 0.16, size=len(labels)), 0, 1)

    baseline_path = tmp_path / "baseline_predictions.csv"
    candidate_path = tmp_path / "candidate_predictions.csv"
    _write_prediction_csv(baseline_path, labels=labels, scores=baseline_scores, scorer="baseline")
    _write_prediction_csv(
        candidate_path, labels=labels, scores=candidate_scores, scorer="candidate"
    )
    baseline_ref = _prediction_ref(baseline_path, scorer="baseline", n_rows=len(labels))
    candidate_ref = _prediction_ref(candidate_path, scorer="candidate", n_rows=len(labels))

    selector = CISafeThresholdSelector(
        max_fpr=0.15,
        max_fpr_ci_upper=0.25,
        min_recall=0.50,
        confidence=0.95,
    )
    operating_point = selector.selected_operating_point(
        labels,
        candidate_scores,
        bootstrap_selected=True,
        n_resamples=30,
        seed=11,
    )
    threshold_policy = ThresholdPolicyMetadata(
        calibration_slice="calibration",
        score_column="score",
        selector=selector.criterion,
        constraints=selector.constraints,
        claim_enabled=True,
    )
    prediction_refs = [baseline_ref.to_dict(), candidate_ref.to_dict()]
    result = RunResult(
        run_id="v09",
        git_sha=None,
        config={"scenario": "sdd-v3-shaped-generic"},
        by_slice={
            "locked_eval": {
                "n": len(labels),
                "n_positive": int(labels.sum()),
                "by_scorer": {
                    "candidate": {
                        "ci_safe_operating_point": operating_point,
                        "scores": candidate_scores.tolist(),
                    }
                },
                "paired_diffs": {},
            }
        },
        prediction_artifacts=prediction_refs,
        evidence_axes=[EvidenceAxis(name="slice", value="locked_eval").to_dict()],
        pairing_metadata=PairingMetadata(
            paired=True, unit="row", valid_scope="locked_eval"
        ).to_dict(),
        aggregate_evidence=AggregateEvidence(
            status="inferential",
            method="paired bootstrap over retained predictions",
            axes=(EvidenceAxis(name="slice", value="locked_eval"),),
        ).to_dict(),
        threshold_policy=threshold_policy.to_dict(),
    )
    manifest = build_manifest(
        run_id="v09",
        config=result.config,
        source_roles=[
            SourceRoleRecord(source="train_pool", role="train", n_rows=1000),
            SourceRoleRecord(source="calibration", role="calibration", n_rows=200),
            SourceRoleRecord(source="locked_eval", role="locked_eval", n_rows=len(labels)),
            SourceRoleRecord(source="notinject", role="external_diagnostic", n_rows=50),
        ],
        required_source_roles=("train", "calibration", "locked_eval", "external_diagnostic"),
        prediction_artifacts=(baseline_ref, candidate_ref),
    )

    result_payload = result.to_dict()
    manifest_payload = manifest.to_dict()
    validate_payload(result_payload, "results_full.v1.json")
    validate_manifest(manifest_payload)
    loaded_manifest = json.loads(write_manifest(manifest, tmp_path / "run").read_text())
    assert loaded_manifest["prediction_artifacts"][0]["n_rows"] == len(labels)

    paired = paired_diff_from_prediction_refs(
        baseline_ref.to_dict(),
        candidate_ref.to_dict(),
        n_resamples=40,
        seed=13,
    )
    bootstrapped = bootstrap_metric_from_predictions(
        candidate_ref.to_dict(),
        n_resamples=40,
        seed=13,
    )
    assert paired["delta"] >= 0
    assert bootstrapped["n_resamples"] == 40

    claim = ClaimSpec(
        name="locked eval evidence",
        gates=(
            strict_artifact_gate(),
            source_role_gate(("train", "calibration", "locked_eval", "external_diagnostic")),
            external_diagnostic_gate("aggregate_evidence.status"),
        ),
    )
    report = evaluate_claims(result_payload, [claim], manifest=manifest_payload)
    assert not report.has_failures()


@pytest.mark.unit
def test_source_label_similarity_audit_reports_without_dropping() -> None:
    report = audit_source_label_similarity(
        ["repeat me", "Repeat   me", "distinct", "repeat me"],
        sources=["train", "locked_eval", "external_diagnostic", "train"],
        labels=[1, 1, 0, 0],
        threshold=1.0,
        k_neighbors=4,
        strategy=ExactNormalizedHashStrategy(),
    )

    assert report.n_input == 4
    assert report.n_findings >= 3
    assert any(f.relation == "cross_source_same_label" for f in report.findings)
    assert any(f.relation == "within_source_cross_label" for f in report.findings)
    assert all(f.left_index in range(4) and f.right_index in range(4) for f in report.findings)
