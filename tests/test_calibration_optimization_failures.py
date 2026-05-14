"""Optimization-failure path coverage for calibration fitters (v0.26.0).

Closes the audit gap from the v0.26.0 toolkit completeness sweep:
``calibration.py:819`` (Platt, L-BFGS-B failure) and
``calibration.py:1022`` (Temperature, bounded scalar failure) raise
RuntimeError when SciPy's optimizer reports non-convergence. Both
paths are defensive — SciPy rarely fails to converge on well-formed
inputs — so this test module uses ``monkeypatch`` to substitute a
fake optimizer that returns ``success=False``, then asserts the
RuntimeError fires with the documented message.

Coverage matrix (raise-site → test):

| File:line in calibration.py | Test |
|---|---|
| L819 (Platt L-BFGS-B failure) | test_platt_fit_raises_on_optimizer_failure |
| L1022 (temperature bounded-scalar failure) | test_temperature_fit_raises_on_optimizer_failure |

These tests pin the error contract so a future refactor that drops
the ``if not result.success: raise`` guard would fail the suite.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from eval_toolkit import calibration as cal_mod
from eval_toolkit.calibration import fit_platt_calibrator, fit_temperature


def test_platt_fit_raises_on_optimizer_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """``fit_platt_calibrator`` raises RuntimeError when L-BFGS-B reports failure.

    Covers ``calibration.py:819``. Monkeypatches ``scipy.optimize.minimize``
    inside the calibration module to return a fake result with
    ``success=False``; asserts the documented error message fires.
    """

    def _fake_minimize(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            success=False,
            message="ABNORMAL_TERMINATION_IN_LNSRCH (mocked)",
            x=np.array([0.0, 0.0]),
        )

    monkeypatch.setattr(cal_mod, "minimize", _fake_minimize)
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=100)
    s = rng.uniform(0.0, 1.0, size=100)
    with pytest.raises(RuntimeError, match="Platt calibration optimization failed"):
        fit_platt_calibrator(y, s)


def test_temperature_fit_raises_on_optimizer_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """``fit_temperature`` raises RuntimeError when ``minimize_scalar`` reports failure.

    Covers ``calibration.py:1022``. Same monkeypatch pattern as the
    Platt test above, applied to ``minimize_scalar``.
    """

    def _fake_minimize_scalar(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            success=False,
            message="bounded scalar minimize did not converge (mocked)",
            x=1.0,
        )

    monkeypatch.setattr(cal_mod, "minimize_scalar", _fake_minimize_scalar)
    rng = np.random.default_rng(42)
    val_logits = rng.normal(size=(200, 2))
    val_labels = (val_logits[:, 1] > val_logits[:, 0]).astype(int)
    with pytest.raises(RuntimeError, match="temperature optimization failed"):
        fit_temperature(val_logits, val_labels)


def test_platt_fit_succeeds_on_well_formed_input() -> None:
    """Positive control: ``fit_platt_calibrator`` succeeds on standard input.

    Guards against the monkeypatch tests above accidentally pinning a
    behavior that's broken on real inputs.
    """
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200)
    s = (y + rng.normal(0, 0.5, size=200)).clip(0, 1)
    fit = fit_platt_calibrator(y, s)
    assert isinstance(fit.a, float)
    assert isinstance(fit.b, float)


def test_temperature_fit_succeeds_on_well_formed_input() -> None:
    """Positive control: ``fit_temperature`` succeeds on standard input."""
    rng = np.random.default_rng(42)
    base = rng.normal(size=(500, 2))
    labels = (base[:, 1] > base[:, 0]).astype(int)
    logits = base * 3.0  # makes scores overconfident; T_true ≈ 3
    result = fit_temperature(logits, labels)
    assert 0.05 <= result["temperature"] <= 20.0
    assert result["nll_post"] <= result["nll_pre"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
