"""Pins for `fmp_compat.key_override` — the scoped caller-key context manager.

Four skills honour a caller-supplied FMP key. Each used to do a process-global
`os.environ["FMP_API_KEY"] = key` at construction time, so two adapters built
with different keys in one interpreter made every LATER `fmp_get` call use the
most recently constructed adapter's key (codex gate P2, 2026-09-02). The fix is
one shared context manager here — scoped to the call, restoring what it found.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fmp_compat  # noqa: E402


@pytest.fixture(autouse=True)
def _ambient(monkeypatch):
    """A house credential is ALWAYS set in production: importing fmp_compat
    self-loads secrets.env. The fallback key is never this manager's business."""
    monkeypatch.setenv("FMP_API_KEY", "ambient-house-key")
    monkeypatch.setenv("FMP_FALLBACK_API_KEY", "ambient-fallback")


class TestKeyOverride:
    def test_caller_key_is_in_effect_inside_the_block(self):
        with fmp_compat.key_override("caller-key"):
            assert fmp_compat.get_fmp_keys()[0] == "caller-key"

    def test_ambient_key_is_restored_after_the_block(self):
        with fmp_compat.key_override("caller-key"):
            pass
        assert os.environ["FMP_API_KEY"] == "ambient-house-key"

    def test_absent_key_is_deleted_again_afterwards(self, monkeypatch):
        """MUTANT: restore with `os.environ[k] = saved or ''` -> an empty string
        survives and `get_fmp_keys()` behaves differently from a missing var."""
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        with fmp_compat.key_override("caller-key"):
            assert fmp_compat.get_fmp_keys()[0] == "caller-key"
        assert "FMP_API_KEY" not in os.environ

    @pytest.mark.parametrize("falsy", [None, ""])
    def test_a_falsy_key_is_a_no_op(self, falsy):
        with fmp_compat.key_override(falsy):
            assert fmp_compat.get_fmp_keys()[0] == "ambient-house-key"
        assert os.environ["FMP_API_KEY"] == "ambient-house-key"

    def test_the_fallback_key_is_never_touched(self):
        with fmp_compat.key_override("caller-key"):
            assert os.environ["FMP_FALLBACK_API_KEY"] == "ambient-fallback"
            # ...and the caller key does not shadow it in the failover order.
            assert fmp_compat.get_fmp_keys() == ["caller-key", "ambient-fallback"]
        assert os.environ["FMP_FALLBACK_API_KEY"] == "ambient-fallback"

    def test_the_ambient_key_is_restored_when_the_body_raises(self):
        """MUTANT: restore outside a `finally` -> one failed call leaks the
        caller key into every subsequent request in the interpreter."""
        with pytest.raises(RuntimeError):
            with fmp_compat.key_override("caller-key"):
                raise RuntimeError("upstream blew up")
        assert os.environ["FMP_API_KEY"] == "ambient-house-key"

    def test_nested_overrides_unwind_in_order(self):
        with fmp_compat.key_override("outer-key"):
            with fmp_compat.key_override("inner-key"):
                assert fmp_compat.get_fmp_keys()[0] == "inner-key"
            assert fmp_compat.get_fmp_keys()[0] == "outer-key"
        assert os.environ["FMP_API_KEY"] == "ambient-house-key"
