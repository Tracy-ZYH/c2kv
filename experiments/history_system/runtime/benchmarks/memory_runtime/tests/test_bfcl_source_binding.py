"""The chosen BFCL checkout must win over an ambient installed package."""
import sys
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.event_native_bfcl import bind_benchmark


def make_package(root, value):
    package = root / 'bfcl_eval'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text(f'identity = {value!r}\n')
    return package


def test_requested_checkout_wins(tmp_path, monkeypatch):
    ambient = tmp_path / 'ambient'
    requested = tmp_path / 'requested'
    make_package(ambient, 'wrong')
    package = make_package(requested, 'correct')
    monkeypatch.setattr(sys, 'path', [str(ambient), *sys.path])
    monkeypatch.delitem(sys.modules, 'bfcl_eval', raising=False)
    receipt = bind_benchmark(requested)
    assert receipt['imported_package'] == str((package / '__init__.py').resolve())
    assert sys.modules['bfcl_eval'].identity == 'correct'
    assert len(receipt['sha256']['__init__.py']) == 64


def test_already_imported_other_checkout_is_rejected(tmp_path, monkeypatch):
    requested = tmp_path / 'requested'
    make_package(requested, 'correct')
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setitem(sys.modules, 'bfcl_eval', SimpleNamespace(__file__=str(tmp_path / 'wrong/__init__.py')))
    with pytest.raises(RuntimeError, match='differs from requested checkout'):
        bind_benchmark(requested)


def test_missing_checkout_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='bfcl_eval/__init__.py'):
        bind_benchmark(tmp_path)
