# tests/test_smoke.py

import pytest

def test_app_import():
    """
    A simple smoke test that tries to import the main application module.
    If this fails, something is seriously wrong with the setup or dependencies.
    """
    try:
        from src import tech_doc_helper
    except ImportError as e:
        pytest.fail(f"Failed to import the tech_doc_helper module: {e}")
