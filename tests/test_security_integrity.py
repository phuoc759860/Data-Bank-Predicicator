"""Security, deployment configuration, and data-integrity checks."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "bank-full.csv"
EXPECTED_RAW_SHA256 = "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d"


def test_original_raw_dataset_is_unchanged() -> None:
    digest = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
    assert digest == EXPECTED_RAW_SHA256
    assert RAW_PATH.stat().st_size == 4_610_348


def test_gitignore_covers_caches_secrets_and_environments() -> None:
    content = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in [
        "__pycache__/",
        ".pytest_cache/",
        ".venv/",
        ".env",
        ".streamlit/secrets.toml",
    ]:
        assert pattern in content


def test_upload_limits_and_csv_restriction_are_configured() -> None:
    config = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "maxUploadSize = 50" in config
    page_source = (
        PROJECT_ROOT / "app" / "pages" / "3_Batch_Prediction.py"
    ).read_text(encoding="utf-8")
    assert 'type=["csv"]' in page_source
    assert "validate_raw_input" in page_source
    assert "predict_batch" in page_source


def test_no_hardcoded_credentials_or_paid_ai_clients() -> None:
    python_files = list((PROJECT_ROOT / "app").rglob("*.py")) + list(
        (PROJECT_ROOT / "src").rglob("*.py")
    )
    forbidden_patterns = [
        re.compile(r"api[_-]?key\s*=\s*['\"][^'\"]+", re.IGNORECASE),
        re.compile(r"password\s*=\s*['\"][^'\"]+", re.IGNORECASE),
        re.compile(r"secret\s*=\s*['\"][^'\"]+", re.IGNORECASE),
        re.compile(r"bearer\s+[a-z0-9._-]{8,}", re.IGNORECASE),
        re.compile(r"\b(openai|anthropic|cohere|gemini)\b", re.IGNORECASE),
    ]
    for path in python_files:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert not pattern.search(text), f"Sensitive pattern in {path}: {pattern.pattern}"


def test_app_source_has_no_destructive_raw_data_calls() -> None:
    app_files = list((PROJECT_ROOT / "app").rglob("*.py"))
    destructive = re.compile(
        r"(to_csv|write_text|write_bytes|open)\s*\([^)]*bank-full", re.IGNORECASE
    )
    for path in app_files:
        text = path.read_text(encoding="utf-8")
        assert not destructive.search(text), f"Potential raw-data write in {path}"
