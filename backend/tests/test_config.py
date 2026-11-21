"""Unit tests for configuration validation and fail-fast startup behavior."""
import pytest
from app.config import Settings
from app.exceptions import ConfigurationError


def test_missing_openai_key_fails():
    s = Settings(
        openai_api_key="",
        openrouter_api_key="valid-key",
        database_url="mysql+pymysql://user:pass@localhost:3306/db"
    )
    with pytest.raises(ConfigurationError) as exc_info:
        s.validate()
    assert "CONFIG_OPENAI_MISSING" in str(exc_info.value)
    assert exc_info.value.code == "CONFIG_OPENAI_MISSING"


def test_missing_openrouter_key_fails():
    s = Settings(
        openai_api_key="valid-openai-key",
        openrouter_api_key="",
        database_url="mysql+pymysql://user:pass@localhost:3306/db"
    )
    with pytest.raises(ConfigurationError) as exc_info:
        s.validate()
    assert "CONFIG_OPENROUTER_MISSING" in str(exc_info.value)
    assert exc_info.value.code == "CONFIG_OPENROUTER_MISSING"


def test_missing_database_url_fails():
    s = Settings(
        openai_api_key="valid-openai-key",
        openrouter_api_key="valid-openrouter-key",
        database_url=""
    )
    with pytest.raises(ConfigurationError) as exc_info:
        s.validate()
    assert "DATABASE_CONNECTION_FAILED" in str(exc_info.value)
    assert exc_info.value.code == "DATABASE_CONNECTION_FAILED"


def test_valid_configuration_passes():
    s = Settings(
        openai_api_key="valid-openai-key",
        openrouter_api_key="valid-openrouter-key",
        database_url="mysql+pymysql://user:pass@localhost:3306/classicmodels"
    )
    s.validate()  # should not raise
