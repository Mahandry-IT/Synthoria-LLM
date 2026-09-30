from app.core.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(gemini_api_key="k", **overrides)


def test_gemini_chains_default_to_flash_lite_then_flash_when_unset():
    settings = _settings(gemini_model_flash="flash-full", gemini_model_flash_lite="flash-lite")

    assert settings.gemini_chain_generation == ["flash-lite", "flash-full"]
    assert settings.gemini_chain_light == ["flash-lite", "flash-full"]
    assert settings.gemini_chain_search == ["flash-full", "flash-lite"]


def test_gemini_chains_stay_synced_with_flash_model_overrides():
    """Le défaut est calculé depuis gemini_model_flash/gemini_model_flash_lite, pas figé — pas de
    double configuration à maintenir en synchronisation manuelle."""
    settings = _settings(gemini_model_flash="custom-flash", gemini_model_flash_lite="custom-lite")

    assert "custom-flash" in settings.gemini_chain_generation
    assert "custom-lite" in settings.gemini_chain_generation


def test_explicit_gemini_chain_overrides_the_derived_default():
    settings = _settings(gemini_chain_generation=["model-a", "model-b", "model-c"])

    assert settings.gemini_chain_generation == ["model-a", "model-b", "model-c"]
