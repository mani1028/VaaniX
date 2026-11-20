"""Platform registry loads without network."""

from bhasha_platform.catalog import load_registry


def test_registry_has_open_families() -> None:
    data = load_registry()
    assert data["product_name"] == "Bhasha Platform"
    assert "asr" in data and "tts" in data and "translate" in data
    assert "sarvam_weights" in data["excluded"]
    assert data["asr"]["backends"]["telugu_micro"]["license"] == "Apache-2.0"
    assert data["translate"]["backends"]["indictrans2_indic_en_dist"]["license"] == "MIT"


def test_platform_health_route_mounted() -> None:
    from fastapi.testclient import TestClient

    import os

    os.environ.setdefault("TELUGU_SPEECH_AUTH", "local_open")
    os.environ.setdefault("BHASHA_ENABLE_PLATFORM", "1")
    from telugu_speech.api.main import app

    client = TestClient(app)
    health = client.get("/platform/v1/health")
    assert health.status_code == 200
    body = health.json()
    assert body["product"] == "Bhasha Platform"
    catalog = client.get("/platform/v1/catalog")
    assert catalog.status_code == 200
    assert catalog.json()["scope"] == "indic_speech_and_text"
