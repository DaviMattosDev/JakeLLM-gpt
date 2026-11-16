import pytest

from fastapi.testclient import TestClient

from api.main import app, runtime


@pytest.fixture(autouse=True)
def clear_runtime_cache():
    runtime.cache_clear()
    yield
    runtime.cache_clear()


def test_health_does_not_load_model():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_ready_without_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("JAKELLM_CHECKPOINT", str(tmp_path / "not-found.pt"))
    monkeypatch.setenv("JAKELLM_TOKENIZER_DIR", str(tmp_path))
    response = TestClient(app).get("/ready")
    assert response.status_code == 503


def test_api_validates_payload_without_initializing_model():
    response = TestClient(app).post("/generate", json={"prompt": "x", "max_new_tokens": 1000})
    assert response.status_code == 422


def test_index_is_served():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "JakeLLM" in response.text
