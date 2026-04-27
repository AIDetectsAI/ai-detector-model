import io

from ai_detector_model.api import api
from fastapi.testclient import TestClient
from PIL import Image


class _StubInferenceController:
    def __init__(self, *_args, **_kwargs):
        self.metadata = type("_Meta", (), {"model_name": "stub-model"})()
        self.class_to_idx = {"0_real": 0, "1_fake": 1}

    def predict(self, _image):
        return 0.995


def test_verify_image(monkeypatch):
    monkeypatch.setattr(api, "inference_engine", _StubInferenceController())
    client = TestClient(api.app)

    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test.png", buf, "image/png")}
    data = {"type": "image"}
    response = client.post("/verify/image", files=files, data=data)

    assert response.status_code == 200
    assert response.json() == {"certainty": 0.995}


def test_endpoint_logic(monkeypatch):
    monkeypatch.setattr(api, "inference_engine", _StubInferenceController())
    client = TestClient(api.app)

    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test.png", buf, "image/png")}
    data = {"type": "image"}
    response = client.post("/verify/image", files=files, data=data)

    assert response.status_code == 200
    json_data = response.json()
    assert "certainty" in json_data
    assert isinstance(json_data["certainty"], (float, int))
    assert 0 <= json_data["certainty"] <= 1
