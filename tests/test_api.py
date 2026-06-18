import io

from fastapi.testclient import TestClient
from PIL import Image


class _StubInferenceController:
    def __init__(self, *_args, **_kwargs):
        self.metadata = type(
            "_Meta",
            (),
            {
                "model_name": "stub-architecture",
                "parsed_classes": {"0_real": 0, "1_fake": 1},
            },
        )()

    @property
    def class_to_idx(self):
        return {"0_real": 0, "1_fake": 1}

    def predict(self, _image):
        return 0.995


class _StubCaptionController:
    def generate_caption(self, _image):
        return "test caption"


def test_verify_image(monkeypatch):
    from ai_detector_model.api import api

    monkeypatch.setattr(api, "get_inference_engine", lambda: _StubInferenceController())
    monkeypatch.setattr(api, "get_caption_engine", lambda: _StubCaptionController())
    client = TestClient(api.app)

    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test.png", buf, "image/png")}
    response = client.post("/verify/image", files=files, data={"type": "image"})

    assert response.status_code == 200
    assert response.json() == {"certainty": 0.995, "caption": "test caption"}


def test_endpoint_logic(monkeypatch):
    from ai_detector_model.api import api

    monkeypatch.setattr(api, "get_inference_engine", lambda: _StubInferenceController())
    monkeypatch.setattr(api, "get_caption_engine", lambda: _StubCaptionController())
    client = TestClient(api.app)

    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test.png", buf, "image/png")}
    response = client.post("/verify/image", files=files, data={"type": "image"})

    assert response.status_code == 200
    json_data = response.json()
    assert "certainty" in json_data
    assert isinstance(json_data["certainty"], (float, int))
    assert 0 <= json_data["certainty"] <= 1

    assert "caption" in json_data
    assert isinstance(json_data["caption"], str)
