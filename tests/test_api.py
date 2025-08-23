import pytest
from fastapi.testclient import TestClient
from fastapi import File
from ai_detector_model.api import app
from ai_detector_model.api import APIController

def test_verify_image(monkeypatch):

    async def mock_get_image_certainty(self, file, filetype):
        return 0.995
     
    monkeypatch.setattr(APIController, "get_image_certainty", mock_get_image_certainty)
    
    client = TestClient(app)

    files = {"file": "test.png"}
    data = {"type": "image"}
    response = client.post("/verify/image", files=files, data=data)

    assert response.status_code == 200
    assert response.json()["certainty"] == 0.995