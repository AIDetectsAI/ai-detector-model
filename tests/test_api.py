import pytest
from fastapi.testclient import TestClient
from fastapi import File
from ..api import api

def test_verify_image(monkeypatch):

    async def mock_get_image_certainty(self, file, filetype):
        return 0.995
     
    monkeypatch.setattr(api.APIController, "get_image_certainty", mock_get_image_certainty)
    
    client = TestClient(api.app)

    files = {"file": "test.png"}
    data = {"type": "image"}
    response = client.post("/verify/image", files=files, data=data)

    assert response.status_code == 200
    assert response.json()["certainty"] == 0.995