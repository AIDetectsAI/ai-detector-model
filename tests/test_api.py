import io

from ai_detector_model.api.api import APIController, app
from fastapi.testclient import TestClient
from PIL import Image


def test_verify_image(monkeypatch):

    async def mock_get_image_certainty(self, file, filetype):
        return 0.995

    monkeypatch.setattr(APIController, 'get_image_certainty', mock_get_image_certainty)

    client = TestClient(app)

    files = {'file': 'test.png'}
    data = {'type': 'image'}
    response = client.post('/verify/image', files=files, data=data)

    assert response.status_code == 200
    assert response.json()['certainty'] == 0.995


def test_endpoint_logic():
    img = Image.new('RGB', (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    client = TestClient(app)
    files = {'file': ('test.png', buf, 'image/png')}
    data = {'type': 'image'}
    response = client.post('/verify/image', files=files, data=data)

    assert response.status_code == 200
    json_data = response.json()
    assert 'certainty' in json_data
    assert isinstance(json_data['certainty'], (float, int))
    assert 0 <= json_data['certainty'] <= 1
