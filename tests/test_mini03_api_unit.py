from fastapi.testclient import TestClient
from mini03.api import app


def test_segment_vip() -> None:
    with TestClient(app) as client:
        payload = {"purchases_30d": 18, "spend_30d": 1300}
        r = client.post("/segment", json=payload)
        assert r.status_code == 200
        assert r.json()["segment"] == "VIP"
