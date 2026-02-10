import subprocess
import time
import requests


def wait_for_health(url: str, timeout_s: int = 20) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("API did not become healthy in time")


def test_docker_api_smoke() -> None:
    # 1) Run container
    proc = subprocess.Popen(
        ["docker", "run", "--rm", "-p", "8000:8000", "mini02-regression"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # 2) Wait until /health is OK
        wait_for_health("http://127.0.0.1:8000/health")

        # 3) Single predict
        payload = {
            "MedInc": 8.3252,
            "HouseAge": 41,
            "AveRooms": 6.984127,
            "AveBedrms": 1.02381,
            "Population": 322,
            "AveOccup": 2.555556,
            "Latitude": 37.88,
            "Longitude": -122.23,
        }
        r = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=3)
        assert r.status_code == 200
        assert "prediction" in r.json()

        # 4) Batch predict
        batch = [payload, payload]
        r = requests.post("http://127.0.0.1:8000/predict_batch", json=batch, timeout=3)
        assert r.status_code == 200
        data = r.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    finally:
        # Stop container
        proc.terminate()
        proc.wait(timeout=10)
