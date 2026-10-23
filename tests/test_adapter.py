import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from biometric_service import main

API_KEY = "test-only-secret-that-is-longer-than-thirty-two-characters"
IMAGE = ("selfie.jpg", b"fake-image-for-mocked-engine-test", "image/jpeg")
LANDMARKS = [[100, 100], [140, 100], [120, 120], [105, 140], [135, 140]]
LIVE_FACE = {
    "detection": {"landmarks": LANDMARKS},
    "liveness": {"is_live": True, "score": 0.97},
}


class FakeEngineClient:
    def __init__(self, *, live=True):
        self.live = live
        self.people = {}

    async def post(self, path, *, data=None, files=None):
        if path == "/api/v1/detect":
            face = dict(LIVE_FACE)
            if not self.live:
                face["liveness"] = {"is_live": False, "score": 0.02}
            return self._response(200, {"faces": [face]})
        if path == "/api/v1/watchlist/enroll":
            identity_id = data["identity_id"]
            if identity_id in self.people:
                return self._response(409, {"detail": "duplicate"})
            self.people[identity_id] = data["watchlist_name"]
            return self._response(200, {
                "identity_id": identity_id,
                "label": data["label"],
                "watchlist": data["watchlist_name"],
                "face": LIVE_FACE,
            })
        if path == "/api/v1/identify":
            face = dict(LIVE_FACE)
            if not self.live:
                face["liveness"] = {"is_live": False, "score": 0.02}
            matches = [
                {"identity_id": identity_id, "label": identity_id, "similarity": 0.92}
                for identity_id in self.people
            ]
            return self._response(200, {"face": face, "matches": matches})
        raise AssertionError(f"Unexpected upstream POST: {path}")

    async def delete(self, path, *, params=None):
        identity = path.rsplit("/", 1)[-1]
        watchlist = (params or {}).get("watchlist_name")
        if self.people.get(identity) != watchlist:
            return self._response(404, {"detail": "missing"})
        del self.people[identity]
        return self._response(200, {"removed": identity})

    @staticmethod
    def _response(status, body):
        return httpx.Response(status, json=body, request=httpx.Request("POST", "http://engine.test"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOMETRIC_API_KEY", API_KEY)
    main.app.state.data_dir = Path(tmp_path)
    main.app.state.engine_lock = asyncio.Lock()
    old_db = getattr(main.app.state, "registry", None)
    if old_db is not None:
        try:
            old_db.close()
        except Exception:
            pass
    main.app.state.registry = main._init_registry(Path(tmp_path) / "registry.sqlite3")
    main.app.state.engine_client = FakeEngineClient()
    main.app.state.engine_status = {
        "detector_loaded": True,
        "recognizer_loaded": True,
        "passive_liveness_loaded": True,
        "active_liveness_loaded": True,
    }
    test_client = TestClient(main.app)
    yield test_client
    test_client.close()
    main.app.state.registry.close()


def auth():
    return {"Authorization": f"Bearer {API_KEY}"}


def make_collection(client):
    response = client.post("/v1/collections", headers=auth(), json={"id": "students", "name": "Students", "threshold": 0.4})
    assert response.status_code == 201, response.text


def test_health_requires_secret_and_reports_ready(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/health").status_code == 401
    response = client.get("/v1/health", headers=auth())
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["auth_enabled"] is True


def test_collection_create_read_and_conflict(client, tmp_path):
    make_collection(client)
    response = client.get("/v1/collections/students", headers=auth())
    assert response.status_code == 200
    assert response.json()["collection"]["id"] == "students"
    assert client.post("/v1/collections", headers=auth(), json={"id": "students"}).status_code == 409
    saved = json.loads((Path(tmp_path) / "collections" / "students.json").read_text())
    assert saved["collection"]["threshold"] == 0.4


def test_enrollment_and_search_match_legacy_provider_contract(client):
    make_collection(client)
    response = client.post(
        "/v1/collections/students/persons",
        headers=auth(),
        data={"id": "student-123", "external_id": "student-123", "review_mode": "strict"},
        files={"images": IMAGE},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["person"]["id"] == "student-123"
    assert len(data["faces"]) == 1 and data["rejected_images"] == []
    duplicate = client.post(
        "/v1/collections/students/persons",
        headers=auth(),
        data={"id": "student-123", "external_id": "student-123", "review_mode": "strict"},
        files={"images": IMAGE},
    )
    assert duplicate.status_code == 409

    result = client.post(
        "/v1/collections/students/search",
        headers=auth(),
        data={"limit": "5", "threshold": "0.4"},
        files={"image": IMAGE},
    )
    assert result.status_code == 200, result.text
    payload = result.json()
    assert payload["matches"][0]["person"]["external_id"] == "student-123"
    assert payload["searched_face"]["liveness"] == {"status": "ok", "is_live": True, "live_score": 0.97}


def test_liveness_unavailable_does_not_pass(client):
    make_collection(client)
    client.app.state.engine_client = FakeEngineClient(live=False)
    response = client.post(
        "/v1/collections/students/search",
        headers=auth(),
        data={"limit": "5", "threshold": "0.4"},
        files={"image": IMAGE},
    )
    assert response.status_code == 200
    liveness = response.json()["searched_face"]["liveness"]
    assert liveness["is_live"] is False
    assert liveness["status"] == "ok"  # explicit model result is present, but negative


def test_enrollment_rejects_photo_failure_before_adding_person(client):
    make_collection(client)
    client.app.state.engine_client = FakeEngineClient(live=False)
    response = client.post(
        "/v1/collections/students/persons",
        headers=auth(),
        data={"id": "student-124", "external_id": "student-124", "review_mode": "strict"},
        files={"images": IMAGE},
    )
    assert response.status_code == 422
    assert "student-124" not in client.app.state.engine_client.people


def test_detect_maps_five_point_landmarks_and_liveness(client):
    make_collection(client)
    response = client.post(
        "/v1/detect", headers=auth(), data={"max_faces": "2", "collection_id": "students"}, files={"image": IMAGE}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["faces"]) == 1
    assert data["faces"][0]["landmarks"] == LANDMARKS
    assert data["faces"][0]["liveness"]["status"] == "ok"


def test_delete_is_idempotent(client):
    make_collection(client)
    enrolled = client.post(
        "/v1/collections/students/persons", headers=auth(),
        data={"id": "student-125", "external_id": "student-125", "review_mode": "strict"},
        files={"images": IMAGE},
    )
    assert enrolled.status_code == 200
    url = "/v1/collections/students/persons/student-125"
    assert client.delete(url, headers=auth()).json()["deleted"] is True
    second = client.delete(url, headers=auth())
    assert second.status_code == 200
    assert second.json()["deleted"] is True


def test_bad_api_key_is_rejected_without_leaking_details(client):
    response = client.get("/v1/collections", headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"
    assert "test-only-secret" not in response.text


def test_liveness_payload_fails_closed_when_score_is_missing():
    assert main._liveness_payload({"liveness": {"is_live": True}}) == {
        "status": "unavailable", "is_live": True, "live_score": None
    }
    assert main._liveness_payload({}) == {
        "status": "unavailable", "is_live": False, "live_score": None
    }


def test_detect_requires_a_bootstrapped_collection(client):
    response = client.post(
        "/v1/detect", headers=auth(), data={"max_faces": "2", "collection_id": "unknown"}, files={"image": IMAGE}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "REQUEST_REJECTED"
