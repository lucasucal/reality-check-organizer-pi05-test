"""Self-test: boots PolicyServer in-process on an ephemeral port and asserts
the contract v0 behaviors (§6.1): immediate /healthz, /readyz flip after load,
/act echo rule, malformed-request rejection. Prints PASS lines; exits 0."""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

# Package root is starter/rc_sdk (the installable project). Works from starter/
# without pip-installing, and does not import the SmolVLA policy.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rc_sdk"))

from rc_sdk.policy import Policy  # noqa: E402
from rc_sdk.server import PolicyServer  # noqa: E402


class FakePolicy(Policy):
    def load(self) -> None:
        time.sleep(0.3)  # simulate weight load

    def act(self, obs):
        return {"type": "joint_delta", "values": [0.0] * 6, "gripper": 0.0}


class FailedLoadPolicy(Policy):
    def load(self) -> None:
        raise RuntimeError("sensitive diagnostic detail")

    def act(self, obs):
        raise AssertionError("act must not run after failed load")


def request(port: int, method: str, path: str, body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"} if body else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read()), dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}"), dict(err.headers)


def main() -> None:
    manifest_path = os.path.join(os.path.dirname(__file__), "..", "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as source:
            manifest = json.load(source)
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        if manifest.get("contract") != "v0":
            raise ValueError('contract must be "v0"')
        if type(manifest.get("port")) is not int or not 1 <= manifest["port"] <= 65535:
            raise ValueError("port must be an integer from 1 to 65535")
        weights = manifest.get("weights")
        if weights is not None:
            if not isinstance(weights, dict):
                raise ValueError("weights must be an object when present")
            for field in ("repo_id", "revision"):
                value = weights.get(field)
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"weights.{field} must be a string when present")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"FAIL manifest.json: {exc}") from exc
    print("PASS manifest.json is valid (model link is supplied in the dashboard; weights not downloaded)")
    server = PolicyServer(0, FakePolicy())
    port = server.server_address[1]
    t0 = time.monotonic()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    server.start_load()

    status, body, _ = request(port, "GET", "/healthz")
    elapsed = time.monotonic() - t0
    assert status == 200 and body == {}, (status, body)
    assert elapsed < 0.2, f"healthz took {elapsed:.3f}s (liveness only!)"
    print(f"PASS /healthz 200 {{}} immediately (after {elapsed * 1000:.0f} ms, pre-weights)")

    status, body, _ = request(port, "GET", "/readyz")
    assert status == 200 and body["ready"] is False, (status, body)
    assert body["contract"] == "v0" and body["schema_version"] == 2
    assert body["capabilities"] == {"chunking": False, "caching": False}
    print("PASS /readyz not-ready before load, contract v0, schema_version 2, capabilities")

    time.sleep(0.5)
    status, body, _ = request(port, "GET", "/readyz")
    assert status == 200 and body["ready"] is True, (status, body)
    print("PASS /readyz flips ready after load() completes")

    obs = {"episode_id": "ep-123", "step": 7, "task_id": "t1",
           "images": {"wrist": "PLACEHOLDER", "left": "PLACEHOLDER", "right": "PLACEHOLDER"},
           "proprio": {"joint_pos": [0.0] * 6}}
    status, body, headers = request(port, "POST", "/act", obs)
    assert status == 200, (status, body)
    assert body["episode_id"] == "ep-123" and body["step"] == 7  # verbatim echo
    assert body["action"]["type"] == "joint_delta"
    assert headers.get("X-RC-Episode-Id") == "ep-123" and headers.get("X-RC-Step") == "7"
    print("PASS /act echoes episode_id/step verbatim (body + X-RC-* headers) + action")

    status, body, _ = request(port, "POST", "/act", {"episode_id": "ep-123"})
    assert status == 400 and "step" in body["error"], (status, body)
    print("PASS malformed /act (missing step) -> 400 with readable message")

    server.shutdown()

    failed_server = PolicyServer(0, FailedLoadPolicy())
    failed_port = failed_server.server_address[1]
    threading.Thread(target=failed_server.serve_forever, daemon=True).start()
    failed_server.start_load()
    for _ in range(100):
        if failed_server.load_error is not None:
            break
        time.sleep(0.01)
    status, body, _ = request(failed_port, "GET", "/readyz")
    assert status == 200 and body["ready"] is False, (status, body)
    assert body["error"] == "policy failed to load", body
    assert "sensitive diagnostic detail" not in json.dumps(body), body
    failed_server.shutdown()
    print("PASS terminal load failure stays not-ready and exposes only bounded error")

    print("ALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
