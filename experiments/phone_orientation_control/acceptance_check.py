"""Digital-twin acceptance: fake phone -> calibrate -> sync -> trigger escape.

Exercises the full loop headlessly: WebSocket phone session, HTTP escape
trigger, controller state-machine transitions, and the /viz-ws escape_path
push consumed by the digital twin. Requires a running dry-run server:

    PYTHONPATH=../src python server.py
    python acceptance_check.py
"""

import asyncio
import json
import ssl
import sys
import urllib.request

import websockets

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
BASE = "https://localhost:4445"


def get(path):
    with urllib.request.urlopen(BASE + path, context=CTX, timeout=5) as r:
        return json.load(r)


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, context=CTX, timeout=30) as r:
        return json.load(r)


async def drain(ws):
    """Best-effort receive to keep buffers moving; returns parsed or None."""
    try:
        return json.loads(await asyncio.wait_for(ws.recv(), timeout=0.05))
    except (asyncio.TimeoutError, Exception):
        return None


async def main() -> int:
    failures = []

    async with websockets.connect("wss://localhost:4445/viz-ws", ssl=CTX) as viz_ws:
        async with websockets.connect("wss://localhost:4445/ws?client_id=acceptance", ssl=CTX) as phone:
            async def phone_stream():
                while True:
                    await phone.send(json.dumps({"type": "orientation", "quat": [0.0, 0.0, 0.0, 1.0]}))
                    await asyncio.sleep(1 / 30)

            stream = asyncio.create_task(phone_stream())
            await asyncio.sleep(0.3)
            await phone.send(json.dumps({"type": "calibrate"}))
            await asyncio.sleep(0.3)
            await phone.send(json.dumps({"type": "sync", "enabled": True}))
            await asyncio.sleep(0.5)

            status = get("/status")
            print("calibrated:", status["calibrated"], "| sync:", status["sync_enabled"],
                  "| escape_enabled:", status["escape_enabled"])
            if not (status["calibrated"] and status["sync_enabled"] and status["escape_enabled"]):
                failures.append("controller not ready")

            print("trigger:", post("/escape/trigger", {"direction": [1.0, 0.0, 0.0]}))
            seen = []
            deadline = asyncio.get_event_loop().time() + 90
            while asyncio.get_event_loop().time() < deadline:
                state = get("/status")["escape_state"]
                if not seen or seen[-1] != state:
                    seen.append(state)
                    print("state ->", state)
                if state == "idle":
                    break
                await asyncio.sleep(0.2)

            print("transitions:", " -> ".join(seen))
            final = get("/status")
            print("outcomes:", final["escape_outcome_counts"], "| reason:", final["escape_last_reason"])

            # The twin must have received the planned path.
            escape_path_seen = False
            viz_deadline = asyncio.get_event_loop().time() + 5
            while asyncio.get_event_loop().time() < viz_deadline:
                message = await drain(viz_ws)
                if message and message.get("type") == "escape_path":
                    escape_path_seen = True
                    print("viz-ws escape_path:", message["method"],
                          len(message["waypoints_deg"]), "waypoints")
                    break
            print("viz escape_path push:", "OK" if escape_path_seen else "MISSING")
            if not escape_path_seen:
                failures.append("no escape_path on viz-ws")
            if "executing" not in seen:
                # Reaching executing proves the planning phase ran and published;
                # fast plans can finish between two polls, so planning itself is
                # optional to observe.
                failures.append("executing state never observed")
            if "completed" not in seen:
                failures.append("completed state never observed")

            stream.cancel()
    print("ACCEPTANCE:", "FAIL: " + "; ".join(failures) if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
