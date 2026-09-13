#!/usr/bin/env python
"""HTTPS/WebSocket server for Android inline orientation control of SO101.

Android WebXR `inline` orientation is only exposed in secure contexts, so the
server speaks HTTPS. TLS material (project-local CA + server leaf, see
tls_certs.py) is generated automatically on first start, and /ca.crt serves
the CA root so a phone can trust this address once, with no warnings.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import threading
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from lerobot.model.kinematics import RobotKinematics

from controller import (
    ALL_JOINTS,
    ARM_JOINTS,
    ControlConfig,
    OrientationController,
    load_calibration_limits,
    load_joint_limits,
)
from frame_adapter import VERSION as FRAME_ADAPTER_VERSION
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_URDF = REPO_ROOT / "SO101" / "so101_new_calib.urdf"
DEFAULT_CALIBRATION = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/myfollower01.json"
DEFAULT_CERT_DIR = THIS_DIR.parent / "phone_pose_viz"
DEFAULT_RESET_POSE = THIS_DIR / "reset_pose_myfollower01.json"
DEFAULT_ATLAS = THIS_DIR / "direction_atlas.json"
DEFAULT_COLLISIONS = REPO_ROOT / "SO101" / "collisions.json"
DEFAULT_MANUAL_SAMPLES = THIS_DIR / "manual_direction_samples.json"

log = logging.getLogger("phone_orientation_control")
controller: OrientationController | None = None
global_planner = None
clients: set[WebSocket] = set()
send_locks: dict[WebSocket, asyncio.Lock] = {}
active_phone_id: str | None = None
active_phone_ws: WebSocket | None = None
robot_urdf_path = DEFAULT_URDF
direction_atlas_path = DEFAULT_ATLAS
cert_dir: Path = DEFAULT_CERT_DIR


class SimPlanRequest(BaseModel):
    direction: list[float]
    execute: bool = True


class EscapeTriggerRequest(BaseModel):
    direction: list[float] | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(status_broadcaster())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def phone_page():
    return FileResponse(THIS_DIR / "phone.html", headers={"Cache-Control": "no-store"})


@app.get("/status")
async def status():
    if controller is None:
        return {"status": "controller not started"}
    return {**controller.status(), "frame_adapter_version": FRAME_ADAPTER_VERSION}


@app.get("/ca.crt")
async def local_ca_certificate():
    """Project-local CA root; install it once on the phone to trust this server."""
    path = cert_dir / "rootCA.crt"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="local CA certificate not found")
    return FileResponse(
        path,
        media_type="application/x-x509-ca-cert",
        headers={"Cache-Control": "no-store"},
        filename="rootCA.crt",
    )


@app.get("/viz")
async def visualization_page():
    return FileResponse(THIS_DIR / "viz.html", headers={"Cache-Control": "no-store"})


@app.get("/robot/so101.urdf")
async def robot_urdf():
    return FileResponse(robot_urdf_path, media_type="application/xml", headers={"Cache-Control": "no-store"})


@app.get("/robot/assets/{filename}")
async def robot_asset(filename: str):
    if Path(filename).name != filename or not filename.lower().endswith(".stl"):
        raise HTTPException(status_code=404, detail="invalid robot asset")
    path = robot_urdf_path.parent / "assets" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="robot asset not found")
    return FileResponse(path, media_type="model/stl", headers={"Cache-Control": "no-store"})


@app.get("/atlas")
async def direction_atlas():
    if not direction_atlas_path.is_file():
        raise HTTPException(status_code=404, detail="direction atlas has not been generated")
    return FileResponse(
        direction_atlas_path, media_type="application/json", headers={"Cache-Control": "no-store"}
    )


@app.post("/sim/plan")
async def plan_simulation(request: SimPlanRequest):
    if controller is None or global_planner is None:
        raise HTTPException(status_code=503, detail="dry-run global planner is not available")
    if not controller.global_path_execution_allowed():
        raise HTTPException(
            status_code=403,
            detail="global path playback is available in dry-run only",
        )
    try:
        target = np.asarray(request.direction, dtype=float)
        # Freeze an existing path before taking the planning start snapshot.
        controller.cancel_simulated_trajectory()
        current = controller.current_arm_joints()
        started = time.perf_counter()
        result = await asyncio.to_thread(global_planner.plan, current, target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=422, detail="no collision-free path was found for this direction")
    if request.execute:
        ok, message = controller.set_simulated_trajectory(result.waypoints_deg, target.tolist())
        if not ok:
            raise HTTPException(status_code=409, detail=message)
    payload = asdict(result)
    payload["planning_time_s"] = round(time.perf_counter() - started, 4)
    payload["executing"] = request.execute
    return payload


@app.post("/sim/cancel")
async def cancel_simulation():
    if controller is None:
        raise HTTPException(status_code=503, detail="controller is not available")
    ok, message = controller.cancel_simulated_trajectory()
    return {"ok": ok, "message": message}


@app.post("/escape/trigger")
async def trigger_escape(request: EscapeTriggerRequest):
    """Manually start a direction-preserving escape (digital-twin test hook)."""
    if controller is None:
        raise HTTPException(status_code=503, detail="controller is not available")
    ok, message = controller.trigger_escape(request.direction)
    if not ok:
        raise HTTPException(status_code=409, detail=message)
    return {"ok": True, "message": message}


@app.post("/escape/cancel")
async def cancel_escape():
    if controller is None:
        raise HTTPException(status_code=503, detail="controller is not available")
    ok, message = controller.cancel_escape()
    return {"ok": ok, "message": message}


@app.websocket("/viz-ws")
async def visualization_websocket(ws: WebSocket):
    """Read-only visualization stream; it never invalidates phone calibration."""
    await ws.accept()
    sent_escape_generation = -1
    try:
        while True:
            if controller is None:
                # Same defense in depth as /ws: never crash a viewer on a
                # missing controller; report and keep the socket alive.
                await ws.send_json({"type": "status", "status": "controller not started"})
                await asyncio.sleep(1.0)
                continue
            await ws.send_json(
                {"type": "status", "frame_adapter_version": FRAME_ADAPTER_VERSION, **controller.status()}
            )
            payload = controller.escape_path_payload()
            if payload is not None and payload[0] != sent_escape_generation:
                sent_escape_generation, escape_message = payload
                await ws.send_json(escape_message)
            await asyncio.sleep(1.0 / 60.0)
    except (WebSocketDisconnect, RuntimeError):
        pass


async def send_phone_json(ws: WebSocket, payload: dict) -> None:
    """Serialize writers and bound backpressure for Android WebSockets."""
    lock = send_locks.setdefault(ws, asyncio.Lock())
    async with lock:
        await asyncio.wait_for(ws.send_json(payload), timeout=5.0)


async def reply(ws: WebSocket, ok: bool, message: str) -> None:
    await send_phone_json(ws, {"type": "command_result", "ok": ok, "message": message})


async def command_reply(ws: WebSocket, command: str, result: tuple[bool, str]) -> None:
    log.info("phone command: %s ok=%s message=%s", command, result[0], result[1])
    await reply(ws, *result)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global active_phone_id, active_phone_ws
    await ws.accept()
    if controller is None:
        # Defense in depth: the app should only ever serve with a built
        # controller, but never crash a client connection if that fails.
        await ws.close(code=1013)
        return
    client_id = ws.query_params.get("client_id", "")
    resumed = bool(client_id and client_id == active_phone_id)
    if not resumed:
        controller.invalidate_calibration()
    active_phone_id = client_id
    active_phone_ws = ws
    clients.add(ws)
    send_locks[ws] = asyncio.Lock()
    controller.set_phone_transport_connected(True)
    log.info("phone connected: %s client=%s resumed=%s", ws.client, client_id[:12], resumed)
    try:
        await send_phone_json(ws, {"type": "status", **controller.status()})
        while True:
            msg = json.loads(await ws.receive_text())
            msg_type = msg.get("type")
            if msg_type == "orientation":
                seq = msg.get("seq")
                try:
                    seq = int(seq) if seq is not None else None
                except (TypeError, ValueError):
                    seq = None
                controller.submit_phone_orientation(msg["quat"], seq=seq)
            elif msg_type == "ping":
                await send_phone_json(
                    ws,
                    {"type": "pong", "client_time": msg.get("time"), "server_time": time.time()},
                )
            elif msg_type == "device":
                await command_reply(ws, "device", controller.set_input_device(str(msg.get("kind", ""))))
            elif msg_type == "calibrate":
                await command_reply(ws, "calibrate", controller.calibrate())
            elif msg_type == "sync":
                enabled = bool(msg.get("enabled"))
                await command_reply(ws, f"sync:{enabled}", controller.set_sync(enabled))
            elif msg_type == "mode":
                mode = str(msg.get("mode"))
                await command_reply(ws, f"mode:{mode}", controller.set_mode(mode))
            elif msg_type == "reset":
                await command_reply(ws, "reset", controller.request_reset())
    except WebSocketDisconnect as exc:
        log.warning(
            "phone disconnected: client=%s code=%s reason=%s",
            client_id[:12], exc.code, getattr(exc, "reason", ""),
        )
        if active_phone_ws is ws:
            controller.set_phone_transport_connected(False)
    except Exception as exc:
        log.exception("websocket error")
        if active_phone_ws is ws:
            controller.set_phone_transport_connected(False)
        try:
            await reply(ws, False, str(exc))
        except Exception:
            pass
    finally:
        clients.discard(ws)
        send_locks.pop(ws, None)
        if active_phone_ws is ws:
            active_phone_ws = None


async def status_broadcaster() -> None:
    while True:
        await asyncio.sleep(0.5)
        if not clients or controller is None:
            continue
        payload = {"type": "status", **controller.status()}
        sockets = list(clients)
        results = await asyncio.gather(
            *(send_phone_json(ws, payload) for ws in sockets),
            return_exceptions=True,
        )
        for ws, result in zip(sockets, results, strict=True):
            if isinstance(result, Exception):
                log.warning("phone status send failed: %r", result)
                clients.discard(ws)
                with suppress(Exception):
                    await ws.close(code=1011, reason="status send timeout")


def start_reload_watchdog(directory: Path) -> None:
    """Auto-restart this server when a .py file in the directory changes.

    Deliberately NOT uvicorn's reload: its child process is spawned without
    running main(), so the `controller` global stays None in the serving
    process. Re-executing the script instead guarantees main() runs again and
    rebuilds the controller.
    """
    snapshot = {p: p.stat().st_mtime for p in directory.rglob("*.py")}

    def watch() -> None:
        nonlocal snapshot
        while True:
            time.sleep(1.0)
            current = {p: p.stat().st_mtime for p in directory.rglob("*.py")}
            if current != snapshot:
                changed = sorted(
                    str(p.relative_to(directory))
                    for p in current
                    if current.get(p) != snapshot.get(p)
                )
                print(f"\n[reload] {changed} changed; restarting server\n", flush=True)
                os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=watch, name="reload-watchdog", daemon=True).start()


def local_ip() -> str:
    sock: socket.socket | None = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        if sock is not None:
            sock.close()


def make_controller(
    args: argparse.Namespace,
    state_validator=None,
    escape_planner=None,
    fallback_planner=None,
) -> OrientationController:
    config = ControlConfig()
    limits = load_joint_limits(args.urdf, args.calibration)
    hard_limits = load_calibration_limits(args.calibration)
    reset_data = json.loads(args.reset_pose.read_text())
    reset_joints = [float(reset_data["joints_deg"][name]) for name in ALL_JOINTS]
    kinematics = RobotKinematics(str(args.urdf), target_frame_name="gripper_frame_link", joint_names=ARM_JOINTS)
    robot = None
    if args.hardware:
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
        from lerobot.robots.so101_follower.so101_follower import SO101Follower

        robot_config = SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            use_degrees=True,
        )
        robot = SO101Follower(robot_config)
        try:
            robot.connect(calibrate=False)
        except Exception:
            # A failed handshake leaves the SDK port open inside this process.
            # Close it without attempting motor writes before propagating the error.
            if robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=False)
            raise
        if not robot.is_calibrated:
            robot.disconnect()
            raise RuntimeError("motor calibration does not match the calibration file; refusing to command hardware")
    instance = OrientationController(
        kinematics,
        limits,
        robot=robot,
        config=config,
        hard_joint_limits=hard_limits,
        reset_joints=reset_joints,
        state_validator=state_validator,
        escape_planner=escape_planner,
        fallback_planner=fallback_planner,
    )
    try:
        instance.start()
    except Exception:
        if robot is not None and robot.is_connected:
            robot.disconnect()
        raise
    return instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware", action="store_true", help="actually connect and command the arm; default is dry-run"
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="myfollower01")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=4445)
    parser.add_argument(
        "--cert-dir",
        type=Path,
        default=DEFAULT_CERT_DIR,
        help="directory for key.pem/cert.pem and the local CA root; generated on first start",
    )
    parser.add_argument("--reset-pose", type=Path, default=DEFAULT_RESET_POSE)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--collisions", type=Path, default=DEFAULT_COLLISIONS)
    parser.add_argument(
        "--manual-samples",
        type=Path,
        default=DEFAULT_MANUAL_SAMPLES,
        help="optional teleoperated direction samples; ignored when the file does not exist",
    )
    parser.add_argument(
        "--no-escape",
        action="store_true",
        help="disable the direction-preserving escape/reconfiguration planner",
    )
    parser.add_argument(
        "--body-model",
        action="store_true",
        help="forbid the box-composed pseudo-human zones behind the base (obstacles.py)",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="disable auto-reload on .py changes (default: reload on in dry-run)",
    )
    return parser.parse_args()


def main() -> None:
    global controller, direction_atlas_path, global_planner, robot_urdf_path, cert_dir
    args = parse_args()
    robot_urdf_path = args.urdf
    direction_atlas_path = args.atlas
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(THIS_DIR / "control.log")],
    )
    # WebXR needs a secure context, so HTTPS stays; make its certificates
    # painless instead: create the local CA + leaf covering every current
    # non-loopback address (campus LAN and Tailscale 100.x alike), and
    # refresh the leaf whenever that address set changes.
    from tls_certs import detect_ipv4_addresses, ensure_certificates

    addresses = detect_ipv4_addresses()
    cert_dir = args.cert_dir
    ensure_certificates(cert_dir, log=log)
    required_paths = [
        args.urdf,
        args.calibration,
        args.reset_pose,
        args.collisions,
    ]
    if not args.hardware or not args.no_escape:
        required_paths.append(args.atlas)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    from direction_atlas import SO101StateValidator

    # Forbidden zones (e.g. the wearer's body in the future back mounting) are
    # shared read-only across every validator instance so tracking, escape
    # planning, and global planning all agree on the same environment.
    obstacle_env = None
    if args.body_model:
        from obstacles import make_pseudo_human

        obstacle_env = make_pseudo_human()
        log.info("body model active: %s", obstacle_env.names())
    limits = load_joint_limits(args.urdf, args.calibration)
    tracking_validator = SO101StateValidator(
        args.urdf,
        args.collisions,
        limits,
        min_tip_height_m=-math.inf,
        min_moving_frame_height_m=-math.inf,
        obstacles=obstacle_env,
    )

    def validate_tracking_target(joints):
        return tracking_validator.evaluate(joints)[0] is not None

    # The escape planner owns this validator exclusively (its own placo stack),
    # so background planning never contends with /sim/plan or the control loop.
    # Constraint strictness matches tracking: no floor envelopes, because escape
    # is part of tracking and its waypoints are re-validated by the controller.
    from escape_planner import EscapePlanner

    escape_validator = SO101StateValidator(
        args.urdf,
        args.collisions,
        limits,
        min_tip_height_m=-math.inf,
        min_moving_frame_height_m=-math.inf,
        obstacles=obstacle_env,
    )
    escape_planner = None if args.no_escape else EscapePlanner(escape_validator)
    # Free-space fallback for escapes whose frozen direction is far from the
    # trapped pose (e.g. pointing backward past the shoulder_pan sweep): the
    # atlas + PlaCo refinement + RRT-Connect machinery, shared with the dry-run
    # global planner but on a tracking-strictness validator and its own placo
    # stack. It runs sequentially after the manifold planner in the same
    # worker thread, so there is no validator contention.
    fallback_planner = None
    if not args.no_escape:
        from global_planner import GlobalDirectionPlanner

        fallback_validator = SO101StateValidator(
            args.urdf,
            args.collisions,
            limits,
            min_tip_height_m=-math.inf,
            min_moving_frame_height_m=-math.inf,
            obstacles=obstacle_env,
        )
        fallback_planner = GlobalDirectionPlanner(
            args.atlas,
            fallback_validator,
            manual_samples_path=args.manual_samples,
        )
    controller = make_controller(
        args,
        state_validator=validate_tracking_target,
        escape_planner=escape_planner,
        fallback_planner=fallback_planner,
    )
    if args.hardware:
        global_planner = None
    else:
        from global_planner import GlobalDirectionPlanner

        planning_validator = SO101StateValidator(args.urdf, args.collisions, limits, obstacles=obstacle_env)

        global_planner = GlobalDirectionPlanner(
            args.atlas,
            planning_validator,
            manual_samples_path=args.manual_samples,
        )

    mode = "HARDWARE" if args.hardware else "DRY-RUN"
    reload = (not args.hardware) and not args.no_reload
    if reload:
        start_reload_watchdog(THIS_DIR)
    lines = [f"\nPhone orientation control [{mode}]"]
    lines += [
        f"\n  phone: https://{addr}:{args.server_port}"
        f"\n  digital twin: https://{addr}:{args.server_port}/viz"
        for addr in addresses
    ]
    lines.append(
        f"\n  phone trust (once): open any of the .../ca.crt URLs above on the"
        f"\n    Android phone, install rootCA.crt as a CA certificate, and that"
        f"\n    https address (LAN or Tailscale 100.x) is trusted without warnings"
    )
    lines.append(
        f"\n  frame_adapter={FRAME_ADAPTER_VERSION} reload={reload}\n"
    )
    print("".join(lines))
    try:
        uvicorn.run(
            # Auto-reload is handled by start_reload_watchdog (execv) in
            # main(); uvicorn's own reload would serve in a child process
            # that never ran main(), leaving the controller global None.
            app,
            host=args.host,
            port=args.server_port,
            ssl_keyfile=str(args.cert_dir / "key.pem"),
            ssl_certfile=str(args.cert_dir / "cert.pem"),
            log_level="info",
            ws_ping_interval=10.0,
            ws_ping_timeout=30.0,
            ws_max_queue=128,
            ws_per_message_deflate=False,
        )
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
