# Android phone orientation → SO101

This experiment intentionally ignores phone position. Android WebXR `inline`
orientation controls only the visual direction from the closed gripper toward
its tip. End-effector position and roll around this tip axis are not IK targets.
The explicit phone/gripper semantic-axis mapping lives in `frame_adapter.py`.

## Shared commanded-state model

- Dry-run is the default. Hardware requires an explicit `--hardware` flag.
- Calibration is mandatory before synchronization.
- Every new browser/WebSocket session invalidates the previous calibration.
- Calibration maps the phone's top edge (`+Y`) to the gripper-frame visual tip
  axis (`+Z`). A flat phone pointing robot-base forward corresponds to the arm's
  all-motors-at-calibration-midpoint reference direction.
- A stale phone pose immediately stops producing new actions while synchronization
  stays armed. Fresh poses from the same page resume automatically. A new/reloaded
  page still invalidates calibration.
- Quaternion jumps larger than 35 degrees per frame are rejected.
- Phone orientation and real-time targets use quaternion low-pass filtering.
- Dry-run and hardware initialize the planner from the same joint values in
  `reset_pose_myfollower01.json`. After that both advance from the previous
  commanded state, so identical phone frames produce identical joint actions.
- Filtering and direction interpolation use a fixed 30 Hz logical clock rather
  than serial-read wall-clock timing. A slow hardware loop changes playback
  speed, but not the action generated for a given control frame.
- Hardware observations are telemetry only. They do not feed back into the
  application-level planner. `/viz` displays the shared planned state and lists
  the measured state separately.
- Phone synchronization uses the same local differential IK, exact calibration
  limits, and URDF self-collision validator in dry-run and hardware. It does not
  invoke the atlas/global planner or automatically unfold to another pose.
- End-effector position and any sphere radius are deliberately unconstrained.
- The complete IK result is passed to LeRobot through `robot.send_action()`.
  This experiment does not implement a joint velocity/acceleration controller.
- Every phone-following target and the sampled edge from the previous command
  are checked against the exact ranges in the LeRobot calibration file (no
  implicit 3-degree margin) and the self-collision geometry in
  `SO101/collisions.json`. The calibrated gripper command remains in `0..100`.
  Floor height is not a phone-following constraint.
- The global planner used by the digital twin validates non-adjacent link
  collisions from `SO101/collisions.json`, joint limits, the gripper floor
  envelope, and moving-link frame heights. Direct parent/child contacts are
  intentionally excluded because their meshes overlap at normal joints.
- The configured reset pose is itself calibration-valid and self-collision-free.
  Reset and hardware startup follow validated joint-space waypoints to it.

## Stall escape (direction-preserving reconfiguration)

Greedy local tracking can saturate joints against calibration limits: two
directions may be reachable, but only through a joint-space detour that keeps
the tip pointed at the target. When that happens the controller now plans an
escape instead of holding forever:

- **Trigger**: after `trap_tick_threshold` (default 15, i.e. 0.5 s) consecutive
  ticks where the local IK step is rejected or makes no error progress while the
  phone target stays stable, the current target direction freezes and a planner
  worker starts. Fast deliberate phone motions reset the counter instead.
- **Pre-planning**: as soon as a few stall ticks accumulate (default 2), a
  background worker plans a free-space path from the current pose to the
  current target with the fallback planner. When the trap then fires and the
  frozen direction still matches (within `preplan_reuse_rad`, default 5°), the
  cached plan is adopted immediately — the trap-to-execution gap becomes the
  stall threshold plus playback, with no planning pause. A moved target
  invalidates the cache at completion and the worker replans at trap time.
  `preplan_enabled`/`preplan_min_interval_s`/`preplan_timeout_s` tune it.
- **Planner** (two tiers, run sequentially in the escape worker):
  - `escape_planner.py`: constrained sampling on the direction
    manifold {q valid : tip_dir(q) within a 3° cone of the target}. Random seeds
    are projected onto the manifold with the same bounded direction-IK step the
    tracker uses; goals are scored by joint margin and task manipulability; a
    CBiRRT-Connect-style search connects the start to the best goal with every
    edge sampled for limits, self-collision, and cone membership. This tier only
    converges when the trapped pose sits near the frozen direction's manifold.
  - Free-space fallback (`global_planner.py`): when the manifold plan fails —
    e.g. a target far past the shoulder_pan sweep, like pointing backward — the
    worker replans with the atlas + PlaCo refinement + RRT-Connect machinery on
    a tracking-strictness validator (no floor envelopes). Sparse atlas regions
    are backfilled by refining random seeds with the same PlaCo pass. The tip
    direction may deviate from the target mid-path; playback runs under the same
    drift-abort guard, so this is meant for pauses, when the phone is stable.
    Adopted plans report `global:direct` / `global:rrt_connect` as their method.
  Paths from either tier are resampled so no command exceeds
  `escape_waypoint_delta_deg` (2°) per tick.
- **Execution**: playback runs *inside* synchronization. Phone updates keep
  streaming and filtering; if the live target moves more than 12° away from the
  frozen direction mid-flight, the escape aborts at waypoint granularity and
  normal tracking resumes toward the new direction.
- **Guardrails**: planned waypoints are re-validated by the controller's exact
  runtime checks before execution; failures arm a cooldown (6 s) during which
  automatic re-triggering is blocked; three escapes toward the same direction
  within one minute latch the auto-trigger until the operator intervenes or
  asks for a different direction; calibration invalidation, reset, and sync-off
  cancel any active escape.
- **Digital twin**: `/viz` shows the planned path in orange with per-waypoint
  markers, plus live escape state/progress. The 触发逃逸 button requests a manual
  escape toward the selected preset direction (manual requests bypass the
  cooldown but still require calibration and sync); 取消逃逸 cancels. The same
  endpoints are available as `POST /escape/trigger {direction|null}` and
  `POST /escape/cancel`. Pass `--no-escape` to disable the feature entirely.
- Constraint strictness matches tracking (no floor envelopes); the old atlas /
  global planner remains dry-run only and unchanged.

## Forbidden zones (future wearable mounting)

The arm's final mounting is on a person's back with the base plate parallel to
the back (the current +X forward direction becomes up). `obstacles.py` provides
the interface for body collision avoidance:

- `ObstacleEnvironment`: forbidden zones in **robot-base coordinates**, with
  `add_box(name, center, half_extents, rotation)` for fast primitives and
  `add_mesh(name, path, transform)` for OBJ/STL models (hppfcl BVH meshes, the
  hook for real 3D body models). Zones carry a clearance `margin_m` (default
  2 cm). `apply_transform(R, t)` re-poses an entire environment, which is how
  the same human model definition is re-mounted when the base orientation
  changes.
- The moving links (shoulder → gripper tip) are approximated as a chain of
  spheres along the joint-origin polyline; every validator instance that
  carries the environment rejects configurations whose proxy comes within the
  margin of any zone (`obstacle_collision`). The base plate itself is excluded:
  it is the attachment surface and is expected to touch the wearer.
- Tracking IK, escape planning, the free-space fallback, and the dry-run
  global planner all share one read-only environment, so every adopted path
  is body-aware end to end. `make_pseudo_human()` builds a box-composed
  stand-in (torso behind the base, shoulder boxes, head offset to +Y) for
  testing; pass `--body-model` to the server to enable it.

## Run without hardware

```bash
PYTHONPATH=src /home/jing/miniconda3/envs/lerobot/bin/python \
  experiments/phone_orientation_control/server.py
```

Android WebXR `inline` orientation is only exposed in secure contexts, so the
server speaks HTTPS. To avoid both an unusable server (missing certificate)
and a phone that flags a self-signed cert as a high-risk site, the first start
generates a project-local CA and a server leaf certificate under
`experiments/phone_pose_viz/` (gitignored; `tls_certs.py` regenerates them).
Trusting this CA once on the phone makes the printed address permanently valid:

1. Open the printed `https://<lan-ip>:4445/ca.crt` address on the phone and
   download `rootCA.crt`.
2. Android: Settings > Security > Encryption & credentials > Install a
   certificate > CA certificate, then pick `rootCA.crt` from Downloads
   (the screen lock PIN is required).
3. Now open the printed main HTTPS address: start the inline sensor, calibrate,
   then enable synchronization. The page clearly shows `DRY-RUN`.

When the machine's LAN IP changes, the server (or `tls_certs.py`) renews the
leaf certificate for the new IP on the next start; the CA root already
installed on the phone keeps working unchanged.

Open the printed `/viz` address on the laptop for the digital twin. It loads the
actual SO101 URDF/STL files and applies the dry-run controller's joint values at
30 Hz. Green is the achieved visual gripper-tip direction; cyan is the phone
target direction. Desktop layout is side-by-side and mobile layout is stacked.

The digital-twin controls can also select front/back/up/down/left/right (or the
live phone direction), preview nearby direction-atlas samples, and run **global
planning and playback**. The planner:

1. selects several joint-space branches from the automatically sampled atlas;
2. refines each branch with a PlaCo axis-alignment task, leaving position and
   roll around the visible tip direction unconstrained;
3. validates multiple endpoints; and
4. uses a checked direct edge or bounded RRT-Connect when changing branches
   requires a detour.

Rebuild the automatic atlas after changing limits, collision pairs, or the URDF:

```bash
PYTHONPATH=src /home/jing/miniconda3/envs/lerobot/bin/python \
  experiments/phone_orientation_control/direction_atlas.py
```

Transient WebSocket reconnects from the same still-open page preserve calibration
and resume the user's previous synchronization intent after fresh XR poses arrive.
Reloading or reopening the page creates a new session and still requires calibration,
because Android WebXR may have created a different local reference space.
Phone poses are capped at about 30 Hz, application heartbeats run every 3 seconds,
the browser replaces a silent socket after 12 seconds, and server-side WebSocket
writes are serialized so status updates and heartbeat replies cannot overlap.
If Android supplies a discontinuous reference frame on the first pose after a
reconnect, the server rebases the phone-to-robot mapping to keep the robot target
continuous instead of rejecting every later pose against the stale frame.

## Hardware run

The reset button follows validated joint-space waypoints to the values in
`reset_pose_myfollower01.json` when hardware is connected.

```bash
PYTHONPATH=src /home/jing/miniconda3/envs/lerobot/bin/python \
  experiments/phone_orientation_control/server.py \
  --hardware --port /dev/ttyACM0 --robot-id myfollower01
```

The measured startup pose is allowed to be outside the calibration range or in
collision so the process can still start. Since an invalid physical start cannot
mathematically be part of an all-valid path, recovery begins at the first
calibration-valid, collision-free sample on the straight route to reset; every
command actually sent by this experiment is valid. After recovery, hardware and
dry-run use the same generated targets. Motor dynamics, load, missed packets, and
servo lag can still make measured motion differ from the planned twin.

The atlas/global planner remains available from `/viz` in dry-run. The server
rejects attempts to play those paths when `--hardware` is active.

## Optional manual direction samples

Automatic sampling is the primary source of global IK seeds. To compare it with
known-good human configurations or fill a sparse region later, connect the
SO101 leader and follower and run:

```bash
PYTHONPATH=src /home/jing/miniconda3/envs/lerobot/bin/python \
  experiments/phone_orientation_control/record_direction_samples.py \
  --follower-port /dev/ttyACM0 --follower-id myfollower01 \
  --leader-port /dev/ttyACM1 --leader-id myleader01
```

The recorder teleoperates at 30 Hz with a 2-degree follower command clamp.
Press Enter to capture the current follower configuration, `u` then Enter to
undo, and `q` then Enter to stop. Captures that violate the same collision,
joint-limit, or floor rules as the automatic atlas are rejected. The file is
saved atomically after every capture. If the default
`manual_direction_samples.json` exists, the dry-run server automatically adds
its samples to the atlas seeds; pass `--manual-samples PATH` to use another file.
