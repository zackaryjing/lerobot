# Scripted wood-pick data generation

This first implementation produces a clean baseline policy and successful
simulation demonstrations without depending on ROS or MoveIt.

## Design

```text
platform reset randomizer
  -> stable-section grasp sampler (stick centre ±3 cm)
  -> position-prioritized DLS IK state machine
  -> full-object-in-box success check
  -> successful-only HDF5 recorder
  -> LeRobot Dataset v3 conversion
```

The controlled Cartesian frame is the estimated midpoint between the two
fingers, not the SO-101 `gripper` link origin. This distinction matters for a
five-DOF arm: orientation is a soft objective, but the point that must cross the
wood remains the hard position objective.

### Random object reset

- samples yaw over the complete circle;
- computes the 8 x 1.15 x 1.15 cm stick's yaw-dependent footprint;
- keeps every corner at least 4 mm inside the 27 x 10 cm platform;
- rejects the visibly present but coarsely unreachable far side of the
  platform (object-centre radius currently limited to 38 cm);
- resets linear and angular velocity to zero.

This produces a uniform rejection sample over the intersection of the platform
and the conservative radial workspace. The radius is a calibration parameter,
not a claim about the exact SO-101 workspace.

### Grasp candidates

Candidates use only the two regular square cross-sections 3 cm to either side
of the stick centre. For each station they sample both asymmetric wrist flips
and approach tilt up to 20 degrees. Tilt is in the vertical/stick-axis plane,
so the two fingers stay at the same height and cannot put one fingertip through
the platform. The finger-closing axis remains transverse to the stick.

Isaac Lab 2.3's pose DLS weights all six task rows equally, despite SO-101
having only five arm joints. `controllers.py` supplies a compatible weighted IK
action term so position can be prioritized over the underactuated orientation.
The first bring-up currently uses zero orientation weight to separate workspace
and offset calibration from wrist-orientation tuning. Increase it only after a
single grasp pose reliably reaches the measured finger midpoint.

### Trajectory and failure feedback

`WoodPickStateMachine` executes minimum-jerk Cartesian segments:

1. settle and open;
2. move to pregrasp;
3. descend to the selected ±3 cm station;
4. close and verify that the stick rises;
5. lift to collision clearance;
6. move through a clear waypoint;
7. move above the box and align the stick with the box's long axis;
8. lower, release, retreat, and wait for settling.

Critical waypoints have measured Cartesian tracking tolerances. A pose that DLS
cannot reach is rejected as an IK failure. A grasp that does not lift the stick
is rejected before transport. Failed episodes are not exported.

The success predicate transforms all eight corners of the STL bounds. It
requires the complete stick to be between the cardboard walls and below the
rim, with low linear and angular speed. Checking only the centre would
incorrectly accept a stick balanced across a wall.

## Generate one dry-run episode

From the AutoDL project directory:

```bash
python simulation/leisaac_wood_pick/scripts/generate_scripted_data.py \
  --headless --enable_cameras --device cuda:0 \
  --num_demos 1 --max_attempts 10
```

Add `--realtime` only when visually observing the run. Offline generation is
otherwise allowed to run faster than wall-clock time.

## Record successful demonstrations

```bash
python simulation/leisaac_wood_pick/scripts/generate_scripted_data.py \
  --headless --enable_cameras --device cuda:0 \
  --record --num_demos 50 --max_attempts 500 \
  --dataset_file datasets/wood_pick_scripted.hdf5
```

The state machine internally consumes an 8D Cartesian IK command, but the
custom recorder deliberately writes the resulting six joint-position targets
as `actions`. Therefore the training label has the same dimension, order, and
meaning as the real SO-101 LeRobot dataset. Recording the internal 8D command
would not be deployable on the real robot. Existing files are never overwritten.

## Convert to LeRobot Dataset v3

LeIsaac 0.4.0's converter currently documents `lerobot==0.4.2` and
`numpy==1.26.0`; keep those conversion-only dependencies out of the Isaac
environment if they conflict with the project training environment.

```bash
python /root/autodl-tmp/leisaac-src/scripts/convert/isaaclab2lerobotv3.py \
  --task_name Joyand-SO101-WoodPick-v0 \
  --task_type so101leader \
  --hdf5_root /root/autodl-tmp/lerobot/datasets \
  --hdf5_files wood_pick_scripted.hdf5 \
  --repo_id joyand/wood_pick_sim \
  --fps 30 \
  --task_description "Pick up the wooden stick and place it in the box"
```

`--task_type so101leader` is intentional. It makes the converter declare a 6D
joint action schema and apply the same USD-radian to real-motor normalization
used by real SO-101 data. Using `so101_state_machine` would incorrectly declare
an 8D Cartesian action feature.

## Planner choice and upgrade path

The current state machine follows LeIsaac 0.4.0's native datagen design and
Isaac Lab's batched differential IK. It is the smallest path to a useful
baseline, but it is not a global collision-free planner.

- **cuRobo is the preferred phase-two planner.** It offers batched collision
  checking, many-seed IK, and minimum-jerk trajectory optimization inside
  Isaac Sim. It needs a validated SO-101 URDF, joint mapping, collision spheres,
  and a world model for the table/platform/box before it is trustworthy.
- **Lula/RMPflow is a reasonable alternative** for smooth reactive obstacle
  avoidance, but an unsupported robot still needs a Lula robot-description
  YAML/XRDF and tuning. It is a local reactive policy rather than a guarantee
  of finding a global path.
- **MoveIt 2 is not the first choice here.** It would add ROS 2, an Isaac bridge,
  SRDF/kinematics/controller configuration, and synchronization overhead just
  to generate data already inside Isaac. It becomes attractive later if the
  exact same planning stack must also execute on the real robot.

The state machine exposes explicit pose/phase boundaries, so replacing its
waypoint interpolation with cuRobo does not change randomization, success
checking, or the 6D recording format.

## Current test status

- CPU-only geometry tests pass for platform containment, ±3 cm grasp geometry,
  and complete-object box containment.
- LeIsaac 0.4.0 successfully instantiates the 8D IK action and the custom
  weighted action term.
- End-to-end testing is temporarily blocked by the concurrently edited robot
  material startup event attempting to bind a material to a USD instance
  proxy. That is a scene-material issue, not a scripted-policy interface issue.
