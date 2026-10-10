# Manual capsule and finite-plane collision avoidance

The existing `SelfCollisionAvoidanceTask` now uses manually specified capsule axes between
frame origins. The task never extracts geometry from URDF collision meshes. KDL still parses
the URDF kinematic tree, but its legacy `collision_capsules()` accessor returns an empty vector.
External users of that accessor must migrate to explicit geometry. Pinocchio remains an
unimplemented backend and cannot initialize this task.

## Configuration

All keys live under `tasks.<task_id>.`, using the existing ROS 2 parameter loader. See
`config/task_priority_controller_ros2_control.yaml` for the full working controller configuration,
`config/task_priority_kinematic_control.yaml` for the standalone runtime, and the vehicle's
`cirtesub_description/config/ros2_control_params_dual_alpha.yaml` for the deployed configuration.

`capsule_names` is an arbitrary nonempty string array. Each name contains only letters,
numbers and underscores, is unique, and identifies these parameters:

```yaml
tasks.self_collision_avoidance.capsule_names: [alpha_right_proximal, alpha_right_distal]
tasks.self_collision_avoidance.capsules.alpha_right_proximal.start_frame: cirtesub/alpha_right/m2_joint_link
tasks.self_collision_avoidance.capsules.alpha_right_proximal.end_frame: cirtesub/alpha_right/m2_1_2_link
tasks.self_collision_avoidance.capsules.alpha_right_proximal.radius: 0.04
tasks.self_collision_avoidance.capsules.alpha_right_distal.start_frame: cirtesub/alpha_right/m2_1_2_link
tasks.self_collision_avoidance.capsules.alpha_right_distal.end_frame: cirtesub/alpha_right/ee_base_link
tasks.self_collision_avoidance.capsules.alpha_right_distal.radius: 0.04
tasks.self_collision_avoidance.ignored_collision_pairs: ["alpha_right_proximal,alpha_right_distal"]
```

The supplied dual-arm configuration defines these two segments on each arm. All six frames
exist with their `cirtesub/` prefix; exact names are used without stripping prefixes or TF
resolution. Unknown or unevaluable frames cause initialization to fail. Radii are in metres,
positive and finite. **0.04 m is an initial testing/visualization radius, not a guarantee of
physical coverage.** No vehicle-body capsule is included in this initial configuration.

Each ignored pair is one string containing exactly two comma-separated capsule names,
without whitespace. Names must exist and differ. Repeated or reversed exclusions are
canonicalized. All unique capsule pairs are constructed once at initialization; no same-arm
or shared-frame pair is implicitly ignored. The initial four capsules produce six pairs,
with the two same-arm adjacent pairs explicitly excluded. The four remaining checks are:

- right proximal / left proximal;
- right proximal / left distal;
- right distal / left proximal;
- right distal / left distal.

## Configurable finite planes

The same task can additionally repel capsules from finite, one-sided rectangular planes.
`plane_names` defaults to an empty string array, preserving capsule-only behavior. Each
plane name follows the capsule naming rules and has its own reference frame and selection:

```yaml
tasks.self_collision_avoidance.plane_names: [body_guard]
tasks.self_collision_avoidance.planes.body_guard.reference_frame: cirtesub/base_link
tasks.self_collision_avoidance.planes.body_guard.normal_axis: x
tasks.self_collision_avoidance.planes.body_guard.position: 0.335
tasks.self_collision_avoidance.planes.body_guard.allowed_side: positive
tasks.self_collision_avoidance.planes.body_guard.bounds_min: [-0.265, -0.141]
tasks.self_collision_avoidance.planes.body_guard.bounds_max: [0.265, 0.459]
tasks.self_collision_avoidance.planes.body_guard.checked_capsules:
  - alpha_right_distal
  - alpha_left_distal
```

`reference_frame` is any frame evaluable with a finite pose and 6-by-model-DoFs Jacobian
by the configured kinematics backend. Omission or an empty string selects the model's
configured `base_frame`; no robot frame name is hardcoded in the implementation. The frame
is read from the existing kinematics cache, not resolved by waiting on an external TF
broadcaster. A plane follows both the translation and rotation of its reference frame.

`normal_axis` is `x`, `y`, or `z` in that frame. `position` is the plane's coordinate on
the normal axis, in metres. `allowed_side` is `positive` or `negative` along that local
axis. The two-element double arrays `bounds_min` and `bounds_max` specify the rectangle
limits on the tangential axes in ascending order: X-normal uses YZ, Y-normal uses XZ,
and Z-normal uses XY. These five geometry fields must be supplied explicitly; only the
reference frame and capsule selection have defaults. Each minimum must be strictly below its corresponding maximum,
and all coordinates and dimensions must be finite. Unknown fields, unknown frames,
duplicate names, and unknown/duplicate selected capsules are rejected.

Omitting `checked_capsules`, or providing an empty string array, selects all capsules.
Capsule-pair exclusions do not affect capsule/plane checks. The supplied configuration
checks only the distal capsules against `body_guard`: the proximal capsule's mounting
endpoint is within 2 cm of X=0.335 in every posture, so its radius intersects this plane
unavoidably. Including it would create a persistent, unsatisfiable repulsion request.
All positions, dimensions and selections can be changed in YAML and require reloading
the controller. An empty `plane_names` list disables plane checks; remove their definition
fields as well to satisfy configuration validation.

The example rectangle is 0.53 m wide by 0.60 m tall, centred at Y=0, Z=0.159, with local
X=0.335. Its marker shows exactly those bounds. For control, each capsule segment is
clipped to the tangential rectangle expanded on each side by its radius. Segments that
do not intersect that expanded region produce an inactive, zero row. Of the remaining
segment, the point with minimum allowed-side normal coordinate is selected. Its clearance
is `allowed_sign * (point[normal_axis] - position) - radius`. This is a conservative
normal barrier at edges/corners, **not Euclidean capsule-to-rectangle distance**. There
is no extra tangential expansion by `activation_distance`, so entering or leaving the
expanded footprint can switch the row abruptly; the barrier is deliberately finite.

Negative clearance continues to request motion towards the allowed side after crossing
the plane. Its known normal never triggers `degenerate_axis_distance`. The Jacobian
includes the selected clipping boundary's derivative and reference-frame angular and
linear motion. Common rigid motion of the whole robot cannot change the distance. At
ties or boundary switches the derivative is nonsmooth and a deterministic branch is used.
The same gains, thresholds, velocity limit, hierarchy and command-deficit metrics apply
to capsule and plane checks; the existing solver still provides no hard barrier guarantee.

Rows for capsule/plane pairs follow the original capsule-pair rows, ordered first by
`plane_names` and then by the plane's selected capsule list (or `capsule_names` when all
are selected). Diagnostics add `planes` and `checked_plane_pairs` counts and identify a
plane pair as `<capsule_name>,plane:<plane_name>`.

## Shared tuning parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `activation_distance` | 0.12 m | Repulsion begins below this surface clearance. |
| `safe_distance` | 0.05 m | Desired minimum separation, reported as `max(0, safe_distance-clearance)`. |
| `gain_scalar` | 1.0 s⁻¹ | Separation speed gain. |
| `max_repulsive_velocity` | 0.08 m/s | Maximum requested pair separation speed. |
| `parallel_epsilon` | 1e-8 | Dimensionless squared sine threshold for near-parallel segments. |
| `segment_length_epsilon` | 1e-9 m | Below this length a segment is treated as a point. |
| `degenerate_axis_distance` | 1e-6 m | Axis distance at/below which both arms stop. |
| `publish_capsules` | true | Enable RViz markers; can be toggled at runtime. |
| `capsule_publish_rate` | 1 Hz | Visualization refresh rate; positive when enabled. |

All numbers must be finite, `activation_distance >= safe_distance >= 0`, gains and maximum
speed nonnegative, numerical tolerances positive, and `parallel_epsilon < 1`. Geometry,
pair exclusions, tolerances, distances, maximum speed and publication frequency require a
controller reload. Scalar gains and task enablement use the existing tuning interfaces;
`publish_capsules` can be toggled without reloading. Enabling publication with a configured
nonpositive rate is rejected. Removed automatic-link filters have no effect and must not be
used in new configurations.

## Geometry and Jacobians

Each cycle reads unique endpoint frames and the base frame from the current KDL cache.
Their poses and translational Jacobians are transformed into the robot base reference:

```
p_base = R_base^T (p_world - p_base_world)
J_base_coordinates = R_base^T (J_frame_linear - J_base_linear + skew(r) J_base_angular)
```

Closest points come from a constrained analytical segment-distance calculation. For
nonparallel segments the valid interior stationary solution and all four clamped boundary
solutions are compared. Parallelism uses `(a*c-b*b)/(a*c)`, evaluated equivalently using the
squared cross product to reduce cancellation. At/below `parallel_epsilon`, the small
denominator is not divided by: boundary candidates are compared directly. Point/segment
and point/point cases are handled separately. Extended-precision arithmetic protects dot
products and squared distances; unrepresentable extreme distances saturate to finite double
limits. Near-parallel boundary handling is a numerical approximation at the chosen tolerance.
Ties select the first candidate deterministically, rather than claiming differentiability.

For a closest-point parameter `s`, the point and its fixed-parameter Jacobian are:

```
P = (1-s) A + s B
J_P = (1-s) J_A + s J_B
```

Both endpoints can move relative to each other. The segment is a kinematic approximation,
not necessarily a single rigid link. For a unique smooth closest pair with positive axis
distance, `n = (Q-P)/axis_distance` and `J_distance = n^T (J_Q-J_P)`; the envelope theorem
avoids differentiating `s` and `t`. Columns follow `WholeBodyModel`, including both arms and
the configured base DoFs. Base columns are zero for self-distance. The supplied system has
14 columns (six base and eight arm), but this size is not hardcoded.

Clearance is axis distance minus both radii: positive means separated, zero contact,
negative capsule overlap. A checked pair below `activation_distance` requests:

```
v = clamp(gain_scalar * (activation_distance - clearance), 0, max_repulsive_velocity)
```

There is one fixed output row per checked capsule or plane pair, in generated pair order. Inactive and
undefined rows are zero; `TaskComputation.active` is true when a valid repulsion row exists.
The original task-priority solver is unchanged. `safe_distance` is a metric, not a hard
barrier constraint. Competing rows, damping and final velocity limits can prevent satisfying
requested separation speeds; `max_velocity_deficit` reports the maximum positive requested
minus achieved separation rate using the final applied generalized command.

## Degenerate geometry and recovery

When any checked pair has axis distance **less than or equal to**
`degenerate_axis_distance`, its normal is not manufactured or reused. The task marks the
fault. After solving the hierarchy, both arm segments of the generalized command are set
to zero; base commands are retained. Commands are finalized before interface writes,
command-topic publication and controller-output diagnostics, in both execution modes.
Nonfinite/unevaluable runtime geometry or Jacobians also stop both arms. Disabling the task
explicitly disables this protection. Motion resumes automatically on the first valid control
cycle in which all checked pairs are above the threshold; no latch or rearm service is added.

A non-control observer reports the geometry error and affected pair at most once per second,
and reports recovery once. A low threshold detects undefined directions, **not all physical
contact**: surface overlap with distinct axes still uses the normal repulsive task.

## Visualization and execution costs

The marker topic is `~/self_collision_avoidance/capsules`, expanded using the host node name.
For the ros2_control controller this is normally:

```
/cirtesub/alpha/controller/task_priority_controller/self_collision_avoidance/capsules
```

Add an RViz MarkerArray display for this topic. Publication uses reliable, volatile QoS,
compatible with the default RViz subscription; any DDS wait is confined to the observer. Its messages use `cirtesub/base_link` and
snapshot timestamps. Each capsule has a cylinder (ID 0) and endpoint spheres (IDs 1 and 2),
in namespace `self_collision_avoidance/<capsule_name>`. All diameters equal twice the radius.
Right-arm markers are blue, left-arm markers green, and active/degenerate capsules red.
Planes are translucent amber rectangular triangle markers in namespace
`self_collision_avoidance/planes/<plane_name>` with ID 0. Both windings are emitted for
visibility from either side; active planes are red. Corners are transformed from each
reference frame into the same base frame and use the same snapshot timestamps. Invalid
geometry deletes the affected markers instead of displaying stale positions.
Zero-length capsules show one sphere and delete the other primitives. Publication disablement,
task disablement and lifecycle cleanup delete only this observer's own markers.

Initialization accesses the configured backend through a weak reference in `TaskContext`,
so tasks cannot retain a backend past the lifetime of its plugin loader.

Frame buffers, capsule poses, pair lists, Jacobian matrices and three snapshot slots are
allocated during configuration. An additive `update_into` API and preallocated manager
outputs retain source compatibility with existing plugins; older by-value plugins use a
compatibility adapter. KDL's additive `read_frame_state` copies into pre-sized buffers.

The control producer and visualization consumer each own one snapshot slot, exchanging the
third with a lock-free atomic index. A slow consumer cannot retain a producer-owned slot;
old snapshots can be dropped. **One dedicated executor/thread and timer per host handles
all collision tasks**, outside the control executor. DDS publication, marker construction,
logging and deletion happen only there or during non-control cleanup. There is no visualization
mutex shared with the control producer, no TF wait, file access, parameter parsing or pair
rebuilding in the capsule update. Even without RViz, the observer consumes diagnostics.

This avoids new allocation/blocking in the regular capsule calculation, but the existing
KDL cache refresh, other task adapters, hierarchy solver and general controller diagnostics
still allocate or publish. This refactor does not make the complete controller hard real time.
Parallel overlapping segments and closest-pair switches can be nonsmooth. Manual geometry,
reactive velocities and damping do **not** provide a formal collision-free or CBF guarantee.

## Verification

Focused tests cover the sixteen requested capsule distance cases, invalid inputs, pair validation,
finite-difference Jacobians, moving-base invariance, stop/recovery and marker geometry.
Plane tests add X/Y/Z normals, both allowed sides, radius-expanded finite bounds, clipping
gradients, contact/crossing recovery, selected/multiple planes, moving and rotated reference
frames, parameter declarations, runtime failure and rectangle publication/deletion.
Snapshot tests retain an old consumer slot through 100,000 publications and check concurrent
snapshot consistency. A ROS integration test verifies publication on the dedicated observer
executor and deletion when visualization is disabled. Build and run from the workspace root:

```bash
colcon build --packages-select task_priority_kinematic_control cirtesub_description
colcon test --packages-select task_priority_kinematic_control
colcon test-result --verbose
```
