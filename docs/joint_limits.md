# Joint limits: preallocated control and active-limit markers

`JointLimitsTask` keeps the existing scalar repulsion and hysteresis. Lower limits enter
when `q < lower + alpha` and release when `q >= lower + delta - eps`. Upper limits enter
when `q > upper - alpha` and release when `q <= upper - delta + eps`. Lower-limit repulsion
takes precedence if both latches are active. Requested speeds remain
`gain_scalar * (lower + delta - q)` or `-gain_scalar * (q - upper + delta)`.

## Output storage and control path

The task implements `prepare_computation` and `update_into`. `TaskManager` prepares its
outputs once, during configuration. There is one fixed row per articulation, following
`WholeBodyModel::all_joint_names()`. For a nonzero repulsion command, row `i` has one
coefficient `1` at column `base_dofs + i`, with the scalar command in both error and desired
velocity. Every other coefficient is zero. Each update clears all rows and vectors before
writing new commands, so a released joint cannot retain a previous restriction.

The whole task is active only if at least one command is nonzero. A zero gain therefore
leaves all control rows inactive, while the activation latches still evolve normally.
Inactive rows neither constrain lower tasks nor change the solver's nullspace. Output
dimensions now remain fixed even when no limits are active; consumers that previously
expected a compact list of active rows must use the model's joint ordering instead.

Configured joint count, DoF count, activation latches, output buffers and status-message
capacity are prepared before control starts. `update_into` requires a previously prepared
output and never resizes it. The temporary active-limit vector is removed. The by-value
`update` API remains a compatibility adapter and still allocates its return buffers.

Disabling the task returns zero output with `disabled`; empty joint positions return zero
output with `waiting_for_joints`. A nonempty position vector with an incompatible size
returns zero output with `invalid_joint_state`, without resizing or introducing a new arm
stop policy. Existing activation hysteresis is retained across those interruptions.

## Optional visualization

```yaml
tasks.joint_limits.publish_limits_marker: true
tasks.joint_limits.limits_marker_publish_rate: 1.0
```

Publication defaults to `false`; the dual-arm vehicle YAML enables it. Both example YAMLs
leave it disabled. The rate defaults to 1 Hz and must always be finite and positive.
`publish_limits_marker` can be toggled during execution. Changing the rate requires reloading
the controller, because the shared observer timer is configured before control starts.
Both parameters are declared and validated in ros2_control and the standalone runtime.

Add an RViz MarkerArray display for `~/<task_id>/limits_markers`. For the deployed controller,
the joint-limit topic is normally:

```text
/cirtesub/alpha/controller/task_priority_controller/joint_limits/limits_markers
```

Each visible articulation has one opaque magenta cylinder, 6 cm in diameter and 7 cm long,
without a text label. The cylinder axis follows the articulation's rotation or translation
axis, expressed in the model base frame, rather than assuming the child link's Z axis.
Its shape and color distinguish joint limits from the self-collision capsules.
The cylinder is visible whenever either activation latch is set, including the recovery band
and zero gain. It disappears when both latches release. Its position is the origin of the
joint's child link, expressed in the configured model base frame. For prismatic joints,
the cylinder follows the translated child link. Model base translation and rotation are
removed when converting the pose; no robot-specific frames are hardcoded.

The marker namespace is `<task_id>/limits/<joint_name>` with a stable ID equal to the joint
index. Arrays can contain `DELETE` entries for inactive or invalid joints to remove their
previous cylinders. No markers are added when all limits are inactive. Clearing uses individual
`DELETE` actions, never `DELETEALL`, so unrelated displays are unaffected.

The optional backend query `joint_child_frame(joint_name, frame_out) const` resolves the
joint-to-link mapping during configuration. Its default implementation returns false and
clears the output string. KDL resolves it from the loaded tree, with exact joint/link names.
Missing mappings are reported during configuration when visualization is enabled; they
disable only the corresponding marker. Missing/nonfinite runtime poses or unavailable joint axes delete the affected
markers while limit control continues. An unavailable base pose suppresses all joint markers.

## Execution and lifecycle

Three snapshots, joint positions/axes/flags and two frame-state scratch buffers are preallocated.
With visualization disabled there are no marker pose reads. When enabled, only the base and
active joints with known frames are read from the existing backend cache, into sized buffers.
The axis comes from the joint's cached Jacobian column (angular for revolute joints, linear
for prismatic joints), and is normalized after conversion to the base frame. Cylinder
quaternions are constructed in the observer thread; this adds no frame reads or topology queries.
There are no TF waits, topology queries, logging or ROS messages in the control update.
`observe_command` stamps and exchanges the prepared snapshot after command finalization.

`CapsuleObserver` handles both collision and joint-limit tasks using one dedicated executor,
thread and timer per host. It also works when no collision task is configured. A lock-free
three-slot `SnapshotBuffer` transfers data; a slow observer can skip old snapshots without
blocking control. Message construction, allocation and DDS publication happen only on the
observer side. The observer uses snapshots instead of reading activation vectors directly.

Normal marker refreshes respect the configured rate, using reliable, volatile QoS. Once a
previously visible task has no valid active markers, clearing occurs at the next observer
tick; clearing can occur sooner than the normal refresh interval. Disabling publication,
disabling the task, or lifecycle cleanup deletes its owned markers. Publication toggles use
an atomic visibility generation so reenabling cannot display a snapshot from before the
toggle; a fresh control snapshot is required.

The usual `update_into` and snapshot transfer do not allocate memory or acquire locks with
KDL. This does not make the whole controller hard real time: KDL cache refresh, other tasks,
the hierarchy solver and controller diagnostics retain their existing costs. Fixed zero rows
also increase solver matrix sizes relative to the old compact output. The solver is unchanged.

## Verification

Tests cover both hysteresis directions, epsilon release, lower-limit precedence, fixed rows,
base-column offsets, lower-task motion after release/disablement, empty/incompatible states,
stable matrix/vector/string storage over repeated updates, and zero gain. Marker tests cover
poses and base transformations, prismatic joint mapping, configuration-only topology reads,
disabled visualization without pose reads, invalid/missing frames, parameter defaults and
validation, cylinder axes/dimensions/identities/colors/deletion, observer refresh rate and lifecycle cleanup.

After building from the workspace root, run:

```bash
colcon test --packages-select task_priority_kinematic_control
colcon test-result --verbose
```
