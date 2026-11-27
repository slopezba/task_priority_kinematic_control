#include "task_priority_kinematic_control/core/stpik_check.hpp"
#include "task_priority_kinematic_control/tasks/pose_task_common.hpp"
#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effector_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/robot_nominal_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effectors_facing_task.hpp"
#include "sura_manipulation_actions/facing_messages.hpp"
#include "sura_manipulation_actions/whole_body_nominal.hpp"
#include "sura_manipulation_actions/cartesian_pose_path.hpp"
#include "sura_manipulation_actions/cartesian_tracking.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <sstream>
#include <iomanip>

namespace task_priority_kinematic_control
{
namespace
{
using Vector6 = Eigen::Matrix<double, 6, 1>;
Eigen::Matrix3d rotation(const WholeBodyState & state)
{
  return (Eigen::AngleAxisd(state.base_rpy.z(), Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(state.base_rpy.y(), Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(state.base_rpy.x(), Eigen::Vector3d::UnitX())).toRotationMatrix();
}
geometry_msgs::msg::Pose poseMessage(const Eigen::Isometry3d & pose)
{
  geometry_msgs::msg::Pose out;
  out.position.x = pose.translation().x(); out.position.y = pose.translation().y();
  out.position.z = pose.translation().z();
  const Eigen::Quaterniond q(pose.rotation());
  out.orientation.x = q.x(); out.orientation.y = q.y();
  out.orientation.z = q.z(); out.orientation.w = q.w(); return out;
}
Vector6 error(const Eigen::Isometry3d & desired, const Eigen::Isometry3d & measured,
  const Eigen::Matrix3d & reference, const Vector6 & activation = Vector6::Ones())
{
  return sura_manipulation_actions::cartesianTrackingError(desired, measured, reference, activation);
}
bool within(const Vector6 & value, const Vector6 & tolerance, double epsilon)
{return value.allFinite() && (value.cwiseAbs().array() <= tolerance.array() + epsilon).all();}
void recordState(StpikAction::Result & out, const WholeBodyState & state)
{
  if (state.joint_positions.size() > 0) {
    out.joint_positions.assign(state.joint_positions.data(),
      state.joint_positions.data() + state.joint_positions.size());
  } else {out.joint_positions.clear();}
  Eigen::Isometry3d base = Eigen::Isometry3d::Identity();
  base.translation() = state.base_position; base.linear() = rotation(state);
  out.base_pose = poseMessage(base);
}
// Purely kinematic integration of the active base DOF and both arms.
bool integrate(const WholeBodyModel & model, const Eigen::VectorXd & velocity, double step,
  WholeBodyState & state)
{
  Vector6 base_velocity = Vector6::Zero();
  const auto & axes_list = model.active_base_dofs();
  for (size_t i = 0; i < axes_list.size(); ++i) {base_velocity(axes_list[i]) = velocity(i);}
  state.base_position += step * base_velocity.head<3>();
  const Eigen::Vector3d increment = step * base_velocity.tail<3>();
  Eigen::Matrix3d next_rotation = rotation(state);
  if (increment.norm() > 0.0) {
    next_rotation = Eigen::AngleAxisd(increment.norm(), increment.normalized()).toRotationMatrix() * next_rotation;
  }
  // Recover ZYX Euler coordinates; angular commands themselves are spatial world rates.
  state.base_rpy.y() = std::asin(std::clamp(-next_rotation(2, 0), -1.0, 1.0));
  state.base_rpy.x() = std::atan2(next_rotation(2, 1), next_rotation(2, 2));
  state.base_rpy.z() = std::atan2(next_rotation(1, 0), next_rotation(0, 0));
  state.joint_velocities = velocity.tail(state.joint_positions.size());
  state.joint_positions += step * state.joint_velocities;
  return state.joint_positions.allFinite() && state.base_position.allFinite() && state.base_rpy.allFinite();
}

// Facing check: the primary EndEffectorsFacingTask follows the progress path while
// robot_nominal receives the secondary whole-body reference, as the facing executor does.
// Tracking rows are [left pointing(2), right pointing(2), roll, distance].
StpikAction::Result run_facing(
  StpikSnapshot & snapshot, const StpikAction::Goal & goal, const StpikOptions & options,
  const StpikCheck::Canceled & canceled, const StpikCheck::Progress & progress,
  const std::function<void(const WholeBodyState &, double)> & observe,
  const StpikCheck::ValidateTasks & validate_tasks, StpikAction::Result out)
{
  namespace facing = sura_manipulation_actions::facing;
  WholeBodyState state = snapshot.initial_state;
  const auto fail = [&](const std::string & code, const std::string & reason,
      uint8_t outcome = StpikAction::Result::FAIL) {
      out.outcome = outcome; out.code = code; out.message = reason;
      recordState(out, state); return out;
    };
  std::string reason;
  facing::PureAxis left_goal, right_goal;
  facing::Path path;
  if (!facing::fromMessage(goal.facing_path, left_goal, right_goal, path, reason))
  {return fail("INVALID_TRAJECTORY", reason);}
  sura_manipulation_actions::WholeBodyNominal nominal;
  if (!nominal.init(goal.nominal_joints, goal.nominal_base, reason)) {return fail("INVALID_NOMINAL", reason);}
  if (nominal.frame() != snapshot.model->world_frame()) {
    return fail("FRAME_MISMATCH", "Nominal trajectory must use the Task Priority world frame");
  }
  std::vector<Eigen::Index> nominal_columns;
  for (const auto & name : snapshot.model->all_joint_names()) {
    const auto found = std::find(nominal.joint_names().begin(), nominal.joint_names().end(), name);
    if (found == nominal.joint_names().end()) {
      return fail("INVALID_NOMINAL", "Nominal trajectory is missing joint " + name);
    }
    nominal_columns.push_back(static_cast<Eigen::Index>(found - nominal.joint_names().begin()));
  }
  std::vector<double> times;
  for (const auto & point : goal.nominal_joints.points) {
    times.push_back(point.time_from_start.sec + 1e-9 * point.time_from_start.nanosec);
    if (times.size() > 1 && times.back() <= times[times.size() - 2])
    {return fail("INVALID_TRAJECTORY", "Joint trajectory times must increase");}
  }
  const auto & values = goal.nominal_progress;
  if (times.size() < 2 || times.front() != 0.0 || values.size() != times.size())
  {return fail("INVALID_TRAJECTORY", "Facing needs a timed nominal trajectory with one progress per point");}
  for (size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i]) || values[i] < 0.0 || values[i] > 1.0 || (i && values[i] < values[i - 1]))
    {return fail("INVALID_TRAJECTORY", "nominal_progress must be nondecreasing in [0, 1]");}
  }
  Vector6 path_tolerance, goal_tolerance;
  for (size_t i = 0; i < 6; ++i) {
    if (!std::isfinite(goal.path_tolerance[i]) || goal.path_tolerance[i] < 0.0 ||
      !std::isfinite(goal.goal_tolerance[i]) || goal.goal_tolerance[i] < 0.0)
    {return fail("INVALID_TOLERANCES", "Six finite nonnegative path/goal tolerances are required");}
    path_tolerance(i) = goal.path_tolerance[i];
    goal_tolerance(i) = goal.goal_tolerance[i];
  }
  if (!std::isfinite(goal.goal_time_tolerance) || goal.goal_time_tolerance < 0.0)
  {return fail("INVALID_TOLERANCES", "Invalid goal convergence allowance");}
  const Eigen::Index dofs = snapshot.model->total_dofs();
  Eigen::VectorXd limits = snapshot.solver.velocity_limits();
  if (limits.size() != dofs || !limits.allFinite() || (limits.array() <= 0.0).any()) {
    return fail("INVALID_VELOCITY_LIMITS", "Runtime controller needs finite positive generalized velocity limits");
  }
  if (!options.velocity_limits.empty()) {
    if (options.velocity_limits.size() != static_cast<size_t>(dofs)) {
      return fail("INVALID_VELOCITY_LIMITS", "Override must match active base + left + right controller order");
    }
    limits = limits.cwiseMin(Eigen::Map<const Eigen::VectorXd>(options.velocity_limits.data(), dofs));
  }
  std::shared_ptr<EndEffectorsFacingTask> primary;
  size_t primary_index = 0;
  for (size_t i = 0; i < snapshot.tasks.size(); ++i) {
    if (snapshot.tasks[i]->id() == goal.task_id) {
      primary = std::dynamic_pointer_cast<EndEffectorsFacingTask>(snapshot.tasks[i]); primary_index = i;
    }
  }
  if (!primary || !primary->enabled())
  {return fail("FACING_TASK_UNAVAILABLE", "Selected task must be an enabled EndEffectorsFacingTask");}
  const auto secondary = find_robot_nominal_task(snapshot.tasks);
  if (!secondary || !secondary->enabled() || secondary->priority() <= primary->priority())
  {return fail("NOMINAL_TASK_UNAVAILABLE", "An enabled RobotNominalTask below the facing task is required");}
  auto simulated_time = std::make_shared<std::chrono::steady_clock::time_point>(snapshot.captured);
  // Both references expire on the simulated monotonic time.
  TaskContext context{snapshot.model, snapshot.backend, {}, [simulated_time]() {return *simulated_time;}};
  for (auto & task : snapshot.tasks) {StpikTaskAccess::bind(*task, context);}
  std::vector<TaskComputation> computations(snapshot.tasks.size());
  for (size_t i = 0; i < snapshot.tasks.size(); ++i) {snapshot.tasks[i]->prepare_computation(computations[i]);}
  const auto wall_start = std::chrono::steady_clock::now();
  const int64_t initial_ns = state.ros_time_valid ? state.ros_time_ns : 1000000000LL;
  const double duration = times.back();
  const double end = duration + goal.goal_time_tolerance;
  double time = 0.0;
  while (true) {
    out.time_from_start = time;
    if (canceled && canceled()) {return fail("CANCELED", "STPIK canceled", StpikAction::Result::CANCELED);}
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count() > options.wall_timeout) {
      return fail("CHECK_TIMEOUT", "Offline simulation exceeded wall-time budget; feasibility is unknown",
        StpikAction::Result::UNAVAILABLE);
    }
    state.ros_time_valid = true;
    state.ros_time_ns = initial_ns + static_cast<int64_t>(time * 1e9);
    *simulated_time = snapshot.captured + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(time));
    double rate = 0.0;
    const double s = facing::progressAtTime(times, values, time, &rate);
    if (!primary->set_facing_reference(
        facing::referenceMessage(left_goal, right_goal, path, s, rate), reason))
    {return fail("REFERENCE_REJECTED", reason);}
    sura_manipulation_actions::WholeBodyNominalSample sample;
    if (!nominal.sample(time, sample, reason)) {return fail("INVALID_NOMINAL", reason);}
    Eigen::VectorXd q(static_cast<Eigen::Index>(nominal_columns.size())), qd(q.size());
    for (Eigen::Index j = 0; j < q.size(); ++j) {
      q(j) = sample.joints(nominal_columns[static_cast<size_t>(j)]);
      qd(j) = sample.joint_velocities(nominal_columns[static_cast<size_t>(j)]);
    }
    if (!apply_robot_nominal_reference(snapshot.tasks, *secondary, sample.base, sample.base_rates, q, qd,
        reason, goal.nominal_activation))
    {return fail("REFERENCE_REJECTED", "robot_nominal rejected its reference: " + reason);}
    snapshot.backend->update(state);
    TaskManager::update_tasks_into(snapshot.tasks, computations, state, *snapshot.backend);
    if (!computations[primary_index].active)
    {return fail("FACING_TASK_INACTIVE", "Facing task cannot be evaluated: " + computations[primary_index].status_message);}
    const auto rows = facing::evaluate(
      snapshot.backend->get_frame_state(primary->left_frame()).pose,
      snapshot.backend->get_frame_state(primary->right_frame()).pose, {}, {},
      left_goal, right_goal, path.sample(s), primary->min_distance());
    if (!rows.valid || !rows.roll_valid)
    {return fail("FACING_UNDEFINED", "Tools too close or facing roll undefined");}
    const Vector6 tracking = rows.error;
    std::copy(tracking.data(), tracking.data() + 6, out.tracking_error.begin());
    for (const auto & task : snapshot.tasks) {
      if (!StpikTaskAccess::safety(*task, state, options.numerical_tolerance, out)) {
        return fail(out.code, out.message.empty() ?
          "Safety constraint violated: " + out.task_id + "/" + out.constraint : out.message);
      }
    }
    if (validate_tasks) {validate_tasks(computations);}
    if (observe) {observe(state, time);}
    if (!tracking.allFinite()) {return fail("NONFINITE_TRACKING", "Invalid facing tracking error");}
    if (time == 0.0 && !within(tracking, path_tolerance, options.numerical_tolerance)) {
      return fail("INITIAL_STATE_MISMATCH", "Facing path start does not match the controller snapshot");
    }
    HierarchySolver::Diagnostic diagnostic;
    auto command = snapshot.solver.solve(computations, &diagnostic, options.singular_value_tolerance, nullptr);
    if (!diagnostic.valid) {
      if (diagnostic.failing_task < snapshot.tasks.size()) {
        out.task_id = snapshot.tasks[diagnostic.failing_task]->id();
      }
      return fail("SOLVER_FAILURE", diagnostic.message);
    }
    const bool rank_deficient = diagnostic.rank_deficient[primary_index];
    out.minimum_singular_value = std::min(out.minimum_singular_value,
      diagnostic.minimum_singular_values[primary_index]);
    TaskManager::finalize_tasks_command(snapshot.tasks, computations, *snapshot.model,
      command, state.ros_time_ns);
    const auto & velocity = command.generalized_velocity;
    if (velocity.size() != dofs || !velocity.allFinite()) {
      return fail("NONFINITE_COMMAND", "Invalid final generalized velocity");
    }
    for (Eigen::Index i = 0; i < dofs; ++i) {
      out.maximum_velocity_ratio = std::max(out.maximum_velocity_ratio, std::abs(velocity(i)) / limits(i));
      if (std::abs(velocity(i)) > limits(i) + options.numerical_tolerance) {
        out.constraint = "generalized_velocity[" + std::to_string(i) + "]";
        return fail("VELOCITY_LIMIT", "Final command exceeds validation/controller velocity cap");
      }
    }
    if (time <= duration && !within(tracking, path_tolerance, options.numerical_tolerance)) {
      out.task_id = goal.task_id;
      return fail(rank_deficient ? "TRACKING_SINGULARITY" : "PATH_TRACKING",
        "Facing path tolerance exceeded");
    }
    if (time >= duration && within(tracking, goal_tolerance, options.numerical_tolerance)) {
      out.outcome = StpikAction::Result::SUCCESS; out.code = "SUCCESS";
      out.message = "Offline Task Priority simulation reached the facing goal within all constraints";
      recordState(out, state); return out;
    }
    if (time >= end) {
      out.task_id = goal.task_id;
      return fail(rank_deficient ? "GOAL_SINGULARITY" : "GOAL_TOLERANCE",
        "Facing goal not reached within the configured convergence allowance");
    }
    if (progress) {progress(time, end);}
    const double boundary = time < duration ? duration : end;
    const double next = boundary - (time + options.dt) < 1e-12 ? boundary : time + options.dt;
    const double step = next - time;
    if (!(step > 0.0)) {return fail("INVALID_TIME_STEP", "Simulation cannot advance time");}
    if (!integrate(*snapshot.model, velocity, step, state))
    {return fail("NONFINITE_STATE", "Integrated state is not finite");}
    time = next;
  }
}
}

bool StpikOptions::valid(std::string & reason) const
{
  for (double positive : {dt, stall_window, min_position_progress, min_orientation_progress,
      singular_value_tolerance, wall_timeout})
  {
    if (!std::isfinite(positive) || positive <= 0.0) {
      reason = "STPIK time steps, progress thresholds and timeout must be positive and finite"; return false;
    }
  }
  for (double nonnegative : {numerical_tolerance, snapshot_joint_tolerance,
      snapshot_position_tolerance, snapshot_orientation_tolerance,
      rejoin_position_tolerance, rejoin_orientation_tolerance})
  {
    if (!std::isfinite(nonnegative) || nonnegative < 0.0) {
      reason = "STPIK numerical, snapshot and rejoin tolerances must be finite and nonnegative"; return false;
    }
  }
  if (dt < 1e-6 || dt > stall_window)
  {reason = "Invalid STPIK dt (require 1e-6 <= dt <= stall_window)"; return false;}
  for (double limit : velocity_limits) {
    if (!std::isfinite(limit) || limit <= 0.0) {reason = "STPIK velocity limits must be positive"; return false;}
  }
  return true;
}

std::string StpikOptions::configuration_key() const
{
  std::ostringstream out;
  out << std::setprecision(17) << dt << ':' << stall_window << ':' << min_position_progress
      << ':' << min_orientation_progress << ':' << numerical_tolerance << ':'
      << singular_value_tolerance << ':' << wall_timeout << ':' << snapshot_joint_tolerance << ':'
      << snapshot_position_tolerance << ':' << snapshot_orientation_tolerance << ':'
      << rejoin_position_tolerance << ':' << rejoin_orientation_tolerance;
  for (double limit : velocity_limits) {out << ':' << limit;}
  return out.str();
}

StpikAction::Result StpikCheck::run(StpikSnapshot snapshot, const StpikAction::Goal & goal,
  const StpikOptions & options, const Canceled & canceled, const Progress & progress,
  const std::function<void(const WholeBodyState &, double)> & observe,
  const std::function<void(const WholeBodyState &, double)> & reference_update,
  const ValidateTasks & validate_tasks, Rollout * rollout)
{
  StpikAction::Result out;
  out.outcome = StpikAction::Result::FAIL;
  out.minimum_clearance = std::numeric_limits<double>::infinity();
  out.minimum_singular_value = std::numeric_limits<double>::infinity();
  WholeBodyState state = snapshot.initial_state;
  recordState(out, state);
  if (snapshot.model) {out.joint_names = snapshot.model->all_joint_names();}
  const auto fail = [&](const std::string & code, const std::string & reason,
      uint8_t outcome = StpikAction::Result::FAIL) {
      out.outcome = outcome; out.code = code; out.message = reason;
      recordState(out, state); return out;
    };
  std::string reason;
  if (!options.valid(reason)) {return fail("INVALID_OPTIONS", reason);}
  if (!snapshot.unavailable_reason.empty()) {
    // An unsupported/busy active task is an explicit rejection, never silently omitted.
    const bool unsupported = snapshot.unavailable_reason.find("UNSUPPORTED_ACTIVE_TASK") == 0;
    const bool busy = snapshot.unavailable_reason.find("ACTIVE_TASK_BUSY") == 0;
    return fail(unsupported ? "UNSUPPORTED_ACTIVE_TASK" : busy ? "ACTIVE_TASK_BUSY" : "SNAPSHOT_UNAVAILABLE",
      snapshot.unavailable_reason, unsupported || busy ? StpikAction::Result::FAIL : StpikAction::Result::UNAVAILABLE);
  }
  if (!snapshot.model || !snapshot.backend || !state.joints_valid || !state.navigation_valid ||
    !state.joint_positions.allFinite() || !state.joint_velocities.allFinite() ||
    !state.base_position.allFinite() || !state.base_rpy.allFinite() ||
    state.joint_positions.size() != static_cast<Eigen::Index>(snapshot.model->all_joint_names().size()))
  {return fail("INVALID_INITIAL_STATE", "Complete finite base and both-arm state is required");}
  out.joint_names = snapshot.model->all_joint_names();
  if (goal.trajectory.header.frame_id != snapshot.model->world_frame()) {
    return fail("FRAME_MISMATCH", "Trajectory must use the Task Priority world frame");
  }
  if (!goal.facing_path.points.empty()) {
    try {
      return run_facing(snapshot, goal, options, canceled, progress, observe, validate_tasks, out);
    } catch (const std::exception & exception) {
      return fail("SIMULATION_ERROR", exception.what());
    }
  }
  sura_manipulation_actions::CartesianPosePath path;
  // Sampled requests use their own poses; otherwise the rest-to-rest profile is rebuilt.
  const bool valid = goal.samples.empty() ?
    path.initTimed(goal.trajectory, reason) : path.initSamples(goal.samples, reason);
  if (!valid) {return fail("INVALID_TRAJECTORY", reason);}
  Vector6 path_tolerance, goal_tolerance;
  for (size_t i = 0; i < 6; ++i) {
    if (!std::isfinite(goal.path_tolerance[i]) || goal.path_tolerance[i] < 0.0 ||
      !std::isfinite(goal.goal_tolerance[i]) || goal.goal_tolerance[i] < 0.0)
    {return fail("INVALID_TOLERANCES", "Six finite nonnegative path/goal tolerances are required");}
    path_tolerance(i) = goal.path_tolerance[i];
    goal_tolerance(i) = goal.goal_tolerance[i];
  }
  const auto & q = goal.tolerance_orientation;
  Eigen::Quaterniond reference(q.w, q.x, q.y, q.z);
  if (!reference.coeffs().allFinite() || std::abs(reference.norm() - 1.0) > 1e-6 ||
    !std::isfinite(goal.goal_time_tolerance) || goal.goal_time_tolerance < 0.0 ||
    !std::isfinite(path.duration() + goal.goal_time_tolerance))
  {return fail("INVALID_TOLERANCES", "Invalid tolerance frame or goal convergence allowance");}
  const Eigen::Matrix3d axes = reference.normalized().toRotationMatrix();
  const Eigen::Index dofs = snapshot.model->total_dofs();
  Eigen::VectorXd limits = snapshot.solver.velocity_limits();
  if (limits.size() != dofs || !limits.allFinite() || (limits.array() <= 0.0).any()) {
    return fail("INVALID_VELOCITY_LIMITS", "Runtime controller needs finite positive generalized velocity limits");
  }
  if (!options.velocity_limits.empty()) {
    if (options.velocity_limits.size() != static_cast<size_t>(dofs)) {
      return fail("INVALID_VELOCITY_LIMITS", "Override must match active base + left + right controller order");
    }
    limits = limits.cwiseMin(Eigen::Map<const Eigen::VectorXd>(options.velocity_limits.data(), dofs));
  }
  // The solver still uses its original runtime caps. Stricter check caps are evaluated below.
  std::shared_ptr<PoseTaskCommon> selected;
  size_t selected_index = 0;
  for (size_t i = 0; i < snapshot.tasks.size(); ++i) {
    if (snapshot.tasks[i]->id() == goal.task_id) {
      selected = std::dynamic_pointer_cast<PoseTaskCommon>(snapshot.tasks[i]); selected_index = i; break;
    }
  }
  if (!selected || !selected->enabled() ||
    (!std::dynamic_pointer_cast<FramePoseTask>(selected) &&
    !std::dynamic_pointer_cast<EndEffectorPoseTask>(selected)))
  {return fail("POSE_TASK_UNAVAILABLE", "Selected enabled task must implement the Pose backend");}
  Vector6 activation = Vector6::Ones();
  if (const auto frame = std::dynamic_pointer_cast<FramePoseTask>(selected)) {
    activation = frame->activation();
  }
  // Optional whole-body nominal, streamed to robot_nominal exactly like the Pose executor.
  sura_manipulation_actions::WholeBodyNominal nominal;
  std::shared_ptr<RobotNominalTask> nominal_task;
  std::vector<Eigen::Index> nominal_columns;
  if (!goal.nominal_joints.points.empty() || !goal.nominal_base.points.empty()) {
    if (!nominal.init(goal.nominal_joints, goal.nominal_base, reason)) {return fail("INVALID_NOMINAL", reason);}
    if (nominal.frame() != snapshot.model->world_frame()) {
      return fail("FRAME_MISMATCH", "Nominal trajectory must use the Task Priority world frame");
    }
    nominal_task = find_robot_nominal_task(snapshot.tasks);
    if (!nominal_task || !nominal_task->enabled()) {
      return fail("NOMINAL_TASK_UNAVAILABLE", "A nominal trajectory requires an enabled RobotNominalTask");
    }
    const auto names = snapshot.model->all_joint_names();
    for (const auto & name : names) {
      const auto found = std::find(nominal.joint_names().begin(), nominal.joint_names().end(), name);
      if (found == nominal.joint_names().end()) {
        return fail("INVALID_NOMINAL", "Nominal trajectory is missing joint " + name);
      }
      nominal_columns.push_back(static_cast<Eigen::Index>(found - nominal.joint_names().begin()));
    }
  }
  auto simulated_time = std::make_shared<std::chrono::steady_clock::time_point>(snapshot.captured);
  TaskContext context{snapshot.model, snapshot.backend, {}, [simulated_time]() {return *simulated_time;}};
  for (auto & task : snapshot.tasks) {StpikTaskAccess::bind(*task, context);}
  if (!selected->set_feedforward_option("auto_update_target_pose",
      rclcpp::Parameter("auto_update_target_pose", false), reason))
  {return fail("POSE_TASK_UNAVAILABLE", reason);}
  std::vector<TaskComputation> computations(snapshot.tasks.size());
  for (size_t i = 0; i < snapshot.tasks.size(); ++i) {snapshot.tasks[i]->prepare_computation(computations[i]);}
  struct Observation {double time; Eigen::Isometry3d measured, desired; Vector6 tracking;};
  std::deque<Observation> history;
  const auto wall_start = std::chrono::steady_clock::now();
  const int64_t initial_ns = state.ros_time_valid ? state.ros_time_ns : 1000000000LL;
  const double end = path.duration() + goal.goal_time_tolerance;
  double time = 0.0;
  bool rank_deficient = false;
  double position_stalled_since = -1.0, orientation_stalled_since = -1.0;
  try {
    while (true) {
      out.time_from_start = time;
      if (canceled && canceled()) {return fail("CANCELED", "STPIK canceled", StpikAction::Result::CANCELED);}
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count() > options.wall_timeout) {
        return fail("CHECK_TIMEOUT", "Offline simulation exceeded wall-time budget; feasibility is unknown",
          StpikAction::Result::UNAVAILABLE);
      }
      const long double offset_ns = static_cast<long double>(time) * 1e9L;
      if (offset_ns > static_cast<long double>(std::numeric_limits<int64_t>::max()) -
        std::max<int64_t>(0, initial_ns)) {
        return fail("INVALID_TIME", "Simulated ROS time overflows");
      }
      state.ros_time_valid = true; state.ros_time_ns = initial_ns + static_cast<int64_t>(offset_ns);
      *simulated_time = snapshot.captured + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(time));
      auto desired = path.sample(time);
      auto feedforward = path.sampleVelocity(time);
      if (rollout) {rollout->target(time, desired, feedforward);}
      geometry_msgs::msg::PoseStamped target;
      target.header.frame_id = goal.trajectory.header.frame_id; target.pose = poseMessage(desired);
      if (!selected->set_pose_goal(target) ||
        !selected->set_feedforward(feedforward, selected->feedforward_epoch()))
      {return fail("REFERENCE_REJECTED", "Isolated Pose task rejected reference/feedforward");}
      if (nominal_task) {
        sura_manipulation_actions::WholeBodyNominalSample sample;
        if (!nominal.sample(time, sample, reason)) {return fail("INVALID_NOMINAL", reason);}
        Eigen::VectorXd q(static_cast<Eigen::Index>(nominal_columns.size())), qd(q.size());
        for (Eigen::Index j = 0; j < q.size(); ++j) {
          q(j) = sample.joints(nominal_columns[static_cast<size_t>(j)]);
          qd(j) = sample.joint_velocities(nominal_columns[static_cast<size_t>(j)]);
        }
        if (!apply_robot_nominal_reference(snapshot.tasks, *nominal_task, sample.base,
          sample.base_rates, q, qd, reason))
        {return fail("REFERENCE_REJECTED", "robot_nominal rejected its reference: " + reason);}
      }
      if (reference_update) {reference_update(state, time);}
      snapshot.backend->update(state);
      TaskManager::update_tasks_into(snapshot.tasks, computations, state, *snapshot.backend);
      const auto & pose_computation = computations[selected_index];
      if (!pose_computation.active || !pose_computation.has_frame_pose ||
        pose_computation.frame_id != goal.tip_frame || !pose_computation.frame_pose.matrix().allFinite())
      {return fail("POSE_TASK_INACTIVE", "Selected task cannot evaluate the requested end-effector frame");}
      const auto measured = pose_computation.frame_pose;
      if (rollout && rollout->observe_pose) {rollout->observe_pose(time, measured);}
      const Vector6 raw_tracking = error(desired, measured, axes);
      const Vector6 tracking = error(desired, measured, axes, activation);
      std::copy(raw_tracking.data(), raw_tracking.data() + 6, out.tracking_error.begin());
      for (const auto & task : snapshot.tasks) {
        if (!StpikTaskAccess::safety(*task, state, options.numerical_tolerance, out)) {
          return fail(out.code, out.message.empty() ?
            "Safety constraint violated: " + out.task_id + "/" + out.constraint : out.message);
        }
      }
      if (validate_tasks) {validate_tasks(computations);}
      if (observe) {observe(state, time);}
      if (!tracking.allFinite()) {return fail("NONFINITE_TRACKING", "Invalid Cartesian tracking error");}
      if (time == 0.0 && !within(tracking, path_tolerance, options.numerical_tolerance)) {
        return fail("INITIAL_STATE_MISMATCH", "Trajectory start does not match the controller snapshot");
      }
      HierarchySolver::Diagnostic diagnostic;
      auto command = snapshot.solver.solve(computations, &diagnostic, options.singular_value_tolerance,
        rollout ? &rollout->prefix : nullptr);
      if (!diagnostic.valid) {
        if (diagnostic.failing_task < snapshot.tasks.size()) {
          out.task_id = snapshot.tasks[diagnostic.failing_task]->id();
        }
        return fail("SOLVER_FAILURE", diagnostic.message);
      }
      rank_deficient = diagnostic.rank_deficient[selected_index];
      out.minimum_singular_value = std::min(out.minimum_singular_value,
        diagnostic.minimum_singular_values[selected_index]);
      if (rollout) {rollout->command(time, state, computations, tracking, rollout->prefix, command);}
      TaskManager::finalize_tasks_command(snapshot.tasks, computations, *snapshot.model,
        command, state.ros_time_ns);
      const auto & velocity = command.generalized_velocity;
      if (rollout) {
        // The actual (possibly saturated/stopped) command must preserve each
        // higher task's achieved velocity, including damping in the particular.
        for (size_t i = 0; i < rollout->prefix.before_task; ++i) {
          const auto & task = computations[i];
          if (task.active && task.jacobian.rows() &&
            (task.jacobian * (velocity - rollout->prefix.particular)).lpNorm<Eigen::Infinity>() >
            options.numerical_tolerance)
          {out.task_id = snapshot.tasks[i]->id(); return fail("PRIORITY_RESIDUAL", "Higher task velocity changed");}
        }
      }
      if (velocity.size() != dofs || !velocity.allFinite()) {
        return fail("NONFINITE_COMMAND", "Invalid final generalized velocity");
      }
      for (Eigen::Index i = 0; i < dofs; ++i) {
        out.maximum_velocity_ratio = std::max(out.maximum_velocity_ratio, std::abs(velocity(i)) / limits(i));
        if (std::abs(velocity(i)) > limits(i) + options.numerical_tolerance) {
          out.constraint = "generalized_velocity[" + std::to_string(i) + "]";
          return fail("VELOCITY_LIMIT", "Final command exceeds validation/controller velocity cap");
        }
      }
      if (time <= path.duration() && !within(tracking, path_tolerance, options.numerical_tolerance)) {
        out.task_id = goal.task_id;
        return fail(rank_deficient ? "TRACKING_SINGULARITY" : "PATH_TRACKING",
          "Cartesian path tolerance exceeded" + std::string(rank_deficient ? " with deficient projected Jacobian" : ""));
      }
      const bool reached = rollout && rollout->stop_when ? rollout->stop_when(time, measured) :
        time >= path.duration() && within(tracking, goal_tolerance, options.numerical_tolerance);
      if (reached) {
        out.outcome = StpikAction::Result::SUCCESS; out.code = "SUCCESS";
        out.message = "Offline Task Priority simulation reached the goal within all constraints";
        recordState(out, state); return out;
      }
      if (time >= end) {
        out.task_id = goal.task_id;
        return fail(rank_deficient ? "GOAL_SINGULARITY" : "GOAL_TOLERANCE",
          "Goal not reached within the configured convergence allowance");
      }
      history.push_back({time, measured, desired, tracking});
      while (history.size() > 1 && time - history[1].time >= options.stall_window) {history.pop_front();}
      if (time - history.front().time >= options.stall_window) {
        const auto & old = history.front();
        const Vector6 movement = error(measured, old.measured, axes, activation);
        const Vector6 demand = error(desired, old.desired, axes, activation);
        // Only demand progress on error outside the per-axis goal tolerances.
        const Vector6 excess = (tracking.cwiseAbs() - goal_tolerance -
          Vector6::Constant(options.numerical_tolerance)).cwiseMax(0.0);
        const double position_excess = excess.head<3>().norm();
        const double orientation_excess = excess.tail<3>().norm();
        const bool position_requested = position_excess > 0.0;
        const bool orientation_requested = orientation_excess > 0.0;
        const double position_required = std::min(options.min_position_progress,
          std::max(options.numerical_tolerance, 0.1 * position_excess));
        const double orientation_required = std::min(options.min_orientation_progress,
          std::max(options.numerical_tolerance, 0.1 * orientation_excess));
        const bool position_progress = movement.head<3>().norm() >= position_required ||
          old.tracking.head<3>().norm() - tracking.head<3>().norm() >= position_required;
        const bool orientation_progress = movement.tail<3>().norm() >= orientation_required ||
          old.tracking.tail<3>().norm() - tracking.tail<3>().norm() >= orientation_required;
        const bool position_stalled = position_requested && !position_progress;
        const bool orientation_stalled = orientation_requested && !orientation_progress;
        if (!position_stalled) {position_stalled_since = -1.0;}
        else if (position_stalled_since < 0.0) {position_stalled_since = time;}
        if (!orientation_stalled) {orientation_stalled_since = -1.0;}
        else if (orientation_stalled_since < 0.0) {orientation_stalled_since = time;}
        // Confirm a stall over an additional full window, resetting on any progress.
        const bool position_confirmed = position_stalled_since >= 0.0 &&
          time - position_stalled_since + 1e-12 >= options.stall_window;
        const bool orientation_confirmed = orientation_stalled_since >= 0.0 &&
          time - orientation_stalled_since + 1e-12 >= options.stall_window;
        if (position_confirmed || orientation_confirmed) {
          out.task_id = goal.task_id;
          out.constraint = position_stalled && orientation_stalled ? "position_and_orientation" :
            position_stalled ? "position" : "orientation";
          std::ostringstream diagnostic;
          diagnostic << std::boolalpha << "Insufficient progress; stalled=" << out.constraint
            << "; position_stalled=" << position_stalled
            << "; orientation_stalled=" << orientation_stalled
            << "; window=[" << old.time << ", " << time << "] s"
            << "; position{requested=" << position_requested
            << ", reference_delta_m=" << demand.head<3>().norm()
            << ", measured_delta_m=" << movement.head<3>().norm()
            << ", error_reduction_m=" << old.tracking.head<3>().norm() - tracking.head<3>().norm()
            << ", minimum_progress_m=" << position_required
            << ", configured_maximum_progress_m=" << options.min_position_progress
            << ", error_outside_tolerance_m=" << position_excess << "}"
            << "; orientation{requested=" << orientation_requested
            << ", reference_delta_rad=" << demand.tail<3>().norm()
            << ", measured_delta_rad=" << movement.tail<3>().norm()
            << ", error_reduction_rad=" << old.tracking.tail<3>().norm() - tracking.tail<3>().norm()
            << ", minimum_progress_rad=" << orientation_required
            << ", configured_maximum_progress_rad=" << options.min_orientation_progress
            << ", error_outside_tolerance_rad=" << orientation_excess << "}";
          return fail(rank_deficient ? "STALLED_SINGULARITY" : "STALLED",
            diagnostic.str());
        }
      }
      if (progress) {progress(time, end);}
      const double boundary = time < path.duration() ? path.duration() : end;
      // Avoid an extra sub-nanosecond knot when repeated dt additions fall a
      // few ULPs below the exact end of the segment.
      const double next = boundary - (time + options.dt) < 1e-12 ?
        boundary : time + options.dt;
      const double step = next - time;
      if (!(step > 0.0) || time + step <= time) {return fail("INVALID_TIME_STEP", "Simulation cannot advance time");}
      if (!integrate(*snapshot.model, velocity, step, state)) {
        return fail("NONFINITE_STATE", "Integrated state is not finite");
      }
      time = next;
    }
  } catch (const std::exception & exception) {
    return fail("SIMULATION_ERROR", exception.what());
  }
}
}  // namespace task_priority_kinematic_control
