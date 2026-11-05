#include "task_priority_kinematic_control/core/stpik_check.hpp"
#include "task_priority_kinematic_control/tasks/pose_task_common.hpp"
#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effector_pose_task.hpp"
#include "sura_manipulation_actions/cartesian_pose_path.hpp"
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
  const Eigen::Matrix3d & reference)
{
  Vector6 out;
  out.head<3>() = reference.transpose() * (desired.translation() - measured.translation());
  const Eigen::AngleAxisd delta(desired.rotation() * measured.rotation().transpose());
  out.tail<3>() = reference.transpose() * (delta.axis() * delta.angle()); return out;
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
      snapshot_position_tolerance, snapshot_orientation_tolerance})
  {
    if (!std::isfinite(nonnegative) || nonnegative < 0.0) {
      reason = "STPIK numerical and snapshot tolerances must be finite and nonnegative"; return false;
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
      << snapshot_position_tolerance << ':' << snapshot_orientation_tolerance;
  for (double limit : velocity_limits) {out << ':' << limit;}
  return out.str();
}

StpikAction::Result StpikCheck::run(StpikSnapshot snapshot, const StpikAction::Goal & goal,
  const StpikOptions & options, const Canceled & canceled, const Progress & progress)
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
  sura_manipulation_actions::CartesianPosePath path;
  if (!path.initTimed(goal.trajectory, reason)) {return fail("INVALID_TRAJECTORY", reason);}
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
      const auto desired = path.sample(time);
      geometry_msgs::msg::PoseStamped target;
      target.header.frame_id = goal.trajectory.header.frame_id; target.pose = poseMessage(desired);
      if (!selected->set_pose_goal(target) ||
        !selected->set_feedforward(path.sampleVelocity(time), selected->feedforward_epoch()))
      {return fail("REFERENCE_REJECTED", "Isolated Pose task rejected reference/feedforward");}
      snapshot.backend->update(state);
      TaskManager::update_tasks_into(snapshot.tasks, computations, state, *snapshot.backend);
      const auto & pose_computation = computations[selected_index];
      if (!pose_computation.active || !pose_computation.has_frame_pose ||
        pose_computation.frame_id != goal.tip_frame || !pose_computation.frame_pose.matrix().allFinite())
      {return fail("POSE_TASK_INACTIVE", "Selected task cannot evaluate the requested end-effector frame");}
      const auto measured = pose_computation.frame_pose;
      const Vector6 tracking = error(desired, measured, axes);
      std::copy(tracking.data(), tracking.data() + 6, out.tracking_error.begin());
      for (const auto & task : snapshot.tasks) {
        if (!StpikTaskAccess::safety(*task, state, options.numerical_tolerance, out)) {
          return fail(out.code, "Safety constraint violated: " + out.task_id + "/" + out.constraint);
        }
      }
      if (!tracking.allFinite()) {return fail("NONFINITE_TRACKING", "Invalid Cartesian tracking error");}
      if (time == 0.0 && !within(tracking, path_tolerance, options.numerical_tolerance)) {
        return fail("INITIAL_STATE_MISMATCH", "Trajectory start does not match the controller snapshot");
      }
      HierarchySolver::Diagnostic diagnostic;
      auto command = snapshot.solver.solve(computations, &diagnostic, options.singular_value_tolerance);
      if (!diagnostic.valid) {
        if (diagnostic.failing_task < snapshot.tasks.size()) {
          out.task_id = snapshot.tasks[diagnostic.failing_task]->id();
        }
        return fail("SOLVER_FAILURE", diagnostic.message);
      }
      rank_deficient = diagnostic.rank_deficient[selected_index];
      out.minimum_singular_value = std::min(out.minimum_singular_value,
        diagnostic.minimum_singular_values[selected_index]);
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
      if (time <= path.duration() && !within(tracking, path_tolerance, options.numerical_tolerance)) {
        out.task_id = goal.task_id;
        return fail(rank_deficient ? "TRACKING_SINGULARITY" : "PATH_TRACKING",
          "Cartesian path tolerance exceeded" + std::string(rank_deficient ? " with deficient projected Jacobian" : ""));
      }
      if (time >= path.duration() && within(tracking, goal_tolerance, options.numerical_tolerance)) {
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
        const Vector6 movement = error(measured, old.measured, axes);
        const Vector6 demand = error(desired, old.desired, axes);
        const bool position_requested = demand.head<3>().norm() >= options.min_position_progress ||
          (tracking.head<3>().cwiseAbs().array() > goal_tolerance.head<3>().array()).any();
        const bool orientation_requested = demand.tail<3>().norm() >= options.min_orientation_progress ||
          (tracking.tail<3>().cwiseAbs().array() > goal_tolerance.tail<3>().array()).any();
        const bool position_progress = movement.head<3>().norm() >= options.min_position_progress ||
          old.tracking.head<3>().norm() - tracking.head<3>().norm() >= options.min_position_progress;
        const bool orientation_progress = movement.tail<3>().norm() >= options.min_orientation_progress ||
          old.tracking.tail<3>().norm() - tracking.tail<3>().norm() >= options.min_orientation_progress;
        const bool position_stalled = position_requested && !position_progress;
        const bool orientation_stalled = orientation_requested && !orientation_progress;
        if (position_stalled || orientation_stalled) {
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
            << ", minimum_progress_m=" << options.min_position_progress << "}"
            << "; orientation{requested=" << orientation_requested
            << ", reference_delta_rad=" << demand.tail<3>().norm()
            << ", measured_delta_rad=" << movement.tail<3>().norm()
            << ", error_reduction_rad=" << old.tracking.tail<3>().norm() - tracking.tail<3>().norm()
            << ", minimum_progress_rad=" << options.min_orientation_progress << "}";
          return fail(rank_deficient ? "STALLED_SINGULARITY" : "STALLED",
            diagnostic.str());
        }
      }
      if (progress) {progress(time, end);}
      double step = std::min(options.dt, end - time);
      if (time < path.duration()) {step = std::min(step, path.duration() - time);}
      if (!(step > 0.0) || time + step <= time) {return fail("INVALID_TIME_STEP", "Simulation cannot advance time");}
      Eigen::Matrix<double, 6, 1> base_velocity = Eigen::Matrix<double, 6, 1>::Zero();
      const auto & axes_list = snapshot.model->active_base_dofs();
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
      if (!state.joint_positions.allFinite() || !state.base_position.allFinite() || !state.base_rpy.allFinite()) {
        return fail("NONFINITE_STATE", "Integrated state is not finite");
      }
      time += step;
    }
  } catch (const std::exception & exception) {
    return fail("SIMULATION_ERROR", exception.what());
  }
}
}  // namespace task_priority_kinematic_control
