#include "task_priority_kinematic_control/core/cartesian_trajectory_coordinator.hpp"
#include "task_priority_kinematic_control/tasks/pose_task_common.hpp"
#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effectors_relative_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effector_plan_relative_pose_task.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace task_priority_kinematic_control
{
namespace
{
using Coordinates = sura_manipulation_actions::BaseCoordinates;
double seconds(const builtin_interfaces::msg::Duration & d) {return d.sec + d.nanosec * 1e-9;}
int64_t nanoseconds(const builtin_interfaces::msg::Time & t)
{return static_cast<int64_t>(t.sec) * 1000000000LL + t.nanosec;}
bool positive_values(const std::vector<double> & values, size_t size)
{return values.size() == size && std::all_of(values.begin(), values.end(), [](double v) {return std::isfinite(v) && v > 0.0;});}
bool finite_values(const std::vector<double> & values, size_t size)
{return values.size() == size && std::all_of(values.begin(), values.end(), [](double v) {return std::isfinite(v);});}
bool tolerance_values(const std::vector<double> & values, size_t size)
{return values.size() == size && std::all_of(values.begin(), values.end(), [](double v) {return std::isfinite(v) && v >= 0.0;});}
Eigen::Isometry3d pose_from_message(const geometry_msgs::msg::Pose & pose)
{
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.translation() << pose.position.x, pose.position.y, pose.position.z;
  Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  if (!q.coeffs().allFinite() || q.norm() < 1e-12 || !result.translation().allFinite()) {
    throw std::invalid_argument("Invalid pose in coordinated trajectory goal");
  }
  result.linear() = q.normalized().toRotationMatrix();
  return result;
}
Coordinates pose_error(const Eigen::Isometry3d & desired, const Eigen::Isometry3d & measured,
  const Eigen::Matrix3d & reference_rotation)
{
  Coordinates result;
  result.head<3>() = reference_rotation.transpose() * (desired.translation() - measured.translation());
  const Eigen::AngleAxisd difference(desired.rotation() * measured.rotation().transpose());
  result.tail<3>() = reference_rotation.transpose() * (difference.axis() * difference.angle());
  return result;
}
bool exceeded(const Eigen::VectorXd & error, const std::vector<double> & tolerance)
{
  for (Eigen::Index i = 0; i < error.size(); ++i) {if (std::abs(error(i)) > tolerance[static_cast<size_t>(i)] + 1e-9) {return true;}}
  return false;
}
}

void CartesianTrajectoryCoordinator::configure(TaskManager & manager,
  const std::shared_ptr<WholeBodyModel> & model, const rclcpp::Clock::SharedPtr & clock,
  const Settings & settings)
{
  if (!std::isfinite(settings.other_arm_tolerance) || settings.other_arm_tolerance < 0.0) {
    throw std::invalid_argument("Other-arm tracking tolerance must be finite and nonnegative");
  }
  manager_ = &manager; model_ = model; clock_ = clock; settings_ = settings;
  snapshots_.clear(); status_ = Status();
}

std::shared_ptr<TaskBase> CartesianTrajectoryCoordinator::find_task(const std::string & id) const
{
  if (!manager_) {return {};}
  for (const auto & task : manager_->tasks()) {if (task->id() == id) {return task;}}
  return {};
}

bool CartesianTrajectoryCoordinator::validate_goal(const Action::Goal & goal,
  const WholeBodyState & state, const KinematicsBackend & backend, std::string & error) const
{
  Action::Goal ordered;
  return normalize_goal(goal, ordered, error) && validate_ordered_goal(ordered, state, backend, error);
}

bool CartesianTrajectoryCoordinator::normalize_goal(
  const Action::Goal & request, Action::Goal & ordered, std::string & error) const
{
  if (!model_) {error = "Coordinated execution is not configured"; return false;}
  const auto & incoming = request.joint_trajectory.joint_names;
  const std::set<std::string> incoming_set(incoming.begin(), incoming.end());
  const auto matches = [&](const std::vector<std::string> & names) {
      return !names.empty() && incoming.size() == names.size() &&
        incoming_set.size() == incoming.size() &&
        incoming_set == std::set<std::string>(names.begin(), names.end());
    };
  const bool left = matches(model_->left_arm_joints());
  const bool right = matches(model_->right_arm_joints());
  if (left == right || request.arm.empty()) {
    error = "Trajectory must contain exactly one configured arm's complete joints without duplicates";
    return false;
  }
  const auto & names = left ? model_->left_arm_joints() : model_->right_arm_joints();
  std::vector<size_t> indices;
  for (const auto & name : names) {
    indices.push_back(static_cast<size_t>(std::distance(incoming.begin(),
      std::find(incoming.begin(), incoming.end(), name))));
  }
  if (request.available_dofs.size() != names.size() + 6) {
    error = "Available DOF mask needs six base axes plus arm joints"; return false;
  }
  ordered = request;
  const auto reorder = [&](auto & values, bool optional = false) {
      if (optional && values.empty()) {return true;}
      if (values.size() != indices.size()) {return false;}
      const auto original = values;
      for (size_t i = 0; i < indices.size(); ++i) {values[i] = original[indices[i]];}
      return true;
    };
  for (auto & point : ordered.joint_trajectory.points) {
    if (!reorder(point.positions) || !reorder(point.velocities) ||
      !reorder(point.accelerations) || !reorder(point.effort, true))
    {error = "Coordinated trajectory point dimensions must match its joint names"; return false;}
  }
  if (!reorder(ordered.tracking_tolerances.joint_path) ||
    !reorder(ordered.tracking_tolerances.joint_goal))
  {error = "Joint tracking tolerances must match the trajectory joint names"; return false;}
  for (size_t i = 0; i < indices.size(); ++i) {
    ordered.available_dofs[6 + i] = request.available_dofs[6 + indices[i]];
  }
  ordered.joint_trajectory.joint_names = names;
  return true;
}

bool CartesianTrajectoryCoordinator::validate_ordered_goal(const Action::Goal & goal,
  const WholeBodyState & state, const KinematicsBackend & backend, std::string & error) const
{
  if (!manager_ || !model_ || !clock_) {error = "Coordinated execution is not configured"; return false;}
  if (status_.running) {error = "A coordinated trajectory is already running"; return false;}
  if (!state.navigation_valid || !state.joints_valid || !state.base_position.allFinite() ||
    !state.base_rpy.allFinite() || !state.joint_positions.allFinite())
  {error = "Current whole-body state is unavailable"; return false;}
  const bool left = goal.joint_trajectory.joint_names == model_->left_arm_joints();
  const bool right = goal.joint_trajectory.joint_names == model_->right_arm_joints();
  if (left == right || goal.arm.empty()) {error = "Trajectory must contain exactly one configured arm's complete ordered joints"; return false;}
  const auto base = std::dynamic_pointer_cast<BasePoseTrajectoryTask>(find_task(settings_.base_task));
  const auto joints = std::dynamic_pointer_cast<JointTrajectoryTask>(find_task(
      left ? settings_.left_joint_task : settings_.right_joint_task));
  const auto hold = std::dynamic_pointer_cast<JointNominalTask>(find_task(
      left ? settings_.right_hold_task : settings_.left_hold_task));
  if (!base || !joints || !hold) {error = "Required base, arm trajectory or other-arm hold task is missing"; return false;}
  if (hold->build_status().joint_names != (left ? model_->right_arm_joints() : model_->left_arm_joints())) {
    error = "Other-arm hold task does not contain the configured opposite arm joints"; return false;
  }
  const auto names = joints->build_status().joint_names;
  if (goal.joint_trajectory.joint_names != names) {error = "Execution joint order differs from configured arm task"; return false;}
  if (goal.available_dofs.size() != names.size() + 6) {error = "Available DOF mask needs six base axes plus arm joints"; return false;}
  for (int axis = 0; axis < 6; ++axis) {
    if (goal.available_dofs[axis] && std::find(model_->active_base_dofs().begin(), model_->active_base_dofs().end(), axis) == model_->active_base_dofs().end()) {
      error = "Goal enables a base axis unavailable to the controller"; return false;
    }
  }
  if (!base->validate_trajectory(goal.base_trajectory, error) ||
    !joints->validate_trajectory(goal.joint_trajectory, error)) {return false;}
  if (goal.base_trajectory.points.size() != goal.joint_trajectory.points.size()) {error = "Base and arm waypoint counts differ"; return false;}
  if (goal.joint_trajectory.points.front().time_from_start.sec != 0 ||
    goal.joint_trajectory.points.front().time_from_start.nanosec != 0)
  {error = "Arm trajectory must begin at zero"; return false;}
  for (size_t i = 0; i < goal.base_trajectory.points.size(); ++i) {
    const auto & point = goal.joint_trajectory.points[i];
    if (!finite_values(point.velocities, names.size()) || !finite_values(point.accelerations, names.size())) {
      error = "Coordinated quintic execution requires arm velocities and accelerations at every waypoint"; return false;
    }
    if (goal.base_trajectory.points[i].time_from_start != goal.joint_trajectory.points[i].time_from_start) {
      error = "Base and arm waypoint times differ"; return false;
    }
  }
  if (nanoseconds(goal.base_trajectory.header.stamp) != nanoseconds(goal.joint_trajectory.header.stamp)) {
    error = "Base and arm start timestamps differ"; return false;
  }
  const int64_t start = nanoseconds(goal.base_trajectory.header.stamp);
  if (start != 0 && start <= clock_->now().nanoseconds()) {error = "Coordinated start timestamp is in the past"; return false;}
  if (goal.tip_frame.empty()) {error = "Controlled tip frame cannot be empty"; return false;}
  if (goal.base_reference_pose.header.frame_id != model_->world_frame() ||
    goal.target_pose.header.frame_id != model_->world_frame() ||
    goal.tolerance_reference_pose.header.frame_id != model_->world_frame())
  {error = "Execution reference poses must be in the world frame"; return false;}
  const auto & limits = goal.base_limits;
  const auto & tolerance = goal.tracking_tolerances;
  const bool bounded = !limits.lower_limits.empty() || !limits.upper_limits.empty();
  if ((bounded && (!finite_values(limits.lower_limits, 6) || !finite_values(limits.upper_limits, 6))) ||
    !positive_values(limits.max_velocity, 6) || !positive_values(limits.max_acceleration, 6) ||
    !tolerance_values(tolerance.base_path, 6) || !tolerance_values(tolerance.base_goal, 6) ||
    !tolerance_values(tolerance.joint_path, names.size()) || !tolerance_values(tolerance.joint_goal, names.size()) ||
    !tolerance_values(tolerance.tip_path, 6) || !tolerance_values(tolerance.tip_goal, 6) ||
    tolerance.goal_time_tolerance.sec < 0 || tolerance.goal_time_tolerance.nanosec >= 1000000000U)
  {error = "Execution limits and tolerances must be finite and correctly sized"; return false;}
  for (size_t i = 0; bounded && i < 6; ++i) {
    if (limits.lower_limits[i] > limits.upper_limits[i]) {error = "Base lower limits cannot exceed upper limits"; return false;}
  }
  try {
    const auto initial = pose_from_message(goal.base_reference_pose.pose);
    pose_from_message(goal.target_pose.pose); pose_from_message(goal.tolerance_reference_pose.pose);
    backend.compute_frame_state(state, goal.tip_frame);
    std::vector<size_t> dependencies;
    if (backend.frame_joint_dependencies(goal.tip_frame, dependencies)) {
      for (const auto & name : names) {
        const auto index = static_cast<size_t>(model_->joint_index(name));
        if (std::find(dependencies.begin(), dependencies.end(), index) == dependencies.end()) {
          error = "Controlled tip frame is not downstream of all selected arm joints"; return false;
        }
      }
    } else if (goal.tip_frame != (left ? model_->left_tip_frame() : model_->right_tip_frame())) {
      error = "Backend cannot certify the requested tool frame for this arm"; return false;
    }
    Coordinates current;
    current << state.base_position, state.base_rpy;
    const auto reference = sura_manipulation_actions::baseCoordinates(initial, current(5));
    Coordinates difference = reference - current;
    for (Eigen::Index i = 3; i < 6; ++i) {difference(i) = std::remainder(difference(i), 2*std::acos(-1.0));}
    if (exceeded(difference, tolerance.base_path)) {error = "Base moved since planning"; return false;}
    for (size_t i = 0; i < names.size(); ++i) {
      const int index = model_->joint_index(names[i]);
      if (index < 0 || index >= state.joint_positions.size() ||
        std::abs(goal.joint_trajectory.points.front().positions[i] - state.joint_positions(index)) > tolerance.joint_path[i])
      {error = "Arm moved since planning"; return false;}
    }
    std::vector<sura_manipulation_actions::BaseTrajectoryKnot> knots;
    if (!sura_manipulation_actions::prepareBaseTrajectory(goal.base_trajectory, knots, error)) {return false;}
    const auto planned_start = sura_manipulation_actions::baseCoordinates(initial, knots.front().coordinates);
    if ((planned_start - knots.front().coordinates).cwiseAbs().maxCoeff() > 1e-8) {
      error = "Base trajectory initial pose differs from the planning reference"; return false;
    }
    Coordinates lower = Coordinates::Constant(-std::numeric_limits<double>::infinity());
    Coordinates upper = Coordinates::Constant(std::numeric_limits<double>::infinity());
    if (bounded) {
      lower = Eigen::Map<const Coordinates>(limits.lower_limits.data());
      upper = Eigen::Map<const Coordinates>(limits.upper_limits.data());
      lower.head<3>() += initial.translation(); upper.head<3>() += initial.translation();
    }
    const std::vector<double> absolute_lower = bounded ?
      std::vector<double>(lower.data(), lower.data()+6) : std::vector<double>();
    const std::vector<double> absolute_upper = bounded ?
      std::vector<double>(upper.data(), upper.data()+6) : std::vector<double>();
    if (!sura_manipulation_actions::validateBaseTrajectoryLimits(knots, absolute_lower,
      absolute_upper, limits.max_velocity, limits.max_acceleration, goal.available_dofs, error))
    {return false;}
    for (size_t i = 0; i < knots.size(); ++i) {
      const auto & knot = knots[i];
      if ((knot.coordinates.array() < lower.array() - 1e-9).any() ||
        (knot.coordinates.array() > upper.array() + 1e-9).any()) {error = "Base trajectory violates requested bounds"; return false;}
      for (size_t axis = 0; axis < 6; ++axis) {
        if (!goal.available_dofs[axis] && (std::abs(knot.coordinates(axis) - knots.front().coordinates(axis)) > 1e-8 ||
          std::abs(knot.velocity(axis)) > 1e-8 || std::abs(knot.acceleration(axis)) > 1e-8)) {
          error = "Base trajectory moves a disabled DOF"; return false;
        }
      }
      for (size_t j = 0; j < names.size(); ++j) {
        if (!goal.available_dofs[j+6] && (std::abs(goal.joint_trajectory.points[i].positions[j] -
          goal.joint_trajectory.points.front().positions[j]) > 1e-8 ||
          std::abs(goal.joint_trajectory.points[i].velocities[j]) > 1e-8 ||
          std::abs(goal.joint_trajectory.points[i].accelerations[j]) > 1e-8))
        {error = "Arm trajectory moves a disabled DOF"; return false;}
      }
    }
  } catch (const std::exception & exception) {error = exception.what(); return false;}
  return true;
}

bool CartesianTrajectoryCoordinator::install(const Action::Goal & request,
  const WholeBodyState & state, const KinematicsBackend & backend, std::string & error)
{
  Action::Goal ordered;
  if (!normalize_goal(request, ordered, error) ||
    !validate_ordered_goal(ordered, state, backend, error)) {return false;}
  const int64_t requested = nanoseconds(request.base_trajectory.header.stamp);
  const int64_t now_ns = clock_->now().nanoseconds();
  if (requested != 0 && requested <= now_ns) {error = "Coordinated start timestamp is in the past"; return false;}
  // Validate replacements before releasing the preceding held pose.
  if (owns_tasks() && !release(state, backend, error)) {return false;}
  goal_ = std::move(ordered);
  const bool left = goal_.joint_trajectory.joint_names == model_->left_arm_joints();
  start_ns_ = requested == 0 ? now_ns + 100000000LL : requested;
  goal_.base_trajectory.header.stamp = rclcpp::Time(start_ns_, clock_->get_clock_type());
  goal_.joint_trajectory.header.stamp = goal_.base_trajectory.header.stamp;
  base_task_ = std::dynamic_pointer_cast<BasePoseTrajectoryTask>(find_task(settings_.base_task));
  joint_task_ = std::dynamic_pointer_cast<JointTrajectoryTask>(find_task(
    left ? settings_.left_joint_task : settings_.right_joint_task));
  other_hold_task_ = std::dynamic_pointer_cast<JointNominalTask>(find_task(
    left ? settings_.right_hold_task : settings_.left_hold_task));
  task_joint_names_ = joint_task_->build_status().joint_names;
  feedback_indices_.clear();
  for (const auto & name : task_joint_names_) {
    const auto & incoming = request.joint_trajectory.joint_names;
    feedback_indices_.push_back(static_cast<size_t>(std::distance(incoming.begin(),
      std::find(incoming.begin(), incoming.end(), name))));
  }
  reference_state_ = state;
  duration_ = seconds(goal_.base_trajectory.points.back().time_from_start);
  previous_ns_ = state.ros_time_valid ? state.ros_time_ns : now_ns;
  previous_solution_ns_ = previous_ns_;
  previous_solution_.setZero();
  sura_manipulation_actions::prepareBaseTrajectory(goal_.base_trajectory, base_knots_, error);
  const auto base_reference = pose_from_message(goal_.base_reference_pose.pose);
  lower_ = Coordinates::Constant(-std::numeric_limits<double>::infinity());
  upper_ = Coordinates::Constant(std::numeric_limits<double>::infinity());
  if (!goal_.base_limits.lower_limits.empty()) {
    lower_ = Eigen::Map<const Coordinates>(goal_.base_limits.lower_limits.data());
    upper_ = Eigen::Map<const Coordinates>(goal_.base_limits.upper_limits.data());
    lower_.head<3>() += base_reference.translation(); upper_.head<3>() += base_reference.translation();
  }
  for (const auto & task : manager_->tasks()) {
    if (std::find(settings_.safety_tasks.begin(), settings_.safety_tasks.end(), task->id()) != settings_.safety_tasks.end()) {continue;}
    snapshots_.push_back({task, task->enabled(), task->priority(), task->build_status().joint_activation});
    task->set_enabled(false);
    task->reset();
  }
  base_task_->set_enabled(true); joint_task_->set_enabled(true); other_hold_task_->set_enabled(true);
  const auto hold_names = other_hold_task_->build_status().joint_names;
  std::vector<double> hold_positions;
  for (const auto & name : hold_names) {hold_positions.push_back(state.joint_positions(model_->joint_index(name)));}
  other_hold_task_->clear_velocity();
  other_hold_task_->set_joint_activation(std::vector<bool>(hold_names.size(), true), error);
  other_hold_task_->set_joint_target(hold_positions, error);
  joint_task_->set_execution_tolerances(goal_.tracking_tolerances.joint_goal,
    seconds(goal_.tracking_tolerances.goal_time_tolerance));
  base_task_->set_motion_limits(goal_.base_limits.max_velocity, goal_.base_limits.max_acceleration);
  if (!base_task_->set_base_trajectory(goal_.base_trajectory, goal_.tracking_tolerances.base_goal,
    seconds(goal_.tracking_tolerances.goal_time_tolerance), error) ||
    !joint_task_->set_joint_trajectory(goal_.joint_trajectory, error))
  {
    std::string ignored; release(state, backend, ignored); return false;
  }
  status_ = Status(); status_.running = true; status_.message = "waiting_for_start";
  status_.joint_error = Eigen::VectorXd::Zero(task_joint_names_.size());
  return true;
}

bool CartesianTrajectoryCoordinator::update(const WholeBodyState & state, const KinematicsBackend & backend)
{
  if (!status_.running) {return false;}
  const int64_t now_ns = state.ros_time_valid ? state.ros_time_ns : clock_->now().nanoseconds();
  if (now_ns < previous_ns_) {cancel_to_measured(state, "ROS time moved backwards", false); return true;}
  previous_ns_ = now_ns;
  const double elapsed = static_cast<double>(now_ns - start_ns_) * 1e-9;
  status_.progress = duration_ > 0 ? static_cast<float>(std::clamp(elapsed/duration_, 0.0, 1.0)) : 1.0F;
  status_.message = elapsed < 0 ? "waiting_for_start" : "tracking";
  std::string error;
  sura_manipulation_actions::BaseTrajectorySample base;
  Eigen::VectorXd q;
  if (!sura_manipulation_actions::sampleBaseTrajectory(base_knots_, std::max(0.0, elapsed), base, error) ||
    !joint_task_->sample_trajectory(std::max(0.0, elapsed), q))
  {cancel_to_measured(state, error.empty() ? "Trajectory samples were cleared" : error, false); return true;}
  WholeBodyState desired = reference_state_;
  desired.base_position = base.coordinates.head<3>(); desired.base_rpy = base.coordinates.tail<3>();
  Coordinates measured;
  measured << state.base_position, state.base_rpy;
  measured = sura_manipulation_actions::baseCoordinates(
    sura_manipulation_actions::basePose(measured), base.coordinates);
  status_.base_error = base.coordinates - measured;
  for (Eigen::Index axis = 3; axis < 6; ++axis) {status_.base_error(axis) = std::remainder(status_.base_error(axis), 2*std::acos(-1.0));}
  Eigen::VectorXd joint_error(task_joint_names_.size());
  for (size_t j = 0; j < task_joint_names_.size(); ++j) {
    const int index = model_->joint_index(task_joint_names_[j]);
    desired.joint_positions(index) = q(j);
    joint_error(j) = q(j) - state.joint_positions(index);
    status_.joint_error(feedback_indices_[j]) = joint_error(j);
  }
  const auto & other_names = task_joint_names_ == model_->left_arm_joints() ?
    model_->right_arm_joints() : model_->left_arm_joints();
  for (const auto & name : other_names) {
    const int index = model_->joint_index(name);
    if (std::abs(state.joint_positions(index) - reference_state_.joint_positions(index)) >
      settings_.other_arm_tolerance + 1e-9)
    {
      cancel_to_measured(state, "Unselected arm moved beyond hold tolerance: " + name, false);
      return true;
    }
  }
  try {
    const auto expected_tip = backend.compute_frame_state(desired, goal_.tip_frame).pose;
    const auto measured_tip = backend.compute_frame_state(state, goal_.tip_frame).pose;
    const auto tolerance_frame = pose_from_message(goal_.tolerance_reference_pose.pose);
    status_.tip_error = pose_error(expected_tip, measured_tip, tolerance_frame.rotation());
    if (elapsed >= duration_) {
      // Final success is checked against the original Cartesian objective.
      status_.tip_error = pose_error(pose_from_message(goal_.target_pose.pose), measured_tip, tolerance_frame.rotation());
    }
  } catch (const std::exception & exception) {cancel_to_measured(state, exception.what(), false); return true;}
  if (!measured.allFinite() || (measured.array() < lower_.array() - 1e-8).any() ||
    (measured.array() > upper_.array() + 1e-8).any())
  {cancel_to_measured(state, "Measured base pose violated requested limits", false); return true;}
  for (Eigen::Index axis = 0; axis < 6; ++axis) {
    if (std::abs(base.velocity(axis)) > goal_.base_limits.max_velocity[axis] + 1e-8 ||
      std::abs(base.acceleration(axis)) > goal_.base_limits.max_acceleration[axis] + 1e-8)
    {cancel_to_measured(state, "Interpolated base dynamics violated requested limits", false); return true;}
  }
  if (elapsed >= 0 && elapsed < duration_ &&
    (exceeded(status_.base_error, goal_.tracking_tolerances.base_path) ||
    exceeded(joint_error, goal_.tracking_tolerances.joint_path) ||
    exceeded(status_.tip_error, goal_.tracking_tolerances.tip_path)))
  {cancel_to_measured(state, "Cartesian trajectory path tolerance violated", false); return true;}
  const auto base_status = base_task_->base_trajectory_status();
  const auto joint_status = joint_task_->joint_trajectory_status();
  if (base_status.timed_out || joint_status.timed_out ||
    elapsed > duration_ + seconds(goal_.tracking_tolerances.goal_time_tolerance))
  {cancel_to_measured(state, "Cartesian trajectory goal tolerance timed out", false); return true;}
  if (elapsed >= duration_ &&
    !exceeded(status_.base_error, goal_.tracking_tolerances.base_goal) &&
    !exceeded(joint_error, goal_.tracking_tolerances.joint_goal) &&
    !exceeded(status_.tip_error, goal_.tracking_tolerances.tip_goal))
  {status_.running = false; status_.complete = status_.success = status_.holding = true; status_.message = "holding_final_pose"; return false;}
  return false;
}

bool CartesianTrajectoryCoordinator::observe_command(const WholeBodyCommand & command,
  const WholeBodyState & state, const KinematicsBackend &)
{
  if (!status_.running) {return false;}
  Coordinates world_velocity = Coordinates::Zero();
  if (command.generalized_velocity.size() < static_cast<Eigen::Index>(model_->base_dofs())) {
    cancel_to_measured(state, "Solver returned an incomplete base command", false); return true;
  }
  for (size_t column = 0; column < model_->active_base_dofs().size(); ++column) {
    world_velocity(model_->active_base_dofs()[column]) = command.generalized_velocity(column);
  }
  Coordinates current;
  current << state.base_position, state.base_rpy;
  Coordinates rates = world_velocity;
  if (std::abs(std::cos(current(4))) < 1e-4) {
    cancel_to_measured(state, "Base command conversion reached an Euler singularity", false); return true;
  }
  rates.tail<3>() = sura_manipulation_actions::baseAngularVelocityMap(current).inverse() * world_velocity.tail<3>();
  const int64_t now_ns = state.ros_time_valid ? state.ros_time_ns : clock_->now().nanoseconds();
  const double dt = std::max(0.0, static_cast<double>(now_ns - previous_solution_ns_) * 1e-9);
  if (!rates.allFinite()) {cancel_to_measured(state, "Solver produced a nonfinite base command", false); return true;}
  for (Eigen::Index axis = 0; axis < 6; ++axis) {
    if (std::abs(rates(axis)) > goal_.base_limits.max_velocity[axis] + 1e-8 ||
      std::abs(rates(axis) - previous_solution_(axis)) > goal_.base_limits.max_acceleration[axis] * dt + 1e-8)
    {
      // A safety task remains free to override motion. End the coordinated
      // goal and solve again against measured holds rather than clipping safety.
      cancel_to_measured(state, "Solved base command violated goal velocity or acceleration limits", false);
      return true;
    }
  }
  previous_solution_ = rates; previous_solution_ns_ = now_ns;
  return false;
}

void CartesianTrajectoryCoordinator::install_measured_hold(const WholeBodyState & state)
{
  if (base_task_) {base_task_->hold_measured(state);}
  if (joint_task_) {
    trajectory_msgs::msg::JointTrajectory hold;
    hold.joint_names = task_joint_names_;
    trajectory_msgs::msg::JointTrajectoryPoint point;
    for (const auto & name : task_joint_names_) {point.positions.push_back(state.joint_positions(model_->joint_index(name)));}
    point.velocities.assign(point.positions.size(), 0.0); point.accelerations = point.velocities;
    hold.points.push_back(point);
    std::string ignored; joint_task_->set_enabled(true); joint_task_->set_joint_trajectory(hold, ignored);
  }
  if (other_hold_task_) {
    std::vector<double> values;
    for (const auto & name : other_hold_task_->build_status().joint_names) {values.push_back(state.joint_positions(model_->joint_index(name)));}
    std::string ignored; other_hold_task_->clear_velocity(); other_hold_task_->set_joint_target(values, ignored);
  }
}

void CartesianTrajectoryCoordinator::cancel_to_measured(const WholeBodyState & state,
  const std::string & reason, bool canceled)
{
  if (!owns_tasks()) {return;}
  install_measured_hold(state);
  status_.running = false; status_.complete = true; status_.success = false;
  status_.canceled = canceled; status_.holding = true; status_.message = reason;
}

bool CartesianTrajectoryCoordinator::release(const WholeBodyState & state,
  const KinematicsBackend & backend, std::string & message)
{
  if (!owns_tasks()) {message = "No coordinated trajectory reservation"; return true;}
  if (joint_task_) {joint_task_->restore_execution_defaults();}
  for (const auto & snapshot : snapshots_) {
    auto & task = snapshot.task;
    task->set_enabled(false); task->reset(); task->set_priority(snapshot.priority);
    if (auto nominal = std::dynamic_pointer_cast<JointNominalTask>(task)) {
      std::vector<double> values;
      for (const auto & name : nominal->build_status().joint_names) {values.push_back(state.joint_positions(model_->joint_index(name)));}
      std::string ignored; nominal->clear_velocity(); nominal->set_joint_target(values, ignored);
      nominal->set_joint_activation(snapshot.joint_activation, ignored);
    }
    task->set_enabled(snapshot.enabled);
    if (auto pose_task = std::dynamic_pointer_cast<PoseTaskCommon>(task)) {
      pose_task->clear_feedforward();
      // Auto-target initialization captures the measured pose on its first
      // update; never resume a target from before the coordinated movement.
      try {
        const auto output = task->update(state, backend);
        if (output.has_frame_pose) {pose_task->capture_measured_target(output.frame_pose);}
      } catch (const std::exception &) {task->reset();}
    } else if (auto relative = std::dynamic_pointer_cast<EndEffectorsRelativePoseTask>(task)) {
      try {relative->capture_measured_relative_pose(backend);} catch (const std::exception &) {task->reset();}
    } else if (auto planned = std::dynamic_pointer_cast<EndEffectorPlanRelativePoseTask>(task)) {
      try {planned->capture_measured_relative_pose(backend);} catch (const std::exception &) {task->reset();}
    }
  }
  snapshots_.clear(); status_.running = false; status_.holding = false;
  message = "Coordinated reservation released; task targets captured at measured state";
  return true;
}

void CartesianTrajectoryCoordinator::shutdown(const WholeBodyState & state,
  const KinematicsBackend & backend, const std::string & reason)
{
  if (status_.running) {cancel_to_measured(state, reason, false);}
  std::string ignored; release(state, backend, ignored);
}

bool CartesianTrajectoryCoordinator::reserved(const std::string & task_id) const
{
  if (task_id == "*") {return owns_tasks();}
  return std::any_of(snapshots_.begin(), snapshots_.end(), [&](const Snapshot & s) {return s.task->id() == task_id;});
}

bool CartesianTrajectoryCoordinator::allow_mutation(const std::string & task_id,
  const WholeBodyState & state, const KinematicsBackend & backend, std::string & message)
{
  if (!reserved(task_id)) {return true;}
  if (status_.running) {message = "Task is reserved by coordinated Cartesian execution"; return false;}
  return release(state, backend, message);
}

}  // namespace task_priority_kinematic_control
