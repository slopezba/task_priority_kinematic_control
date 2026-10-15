#include "task_priority_kinematic_control/tasks/base_pose_trajectory_task.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace task_priority_kinematic_control
{
namespace
{
int64_t stamp_ns(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
}
double duration_seconds(const builtin_interfaces::msg::Duration & duration)
{
  return static_cast<double>(duration.sec) + duration.nanosec * 1e-9;
}
}

void BasePoseTrajectoryTask::configure(const std::string & id, const std::string & plugin,
  const std::map<std::string, rclcpp::Parameter> & parameters, const TaskContext & context)
{
  configure_common(id, plugin, parameters, context);
  const auto gain = get_double_array_param(parameters, "gain", {});
  if (!gain.empty() && gain.size() != 6) {throw std::invalid_argument("Base trajectory gain needs six values");}
  std::string message;
  if (!gain.empty() && !set_gain(gain, message)) {throw std::invalid_argument(message);}
}

bool BasePoseTrajectoryTask::validate_trajectory(
  const trajectory_msgs::msg::MultiDOFJointTrajectory & trajectory, std::string & message) const
{
  if (trajectory.header.stamp.sec < 0 || trajectory.header.stamp.nanosec >= 1000000000U) {
    message = "Base trajectory start timestamp is invalid"; return false;
  }
  if (trajectory.header.frame_id != context_.model->world_frame()) {
    message = "Base trajectory must be expressed in the configured world frame"; return false;
  }
  if (trajectory.joint_names.size() != 1 || trajectory.joint_names.front().empty()) {
    message = "Base trajectory must contain exactly one named floating joint"; return false;
  }
  if (!sura_manipulation_actions::validateBaseTrajectory(trajectory, message)) {return false;}
  if (trajectory.points.front().time_from_start.sec != 0 ||
    trajectory.points.front().time_from_start.nanosec != 0)
  {
    message = "Base trajectory must start at time_from_start zero"; return false;
  }
  for (const auto & point : trajectory.points) {
    const auto & q = point.transforms.front().rotation;
    if (std::abs(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w - 1.0) > 1e-6) {
      message = "Base trajectory quaternion must be normalized"; return false;
    }
  }
  return true;
}

bool BasePoseTrajectoryTask::set_base_trajectory(
  const trajectory_msgs::msg::MultiDOFJointTrajectory & trajectory,
  const std::vector<double> & tolerance, double timeout, std::string & message)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!enabled_) {message = "Base trajectory task is disabled"; return false;}
  if (!context_.clock) {message = "Base trajectory requires the host ROS clock"; return false;}
  if (!validate_trajectory(trajectory, message)) {return false;}
  if (tolerance.size() != 6 || !std::all_of(tolerance.begin(), tolerance.end(),
    [](double x) {return std::isfinite(x) && x >= 0.0;}) || !std::isfinite(timeout) || timeout < 0.0)
  {
    message = "Base trajectory needs six nonnegative goal tolerances and a nonnegative timeout"; return false;
  }
  const int64_t now_ns = context_.clock->now().nanoseconds();
  const int64_t requested = stamp_ns(trajectory.header.stamp);
  if (requested != 0 && requested < now_ns) {message = "Base trajectory start timestamp is in the past"; return false;}
  trajectory_ = trajectory;
  sura_manipulation_actions::prepareBaseTrajectory(trajectory_, knots_, message);
  start_ns_ = requested == 0 ? now_ns : requested;
  previous_ns_ = now_ns;
  command_time_ns_ = now_ns;
  previous_command_.setZero();
  tolerance_ = Eigen::Map<const Coordinates>(tolerance.data());
  timeout_ = timeout;
  installed_ = true;
  ros_cycle_seen_ = false;
  succeeded_ = failed_ = holding_ = false;
  message_ = "waiting_for_start";
  message = "Base trajectory accepted";
  return true;
}

TaskComputation BasePoseTrajectoryTask::update(const WholeBodyState & state,
  const KinematicsBackend &)
{
  std::lock_guard<std::mutex> lock(mutex_);
  TaskComputation out;
  out.status_message = message_;
  if (!enabled_ || !installed_ || failed_) {return out;}
  if (!state.navigation_valid || !state.base_position.allFinite() || !state.base_rpy.allFinite()) {
    out.status_message = "waiting_for_navigation"; return out;
  }
  const int64_t now_ns = state.ros_time_valid ? state.ros_time_ns : context_.clock->now().nanoseconds();
  if (ros_cycle_seen_ && now_ns < previous_ns_ && !holding_) {
    failed_ = true; message_ = "ROS time moved backwards"; out.status_message = message_; return out;
  }
  previous_ns_ = now_ns;
  ros_cycle_seen_ = true;
  const double elapsed = static_cast<double>(now_ns - start_ns_) * 1e-9;
  sura_manipulation_actions::BaseTrajectorySample sample;
  std::string error;
  if (!sura_manipulation_actions::sampleBaseTrajectory(knots_, std::max(0.0, elapsed), sample, error)) {
    failed_ = true; message_ = error; out.status_message = message_; return out;
  }
  target_ = sample.coordinates;
  Coordinates current;
  current << state.base_position, state.base_rpy;
  last_error_ = target_ - current;
  for (Eigen::Index i = 3; i < 6; ++i) {
    last_error_(i) = std::atan2(std::sin(last_error_(i)), std::cos(last_error_(i)));
  }
  if (std::abs(std::cos(current(4))) < 1e-4) {
    failed_ = true; message_ = "base RPY singularity"; out.status_message = message_; return out;
  }
  const Eigen::Matrix3d angular_map = sura_manipulation_actions::baseAngularVelocityMap(current);
  const Eigen::Matrix3d inverse_map = angular_map.inverse();
  out.jacobian = Eigen::MatrixXd::Zero(6, context_.model->total_dofs());
  for (size_t i = 0; i < context_.model->active_base_dofs().size(); ++i) {
    const int axis = context_.model->active_base_dofs()[i];
    if (axis < 3) {out.jacobian(axis, i) = 1.0;}
    else {out.jacobian.block<3, 1>(3, i) = inverse_map.col(axis - 3);}
  }
  out.error = last_error_;
  out.feedforward = elapsed < 0.0 || holding_ || elapsed >= knots_.back().time ?
    Coordinates::Zero().eval() : sample.velocity;
  out.desired_velocity = out.feedforward + gains_.cwiseProduct(last_error_);
  const double dt = std::max(0.0, static_cast<double>(now_ns - command_time_ns_) * 1e-9);
  for (Eigen::Index axis = 0; axis < 6; ++axis) {
    const double limited = std::clamp(out.desired_velocity(axis), -max_velocity_(axis), max_velocity_(axis));
    out.desired_velocity(axis) = std::clamp(limited,
      previous_command_(axis) - max_acceleration_(axis) * dt,
      previous_command_(axis) + max_acceleration_(axis) * dt);
  }
  previous_command_ = out.desired_velocity;
  command_time_ns_ = now_ns;
  out.active = true;
  out.has_frame_pose = true;
  out.frame_id = context_.model->base_frame();
  out.frame_pose = sura_manipulation_actions::basePose(current);
  const double duration = duration_seconds(trajectory_.points.back().time_from_start);
  if (!holding_ && elapsed >= duration && (last_error_.cwiseAbs().array() <= tolerance_.array() + 1e-9).all()) {
    succeeded_ = true; holding_ = true;
  }
  if (!holding_ && elapsed > duration + timeout_) {
    failed_ = true; out.active = false; message_ = "base goal tolerance timed out";
  } else {message_ = holding_ ? "holding" : (elapsed < 0.0 ? "waiting_for_start" : "tracking");}
  out.status_message = message_;
  return out;
}

void BasePoseTrajectoryTask::hold_measured(const WholeBodyState & state)
{
  std::lock_guard<std::mutex> lock(mutex_);
  target_ << state.base_position, state.base_rpy;
  trajectory_ = trajectory_msgs::msg::MultiDOFJointTrajectory();
  trajectory_.header.frame_id = context_.model->world_frame();
  trajectory_.joint_names = {"floating_base"};
  trajectory_.points.push_back(sura_manipulation_actions::baseTrajectoryPoint(
    target_, Coordinates::Zero(), Coordinates::Zero(), builtin_interfaces::msg::Duration()));
  std::string ignored;
  sura_manipulation_actions::prepareBaseTrajectory(trajectory_, knots_, ignored);
  installed_ = holding_ = true;
  succeeded_ = failed_ = false;
  enabled_ = true;
  start_ns_ = previous_ns_ = state.ros_time_ns;
  command_time_ns_ = state.ros_time_valid ? state.ros_time_ns : context_.clock->now().nanoseconds();
  previous_command_.setZero();
  message_ = "holding_measured";
}

bool BasePoseTrajectoryTask::cancel_base_trajectory(std::string & message)
{
  std::lock_guard<std::mutex> lock(mutex_);
  const bool installed = installed_;
  trajectory_ = trajectory_msgs::msg::MultiDOFJointTrajectory();
  knots_.clear();
  installed_ = succeeded_ = failed_ = holding_ = false;
  message_ = enabled_ ? "idle" : "disabled";
  message = "Base trajectory cleared";
  return installed;
}

void BasePoseTrajectoryTask::set_motion_limits(const std::vector<double> & velocity,
  const std::vector<double> & acceleration)
{
  if (velocity.size() != 6 || acceleration.size() != 6 ||
    !std::all_of(velocity.begin(), velocity.end(), [](double v) {return std::isfinite(v) && v > 0.0;}) ||
    !std::all_of(acceleration.begin(), acceleration.end(), [](double v) {return std::isfinite(v) && v > 0.0;}))
  {throw std::invalid_argument("Base motion limits need six positive finite velocities and accelerations");}
  std::lock_guard<std::mutex> lock(mutex_);
  max_velocity_ = Eigen::Map<const Coordinates>(velocity.data());
  max_acceleration_ = Eigen::Map<const Coordinates>(acceleration.data());
}

JointTrajectoryTaskStatus BasePoseTrajectoryTask::base_trajectory_status() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  JointTrajectoryTaskStatus status;
  status.accepted = installed_;
  status.active = installed_ && !holding_ && !failed_;
  status.succeeded = succeeded_;
  status.timed_out = failed_;
  status.final_error_norm = last_error_.norm();
  status.message = message_;
  return status;
}

bool BasePoseTrajectoryTask::set_enabled(bool value)
{
  std::lock_guard<std::mutex> lock(mutex_);
  enabled_ = value;
  if (!value) {installed_ = succeeded_ = failed_ = holding_ = false; trajectory_.points.clear(); knots_.clear(); message_ = "disabled";}
  return true;
}
bool BasePoseTrajectoryTask::enabled() const
{std::lock_guard<std::mutex> lock(mutex_); return enabled_;}
void BasePoseTrajectoryTask::reset()
{std::string message; cancel_base_trajectory(message);}
bool BasePoseTrajectoryTask::set_gain(const std::vector<double> & gain, std::string & message)
{
  if (gain.size() != 6 || !std::all_of(gain.begin(), gain.end(),
    [](double x) {return std::isfinite(x) && x >= 0.0;}))
  {message = "Base trajectory gain requires six finite nonnegative values"; return false;}
  std::lock_guard<std::mutex> lock(mutex_);
  gains_ = Eigen::Map<const Coordinates>(gain.data());
  message = "Base trajectory gain updated"; return true;
}
msg::TaskStatus BasePoseTrajectoryTask::build_status() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto status = TaskBaseCommon::build_status();
  status.active = enabled_ && installed_ && !failed_;
  status.target_type = "base_pose_trajectory";
  status.status_message = message_;
  return status;
}
std::vector<double> BasePoseTrajectoryTask::current_target() const
{std::lock_guard<std::mutex> lock(mutex_); return {target_.data(), target_.data() + target_.size()};}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(task_priority_kinematic_control::BasePoseTrajectoryTask,
  task_priority_kinematic_control::TaskBase)
