#include "task_priority_kinematic_control/tasks/stabilize_limits_task.hpp"

#include <pluginlib/class_list_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace task_priority_kinematic_control
{

void StabilizeLimitsTask::configure(
  const std::string & id, const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  if (!context.model) {throw std::invalid_argument("StabilizeLimitsTask requires a model");}
  const auto lower = get_double_array_param(parameters, "lower_limits", {});
  const auto upper = get_double_array_param(parameters, "upper_limits", {});
  if (lower.size() != 2 || upper.size() != 2) {
    throw std::invalid_argument("Stabilize limits require two lower/upper values in roll,pitch order");
  }
  alpha_ = get_double_param(parameters, "alpha", 0.06981317);
  delta_ = get_double_param(parameters, "delta", 0.13962634);
  eps_ = get_double_param(parameters, "eps", 0.0001);
  gain_ = get_double_param(parameters, "gain_scalar", 2.0);
  max_repulsive_velocity_ = get_double_param(parameters, "max_repulsive_velocity", 0.3);
  for (const double value : {alpha_, delta_, eps_, gain_, max_repulsive_velocity_}) {
    if (!std::isfinite(value)) {throw std::invalid_argument("Stabilize limit parameters must be finite");}
  }
  if (alpha_ < 0 || delta_ <= alpha_ || eps_ < 0 || eps_ >= delta_ - alpha_ ||
    gain_ < 0 || max_repulsive_velocity_ <= 0) {
    throw std::invalid_argument("Require 0 <= alpha < delta, 0 <= eps < delta-alpha, nonnegative gain and positive max velocity");
  }
  for (Eigen::Index axis = 0; axis < 2; ++axis) {
    if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis]) ||
      lower[axis] >= upper[axis] || 2 * delta_ >= upper[axis] - lower[axis]) {
      throw std::invalid_argument("Stabilize limits must be finite, ordered and wider than twice delta");
    }
    lower_limits_(axis) = lower[axis];
    upper_limits_(axis) = upper[axis];
  }
  const double half_pi = std::acos(-1.0) / 2;
  if (lower_limits_.y() <= -half_pi || upper_limits_.y() >= half_pi) {
    throw std::invalid_argument("Pitch limits must lie strictly between -pi/2 and pi/2");
  }
  dofs_ = static_cast<Eigen::Index>(context.model->total_dofs());
  angular_columns_ = {{-1, -1}};
  const auto & base_dofs = context.model->active_base_dofs();
  for (size_t column = 0; column < base_dofs.size(); ++column) {
    if (base_dofs[column] == 3 || base_dofs[column] == 4) {
      angular_columns_[static_cast<size_t>(base_dofs[column] - 3)] = static_cast<Eigen::Index>(column);
    }
  }
  last_status_.reserve(32);
  publish_marker_rate_ = get_double_param(parameters, "publish_marker_rate", 2.0);
  const double period_ns = 1e9 / publish_marker_rate_;
  if (!std::isfinite(publish_marker_rate_) || publish_marker_rate_ <= 0 ||
    !std::isfinite(period_ns) || period_ns < 1 ||
    period_ns >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw std::invalid_argument("publish_marker_rate must be finite, positive and representable");
  }
  snapshot_period_ns_ = std::max<int64_t>(1, static_cast<int64_t>(period_ns));
  has_snapshot_ = false;
  set_debug(get_bool_param(parameters, "debug", false));
  snapshots_.initialize([&](StabilizeLimitsSnapshot & snapshot) {
      snapshot = StabilizeLimitsSnapshot{};
      snapshot.lower = lower_limits_; snapshot.upper = upper_limits_;
      snapshot.alpha = alpha_; snapshot.delta = delta_;
    });
  reset();
}

void StabilizeLimitsTask::prepare_computation(TaskComputation & out) const
{
  out.jacobian.setZero(2, dofs_);
  out.error.setZero(2);
  out.desired_velocity.setZero(2);
  out.feedforward.resize(0);
  out.active = false;
  out.stop_arm_motion = false;
  out.has_frame_pose = false;
  out.has_updated_target = false;
  out.frame_id.clear();
  out.frame_pose.setIdentity();
  out.status_message.reserve(32);
}

TaskComputation StabilizeLimitsTask::update(
  const WholeBodyState & state, const KinematicsBackend & backend)
{
  TaskComputation out;
  prepare_computation(out);
  update_into(state, backend, out);
  return out;
}

void StabilizeLimitsTask::update_into(
  const WholeBodyState & state, const KinematicsBackend &, TaskComputation & out)
{
  out.jacobian.setZero();
  out.error.setZero();
  out.desired_velocity.setZero();
  out.active = false;
  marker_active_ = {{false, false}};
  if (!enabled_ || !state.navigation_valid || !state.base_rpy.allFinite()) {
    reset();
    record_status(out, !enabled_ ? "disabled" :
      (!state.navigation_valid ? "waiting_for_navigation" : "invalid_navigation"), state);
    return;
  }
  const double cp = std::cos(state.base_rpy.y());
  if (std::abs(cp) < 1e-4) {
    reset();
    record_status(out, "roll/pitch_singularity", state);
    return;
  }
  const double yaw = state.base_rpy.z();
  // Euler tilt rates from world angular velocity, as used by stabilize.
  // The yaw column is identically zero: yaw remains unconstrained.
  Eigen::Matrix2d tilt_map;
  tilt_map << std::cos(yaw) / cp, std::sin(yaw) / cp,
    -std::sin(yaw), std::cos(yaw);

  for (Eigen::Index axis = 0; axis < 2; ++axis) {
    const auto index = static_cast<size_t>(axis);
    const double angle = state.base_rpy(axis);
    const double lower = lower_limits_(axis);
    const double upper = upper_limits_(axis);
    if (!lower_active_[index] && angle <= lower + alpha_) {lower_active_[index] = true;}
    if (lower_active_[index] && angle >= lower + delta_ - eps_) {lower_active_[index] = false;}
    if (!upper_active_[index] && angle >= upper - alpha_) {upper_active_[index] = true;}
    if (upper_active_[index] && angle <= upper - delta_ + eps_) {upper_active_[index] = false;}
    if (!lower_active_[index] && !upper_active_[index]) {continue;}

    for (size_t component = 0; component < angular_columns_.size(); ++component) {
      const Eigen::Index column = angular_columns_[component];
      if (column >= 0) {out.jacobian(axis, column) = tilt_map(axis, component);}
    }
    if (out.jacobian.row(axis).isZero(1e-12)) {continue;}
    const double target = lower_active_[index] ? lower + delta_ : upper - delta_;
    out.error(axis) = target - angle;
    out.desired_velocity(axis) = std::clamp(
      gain_ * out.error(axis), -max_repulsive_velocity_, max_repulsive_velocity_);
    out.active = true;
    marker_active_[index] = true;
  }
  record_status(out, out.active ? "repelling_limits" : "limits_inactive", state);
}

void StabilizeLimitsTask::record_status(
  TaskComputation & out, const char * message, const WholeBodyState & state)
{
  out.status_message = message;
  last_status_ = message;
  last_active_ = out.active;
  last_error_ = out.error;
  last_velocity_ = out.desired_velocity;
  if (!debug()) {return;}
  marker_valid_ = state.navigation_valid && state.base_rpy.allFinite();
  marker_angles_ = state.base_rpy.head<2>();
}

void StabilizeLimitsTask::observe_command(const WholeBodyCommand &, int64_t timestamp_ns)
{
  const uint32_t visibility = marker_visibility_state();
  if (!(visibility & 1U) || timestamp_ns < 0) {return;}
  // Sample at marker frequency using the control timestamp, without reading a
  // wall clock or copying a snapshot on every control update. Clock rewinds and
  // debug transitions force a fresh sample.
  if (has_snapshot_ && visibility == last_snapshot_visibility_ && timestamp_ns >= last_snapshot_ns_ &&
    timestamp_ns - last_snapshot_ns_ < snapshot_period_ns_) {return;}
  auto & snapshot = snapshots_.writable();
  snapshot.visualization_state = visibility;
  snapshot.visualization_enabled = true;
  snapshot.enabled = enabled_;
  snapshot.valid = marker_valid_ && std::abs(std::cos(marker_angles_.y())) >= 1e-4;
  snapshot.angles = marker_angles_;
  snapshot.active = marker_active_;
  snapshot.timestamp_ns = timestamp_ns;
  snapshot.captured = std::chrono::steady_clock::now();
  snapshots_.publish();
  has_snapshot_ = true;
  last_snapshot_ns_ = timestamp_ns;
  last_snapshot_visibility_ = visibility;
}

void StabilizeLimitsTask::set_debug(bool enabled)
{
  const uint32_t previous = marker_visibility_state();
  if (((previous & 1U) != 0) == enabled) {return;}
  // The epoch prevents redisplaying snapshots from before a visualization toggle.
  marker_visibility_.store(((previous + 2U) & ~1U) | (enabled ? 1U : 0U), std::memory_order_relaxed);
}

bool StabilizeLimitsTask::set_enabled(bool enabled)
{
  if (enabled != enabled_) {
    TaskBaseCommon::set_enabled(enabled);
    reset();
  }
  return true;
}

void StabilizeLimitsTask::reset()
{
  lower_active_ = {{false, false}};
  upper_active_ = {{false, false}};
  marker_active_ = {{false, false}};
  last_active_ = false;
  last_error_.setZero();
  last_velocity_.setZero();
  last_status_ = enabled_ ? "limits_inactive" : "disabled";
}

bool StabilizeLimitsTask::set_gain_scalar(double gain, std::string & message)
{
  if (!std::isfinite(gain) || gain < 0) {
    message = "Stabilize limit gain must be finite and nonnegative";
    return false;
  }
  gain_ = gain;
  message = "Stabilize limit gain updated";
  return true;
}

msg::TaskStatus StabilizeLimitsTask::build_status() const
{
  auto status = TaskBaseCommon::build_status();
  status.active = enabled_ && last_active_;
  status.status_message = last_status_;
  status.error.assign(last_error_.data(), last_error_.data() + last_error_.size());
  status.command.assign(last_velocity_.data(), last_velocity_.data() + last_velocity_.size());
  return status;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::StabilizeLimitsTask,
  task_priority_kinematic_control::TaskBase)
