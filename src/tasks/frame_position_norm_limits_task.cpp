#include "task_priority_kinematic_control/tasks/frame_position_norm_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/pose_task_common.hpp"

#include <pluginlib/class_list_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace task_priority_kinematic_control
{

namespace
{

constexpr double kNormEpsilon = 1e-9;

// One value (or empty -> fallback) for scalar norm parameters stored as arrays.
double single_value(const std::vector<double> & values, double fallback, const std::string & name)
{
  if (values.empty()) {return fallback;}
  if (values.size() != 1) {
    throw std::invalid_argument("FramePositionNormLimitsTask: '" + name + "' must contain 1 value");
  }
  return values.front();
}

// Three-axis mask from a bool or 0/1 integer array. Empty keeps all axes.
void parse_axes3(const std::map<std::string, rclcpp::Parameter> & parameters,
  std::array<bool, 3> & axes)
{
  axes = {{true, true, true}};
  const auto it = parameters.find("axes");
  if (it == parameters.end()) {return;}
  const auto & parameter = it->second;
  std::vector<bool> values;
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL_ARRAY) {
    values = parameter.as_bool_array();
  } else if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY) {
    for (const auto value : parameter.as_integer_array()) {
      if (value != 0 && value != 1) {
        throw std::invalid_argument("FramePositionNormLimitsTask: axes integers must be 0 or 1");
      }
      values.push_back(value == 1);
    }
  } else if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
    return;
  } else {
    throw std::invalid_argument("FramePositionNormLimitsTask: axes must be a bool or 0/1 integer array");
  }
  if (values.empty()) {return;}
  if (values.size() != 3) {
    throw std::invalid_argument("FramePositionNormLimitsTask: axes must contain 3 values (x y z)");
  }
  for (size_t i = 0; i < 3; ++i) {axes[i] = values[i];}
  if (!axes[0] && !axes[1] && !axes[2]) {
    throw std::invalid_argument("FramePositionNormLimitsTask: at least one axis must be enabled");
  }
}

}  // namespace

void FramePositionNormLimitsTask::configure(
  const std::string & id, const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  if (!context.model) {throw std::invalid_argument("FramePositionNormLimitsTask requires a model");}
  frame_id_ = get_string_param(parameters, "frame_id", "");
  if (frame_id_.empty()) {frame_id_ = context.model->base_frame();}
  is_base_frame_ = frame_id_ == context.model->base_frame();
  parse_axes3(parameters, axes_);

  limit_ = single_value(get_double_array_param(parameters, "threshold", {}), 0.05, "threshold");
  alpha_ = single_value(get_double_array_param(parameters, "activation_margin", {}),
      0.2 * limit_, "activation_margin");
  delta_ = single_value(get_double_array_param(parameters, "release_margin", {}),
      0.4 * limit_, "release_margin");
  const auto gains = get_double_array_param(parameters, "gain", {});
  gain_ = gains.empty() ? get_double_param(parameters, "gain_scalar", 1.0) :
    single_value(gains, 1.0, "gain");
  eps_ = get_double_param(parameters, "eps", 1e-4);
  max_velocity_ = get_double_param(parameters, "max_repulsive_velocity", 0.1);
  always_active_ = get_bool_param(parameters, "always_active", false);

  for (const double value : {limit_, alpha_, delta_, gain_, eps_, max_velocity_}) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("FramePositionNormLimitsTask: parameters must be finite");
    }
  }
  if (gain_ < 0 || eps_ < 0 || max_velocity_ <= 0) {
    throw std::invalid_argument(
            "FramePositionNormLimitsTask: require gain >= 0, eps >= 0 and max_repulsive_velocity > 0");
  }
  if (limit_ <= 0 || alpha_ < 0 || delta_ <= alpha_ || limit_ <= delta_ || eps_ >= delta_ - alpha_) {
    throw std::invalid_argument(
            "FramePositionNormLimitsTask: require 0 <= activation_margin < release_margin < threshold "
            "and eps < release_margin - activation_margin");
  }

  dofs_ = static_cast<Eigen::Index>(context.model->total_dofs());
  frame_.jacobian.setZero(6, dofs_);
  row_.setZero(dofs_);
  last_status_.reserve(32);
  has_target_ = false;
  has_pose_goal_ = false;

  publish_marker_rate_ = get_double_param(parameters, "publish_marker_rate", 2.0);
  const double period_ns = 1e9 / publish_marker_rate_;
  if (!std::isfinite(publish_marker_rate_) || publish_marker_rate_ <= 0 ||
    !std::isfinite(period_ns) || period_ns < 1 ||
    period_ns >= static_cast<double>(std::numeric_limits<int64_t>::max()))
  {
    throw std::invalid_argument(
            "FramePositionNormLimitsTask: publish_marker_rate must be finite, positive and representable");
  }
  snapshot_period_ns_ = std::max<int64_t>(1, static_cast<int64_t>(period_ns));
  has_snapshot_ = false;
  marker_valid_ = false;
  set_debug(get_bool_param(parameters, "debug", false));
  debug_channel_.buffer.initialize([&](FramePositionNormSnapshot & snapshot) {
      snapshot = FramePositionNormSnapshot{};
      snapshot.limit = limit_; snapshot.alpha = alpha_; snapshot.delta = delta_;
      snapshot.axes = axes_; snapshot.always_active = always_active_;
    });
  reset();
}

void FramePositionNormLimitsTask::prepare_computation(TaskComputation & out) const
{
  out.structural_zero_rows = true;
  out.jacobian.setZero(1, dofs_);
  out.error.setZero(1);
  out.desired_velocity.setZero(1);
  out.feedforward.resize(0);
  out.active = false;
  out.stop_arm_motion = false;
  out.has_frame_pose = false;
  out.has_updated_target = false;
  out.frame_id.reserve(frame_id_.size());
  out.frame_pose.setIdentity();
  out.status_message.reserve(32);
}

TaskComputation FramePositionNormLimitsTask::update(
  const WholeBodyState & state, const KinematicsBackend & backend)
{
  TaskComputation out;
  prepare_computation(out);
  update_into(state, backend, out);
  return out;
}

void FramePositionNormLimitsTask::update_into(
  const WholeBodyState & state, const KinematicsBackend & backend, TaskComputation & out)
{
  out.jacobian.setZero();
  out.error.setZero();
  out.desired_velocity.setZero();
  out.active = false;
  out.has_updated_target = false;
  last_norm_ = 0.0;
  last_error_ = 0.0;
  last_velocity_ = 0.0;
  marker_valid_ = false;
  if (!enabled_) {
    record_status(out, "disabled");
    return;
  }

  if (is_base_frame_) {
    frame_.pose = pose_math::base_pose_from_state(state);
    frame_.jacobian = pose_math::base_jacobian_for_model(*context_.model);
  } else if (!backend.read_frame_state(frame_id_, frame_)) {
    row_active_ = false;
    record_status(out, "frame pose unavailable");
    return;
  }
  if (!frame_.pose.matrix().allFinite() || frame_.jacobian.rows() != 6 ||
    frame_.jacobian.cols() != dofs_ || !frame_.jacobian.allFinite())
  {
    row_active_ = false;
    record_status(out, "frame pose unavailable");
    return;
  }
  out.has_frame_pose = true;
  out.frame_id = frame_id_;
  out.frame_pose = frame_.pose;

  if (!has_target_) {
    // No goal received: centre the region where the frame is now.
    target_position_ = frame_.pose.translation();
    target_orientation_ = Eigen::Quaterniond(frame_.pose.rotation()).normalized();
    has_target_ = true;
    out.has_updated_target = true;
    out.updated_target.position.x = target_position_.x();
    out.updated_target.position.y = target_position_.y();
    out.updated_target.position.z = target_position_.z();
    out.updated_target.orientation.x = target_orientation_.x();
    out.updated_target.orientation.y = target_orientation_.y();
    out.updated_target.orientation.z = target_orientation_.z();
    out.updated_target.orientation.w = target_orientation_.w();
  }

  // Masked error in the target axes: e = S R_t^T (p_t - p).
  const Eigen::Matrix3d world_to_target = target_orientation_.toRotationMatrix().transpose();
  Eigen::Vector3d error = world_to_target * (target_position_ - frame_.pose.translation());
  for (Eigen::Index i = 0; i < 3; ++i) {
    if (!axes_[static_cast<size_t>(i)]) {error(i) = 0.0;}
  }
  const double sigma = error.norm();
  last_norm_ = sigma;
  marker_valid_ = std::isfinite(sigma);
  marker_frame_position_ = frame_.pose.translation();

  double target_sigma = 0.0;
  if (always_active_) {
    row_active_ = true;
  } else {
    if (!row_active_ && sigma >= limit_ - alpha_) {row_active_ = true;}
    if (row_active_ && sigma <= limit_ - delta_ + eps_) {row_active_ = false;}
    target_sigma = limit_ - delta_;
  }
  if (!row_active_) {
    record_status(out, "inside_limit");
    return;
  }
  if (sigma < kNormEpsilon) {
    // Direction undefined at the target itself (only reachable with always_active).
    record_status(out, "at_target");
    return;
  }
  // d(sigma)/dt = -(e/sigma)^T S R^T J_lin qdot (e is already masked).
  const Eigen::Vector3d direction = world_to_target.transpose() * (error / sigma);
  row_.noalias() = -direction.transpose() * frame_.jacobian.topRows<3>();
  if (row_.isZero(1e-12)) {
    record_status(out, "no_dof");
    return;
  }
  out.jacobian.row(0) = row_;
  out.error(0) = target_sigma - sigma;
  out.desired_velocity(0) = std::clamp(gain_ * out.error(0), -max_velocity_, max_velocity_);
  last_error_ = out.error(0);
  last_velocity_ = out.desired_velocity(0);
  out.active = true;
  record_status(out, always_active_ ? "tracking" : "holding_limit");
}

void FramePositionNormLimitsTask::record_status(TaskComputation & out, const char * message)
{
  out.status_message = message;
  last_status_ = message;
  last_active_ = out.active;
}

bool FramePositionNormLimitsTask::set_pose_goal(const geometry_msgs::msg::PoseStamped & goal)
{
  if (!context_.model) {return false;}
  if (!goal.header.frame_id.empty() && goal.header.frame_id != context_.model->world_frame()) {
    return false;
  }
  const auto & p = goal.pose.position;
  const auto & o = goal.pose.orientation;
  Eigen::Quaterniond orientation(o.w, o.x, o.y, o.z);
  const Eigen::Vector3d position(p.x, p.y, p.z);
  if (!position.allFinite() || !orientation.coeffs().allFinite()) {return false;}
  // An all-zero quaternion (common from rqt) means "no orientation": keep world axes.
  if (orientation.norm() < 1e-9) {orientation = Eigen::Quaterniond::Identity();}
  orientation.normalize();
  pose_goal_ = goal;
  pose_goal_.header.frame_id = context_.model->world_frame();
  has_pose_goal_ = true;
  target_position_ = position;
  target_orientation_ = orientation;
  has_target_ = true;
  row_active_ = false;
  return true;
}

bool FramePositionNormLimitsTask::set_gain(const std::vector<double> & gain, std::string & message)
{
  if (gain.size() != 1) {
    message = "Position norm gain must contain 1 value";
    return false;
  }
  return set_gain_scalar(gain.front(), message);
}

bool FramePositionNormLimitsTask::set_gain_scalar(double gain, std::string & message)
{
  if (!std::isfinite(gain) || gain < 0.0) {
    message = "Position norm gain must be finite and non-negative";
    return false;
  }
  gain_ = gain;
  message = "Position norm gain updated";
  return true;
}

bool FramePositionNormLimitsTask::set_enabled(bool enabled)
{
  if (enabled != enabled_) {
    TaskBaseCommon::set_enabled(enabled);
    reset();
  }
  return true;
}

void FramePositionNormLimitsTask::reset()
{
  row_active_ = false;
  // A reset forgets any received goal; the next update captures the measured pose.
  has_pose_goal_ = false;
  has_target_ = false;
  last_active_ = false;
  last_norm_ = 0.0;
  last_error_ = 0.0;
  last_velocity_ = 0.0;
  last_status_ = enabled_ ? "inside_limit" : "disabled";
}

msg::TaskStatus FramePositionNormLimitsTask::build_status() const
{
  auto status = TaskBaseCommon::build_status();
  status.active = enabled_ && last_active_;
  status.status_message = last_status_;
  status.target_type = "pose";
  status.error = {last_norm_};
  status.command = {last_velocity_};
  return status;
}

std::vector<double> FramePositionNormLimitsTask::current_target() const
{
  if (!has_target_) {return {};}
  return {target_position_.x(), target_position_.y(), target_position_.z(),
    target_orientation_.x(), target_orientation_.y(), target_orientation_.z(),
    target_orientation_.w()};
}

void FramePositionNormLimitsTask::observe_command(const WholeBodyCommand &, int64_t timestamp_ns)
{
  const uint32_t visibility = marker_visibility_state();
  if (!(visibility & 1U) || timestamp_ns < 0) {return;}
  if (has_snapshot_ && visibility == last_snapshot_visibility_ &&
    timestamp_ns >= last_snapshot_ns_ && timestamp_ns - last_snapshot_ns_ < snapshot_period_ns_)
  {
    return;
  }
  auto & snapshot = debug_channel_.buffer.writable();
  snapshot.visualization_state = visibility;
  snapshot.visualization_enabled = true;
  snapshot.enabled = enabled_;
  snapshot.valid = enabled_ && has_target_ && marker_valid_;
  snapshot.target_position = target_position_;
  snapshot.target_orientation = target_orientation_;
  snapshot.frame_position = marker_frame_position_;
  snapshot.norm = last_norm_;
  snapshot.limit = limit_;
  snapshot.alpha = alpha_;
  snapshot.delta = delta_;
  snapshot.axes = axes_;
  snapshot.active = last_active_;
  snapshot.always_active = always_active_;
  snapshot.timestamp_ns = timestamp_ns;
  snapshot.captured = std::chrono::steady_clock::now();
  debug_channel_.buffer.publish();
  has_snapshot_ = true;
  last_snapshot_ns_ = timestamp_ns;
  last_snapshot_visibility_ = visibility;
}

void FramePositionNormLimitsTask::set_debug(bool enabled)
{
  auto & visibility = debug_channel_.visibility;
  const uint32_t previous = visibility.load(std::memory_order_relaxed);
  if (((previous & 1U) != 0) == enabled) {return;}
  // The epoch prevents redisplaying snapshots from before a visualization toggle.
  visibility.store(((previous + 2U) & ~1U) | (enabled ? 1U : 0U), std::memory_order_relaxed);
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::FramePositionNormLimitsTask,
  task_priority_kinematic_control::TaskBase)
