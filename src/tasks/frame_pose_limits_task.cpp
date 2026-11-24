#include "task_priority_kinematic_control/tasks/frame_pose_limits_task.hpp"
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

constexpr const char * kAxisNames[6] = {"x", "y", "z", "rx", "ry", "rz"};

// Accepts 6 values, a single value broadcast to all axes, or empty (default).
FramePoseLimitsTask::Vector6d six_values(
  const std::vector<double> & values, const FramePoseLimitsTask::Vector6d & fallback,
  const std::string & name)
{
  if (values.empty()) {return fallback;}
  if (values.size() == 1) {return FramePoseLimitsTask::Vector6d::Constant(values.front());}
  if (values.size() != 6) {
    throw std::invalid_argument("FramePoseLimitsTask: '" + name + "' must contain 1 or 6 values");
  }
  FramePoseLimitsTask::Vector6d out;
  for (Eigen::Index i = 0; i < 6; ++i) {out(i) = values[static_cast<size_t>(i)];}
  return out;
}

// Parses the 6-DoF axis mask from a bool or 0/1 integer array. Empty keeps all axes.
bool parse_axes(const rclcpp::Parameter & parameter, std::array<bool, 6> & axes)
{
  std::vector<bool> values;
  if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL_ARRAY) {
    values = parameter.as_bool_array();
  } else if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY) {
    for (const auto value : parameter.as_integer_array()) {
      if (value != 0 && value != 1) {
        throw std::invalid_argument("FramePoseLimitsTask: axis mask integers must be 0 or 1");
      }
      values.push_back(value == 1);
    }
  } else if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
    return false;
  } else {
    throw std::invalid_argument("FramePoseLimitsTask: axis mask must be a bool or 0/1 integer array");
  }
  if (values.empty()) {return false;}
  if (values.size() != 6) {
    throw std::invalid_argument("FramePoseLimitsTask: axis mask must contain 6 values (x y z rx ry rz)");
  }
  for (size_t i = 0; i < 6; ++i) {axes[i] = values[i];}
  return true;
}

}  // namespace

void FramePoseLimitsTask::configure(
  const std::string & id, const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  if (!context.model) {throw std::invalid_argument("FramePoseLimitsTask requires a model");}
  frame_id_ = get_string_param(parameters, "frame_id", "");
  if (frame_id_.empty()) {frame_id_ = context.model->base_frame();}
  is_base_frame_ = frame_id_ == context.model->base_frame();

  axes_ = {{true, true, true, true, true, true}};
  bool parsed = false;
  if (const auto it = parameters.find("axes"); it != parameters.end()) {
    parsed = parse_axes(it->second, axes_);
  }
  if (!parsed) {
    if (const auto it = parameters.find("activation"); it != parameters.end()) {
      parse_axes(it->second, axes_);
    }
  }
  rows_ = 0;
  for (size_t axis = 0; axis < 6; ++axis) {
    axis_row_[axis] = axes_[axis] ? rows_++ : -1;
  }
  if (rows_ == 0) {throw std::invalid_argument("FramePoseLimitsTask: at least one axis must be enabled");}

  threshold_ = six_values(get_double_array_param(parameters, "threshold", {}),
      Vector6d::Constant(0.05), "threshold");
  alpha_ = six_values(get_double_array_param(parameters, "activation_margin", {}),
      threshold_ * 0.2, "activation_margin");
  delta_ = six_values(get_double_array_param(parameters, "release_margin", {}),
      threshold_ * 0.4, "release_margin");
  gains_ = six_values(get_double_array_param(parameters, "gain", {}), Vector6d::Ones(), "gain");
  eps_ = get_double_param(parameters, "eps", 1e-4);
  max_repulsive_velocity_ = get_double_param(parameters, "max_repulsive_velocity", 0.1);

  if (!threshold_.allFinite() || !alpha_.allFinite() || !delta_.allFinite() ||
    !gains_.allFinite() || !std::isfinite(eps_) || !std::isfinite(max_repulsive_velocity_))
  {
    throw std::invalid_argument("FramePoseLimitsTask: parameters must be finite");
  }
  if (eps_ < 0 || max_repulsive_velocity_ <= 0 || (gains_.array() < 0).any()) {
    throw std::invalid_argument(
            "FramePoseLimitsTask: require eps >= 0, max_repulsive_velocity > 0 and nonnegative gains");
  }
  const double pi = std::acos(-1.0);
  for (Eigen::Index axis = 0; axis < 6; ++axis) {
    if (!axes_[static_cast<size_t>(axis)]) {continue;}
    if (alpha_(axis) < 0 || delta_(axis) <= alpha_(axis) || threshold_(axis) <= delta_(axis) ||
      eps_ >= delta_(axis) - alpha_(axis))
    {
      throw std::invalid_argument(
              std::string("FramePoseLimitsTask: axis ") + kAxisNames[axis] +
              " requires 0 <= activation_margin < release_margin < threshold and eps < release-activation");
    }
    if (axis >= 3 && threshold_(axis) >= pi) {
      throw std::invalid_argument(
              std::string("FramePoseLimitsTask: rotation threshold on ") + kAxisNames[axis] +
              " must be below pi");
    }
  }

  dofs_ = static_cast<Eigen::Index>(context.model->total_dofs());
  frame_.jacobian.setZero(6, dofs_);
  local_jacobian_.setZero(6, dofs_);
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
            "FramePoseLimitsTask: publish_marker_rate must be finite, positive and representable");
  }
  snapshot_period_ns_ = std::max<int64_t>(1, static_cast<int64_t>(period_ns));
  has_snapshot_ = false;
  marker_valid_ = false;
  set_debug(get_bool_param(parameters, "debug", false));
  debug_channel_.buffer.initialize([&](FramePoseLimitsSnapshot & snapshot) {
      snapshot = FramePoseLimitsSnapshot{};
      snapshot.threshold = threshold_; snapshot.alpha = alpha_; snapshot.delta = delta_;
      snapshot.axes = axes_;
    });
  reset();
}

void FramePoseLimitsTask::prepare_computation(TaskComputation & out) const
{
  out.structural_zero_rows = true;
  out.jacobian.setZero(rows_, dofs_);
  out.error.setZero(rows_);
  out.desired_velocity.setZero(rows_);
  out.feedforward.resize(0);
  out.active = false;
  out.stop_arm_motion = false;
  out.has_frame_pose = false;
  out.has_updated_target = false;
  out.frame_id.reserve(frame_id_.size());
  out.frame_pose.setIdentity();
  out.status_message.reserve(32);
}

TaskComputation FramePoseLimitsTask::update(
  const WholeBodyState & state, const KinematicsBackend & backend)
{
  TaskComputation out;
  prepare_computation(out);
  update_into(state, backend, out);
  return out;
}

void FramePoseLimitsTask::update_into(
  const WholeBodyState & state, const KinematicsBackend & backend, TaskComputation & out)
{
  out.jacobian.setZero();
  out.error.setZero();
  out.desired_velocity.setZero();
  out.active = false;
  out.has_updated_target = false;
  last_error_.setZero();
  last_velocity_.setZero();
  marker_valid_ = false;
  if (!enabled_) {
    record_status(out, "disabled");
    return;
  }

  if (is_base_frame_) {
    frame_.pose = pose_math::base_pose_from_state(state);
    frame_.jacobian = pose_math::base_jacobian_for_model(*context_.model);
  } else if (!backend.read_frame_state(frame_id_, frame_)) {
    clear_hysteresis();
    record_status(out, "frame pose unavailable");
    return;
  }
  if (!frame_.pose.matrix().allFinite() || frame_.jacobian.rows() != 6 ||
    frame_.jacobian.cols() != dofs_ || !frame_.jacobian.allFinite())
  {
    clear_hysteresis();
    record_status(out, "frame pose unavailable");
    return;
  }
  out.has_frame_pose = true;
  out.frame_id = frame_id_;
  out.frame_pose = frame_.pose;

  const Eigen::Quaterniond current(frame_.pose.rotation());
  if (!has_target_) {
    // No goal received: hold the frame inside the box around where it is now.
    target_position_ = frame_.pose.translation();
    target_orientation_ = current.normalized();
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

  // Error and Jacobian expressed in the target frame axes.
  const Eigen::Matrix3d target_to_world = target_orientation_.toRotationMatrix();
  const Eigen::Matrix3d world_to_target = target_to_world.transpose();
  Vector6d error;
  error.head<3>() = world_to_target * (target_position_ - frame_.pose.translation());
  error.tail<3>() = world_to_target *
    pose_math::quaternion_error(target_orientation_, current.normalized());
  local_jacobian_.topRows<3>().noalias() = world_to_target * frame_.jacobian.topRows<3>();
  local_jacobian_.bottomRows<3>().noalias() = world_to_target * frame_.jacobian.bottomRows<3>();
  last_error_ = error;
  marker_valid_ = error.allFinite();
  marker_frame_position_ = frame_.pose.translation();

  for (size_t index = 0; index < 6; ++index) {
    if (!axes_[index]) {continue;}
    const auto axis = static_cast<Eigen::Index>(index);
    const double e = error(axis);
    const double activate_at = threshold_(axis) - alpha_(axis);
    const double hold_at = threshold_(axis) - delta_(axis);
    // Hysteresis: enter near the bound, leave only once back inside the inner band.
    if (!axis_active_[index] && std::abs(e) >= activate_at) {
      axis_active_[index] = true;
      axis_side_[index] = e >= 0 ? 1.0 : -1.0;
    }
    if (axis_active_[index] && axis_side_[index] * e <= hold_at + eps_) {
      axis_active_[index] = false;
      axis_side_[index] = 0.0;
    }
    if (!axis_active_[index]) {continue;}

    const Eigen::Index row = axis_row_[index];
    out.jacobian.row(row) = local_jacobian_.row(axis);
    if (out.jacobian.row(row).isZero(1e-12)) {
      // No DoF can move this axis; keep the row structural so the solver skips it.
      out.jacobian.row(row).setZero();
      continue;
    }
    // e = target - current: a positive task velocity reduces e, so drive e to the inner bound.
    out.error(row) = e - axis_side_[index] * hold_at;
    out.desired_velocity(row) = std::clamp(
      gains_(axis) * out.error(row), -max_repulsive_velocity_, max_repulsive_velocity_);
    last_velocity_(axis) = out.desired_velocity(row);
    out.active = true;
  }
  record_status(out, out.active ? "holding_bounds" : "inside_bounds");
}

void FramePoseLimitsTask::record_status(TaskComputation & out, const char * message)
{
  out.status_message = message;
  last_status_ = message;
  last_active_ = out.active;
}

bool FramePoseLimitsTask::set_pose_goal(const geometry_msgs::msg::PoseStamped & goal)
{
  if (!context_.model) {return false;}
  if (!goal.header.frame_id.empty() && goal.header.frame_id != context_.model->world_frame()) {
    return false;
  }
  const auto & p = goal.pose.position;
  const auto & o = goal.pose.orientation;
  Eigen::Quaterniond orientation(o.w, o.x, o.y, o.z);
  const Eigen::Vector3d position(p.x, p.y, p.z);
  if (!position.allFinite() || !orientation.coeffs().allFinite() || orientation.norm() < 1e-9) {
    return false;
  }
  orientation.normalize();
  pose_goal_ = goal;
  pose_goal_.header.frame_id = context_.model->world_frame();
  has_pose_goal_ = true;
  target_position_ = position;
  target_orientation_ = orientation;
  has_target_ = true;
  clear_hysteresis();
  return true;
}

bool FramePoseLimitsTask::set_gain(const std::vector<double> & gain, std::string & message)
{
  if (gain.size() != 6 && gain.size() != 1) {
    message = "Frame pose limits gain must contain 1 or 6 values";
    return false;
  }
  for (const double value : gain) {
    if (!std::isfinite(value) || value < 0.0) {
      message = "Frame pose limits gain values must be finite and non-negative";
      return false;
    }
  }
  gains_ = six_values(gain, gains_, "gain");
  message = "Frame pose limits gain updated";
  return true;
}

bool FramePoseLimitsTask::set_gain_scalar(double gain, std::string & message)
{
  if (!std::isfinite(gain) || gain < 0.0) {
    message = "Frame pose limits gain must be finite and non-negative";
    return false;
  }
  gains_.setConstant(gain);
  message = "Frame pose limits gain updated";
  return true;
}

bool FramePoseLimitsTask::set_enabled(bool enabled)
{
  if (enabled != enabled_) {
    TaskBaseCommon::set_enabled(enabled);
    reset();
  }
  return true;
}

void FramePoseLimitsTask::observe_command(const WholeBodyCommand &, int64_t timestamp_ns)
{
  const uint32_t visibility = marker_visibility_state();
  if (!(visibility & 1U) || timestamp_ns < 0) {return;}
  // Sample at marker frequency using the control timestamp (no wall clock and no
  // copy on every control update). Clock rewinds and debug toggles force a sample.
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
  snapshot.error = last_error_;
  snapshot.threshold = threshold_;
  snapshot.alpha = alpha_;
  snapshot.delta = delta_;
  snapshot.axes = axes_;
  snapshot.active = axis_active_;
  snapshot.timestamp_ns = timestamp_ns;
  snapshot.captured = std::chrono::steady_clock::now();
  debug_channel_.buffer.publish();
  has_snapshot_ = true;
  last_snapshot_ns_ = timestamp_ns;
  last_snapshot_visibility_ = visibility;
}

void FramePoseLimitsTask::set_debug(bool enabled)
{
  auto & visibility = debug_channel_.visibility;
  const uint32_t previous = visibility.load(std::memory_order_relaxed);
  if (((previous & 1U) != 0) == enabled) {return;}
  // The epoch prevents redisplaying snapshots from before a visualization toggle.
  visibility.store(((previous + 2U) & ~1U) | (enabled ? 1U : 0U), std::memory_order_relaxed);
}

void FramePoseLimitsTask::clear_hysteresis()
{
  axis_active_ = {{false, false, false, false, false, false}};
  axis_side_ = {{0, 0, 0, 0, 0, 0}};
}

void FramePoseLimitsTask::reset()
{
  clear_hysteresis();
  // A reset forgets any received goal; the next update captures the measured pose.
  has_pose_goal_ = false;
  has_target_ = false;
  last_active_ = false;
  last_error_.setZero();
  last_velocity_.setZero();
  last_status_ = enabled_ ? "inside_bounds" : "disabled";
}

msg::TaskStatus FramePoseLimitsTask::build_status() const
{
  auto status = TaskBaseCommon::build_status();
  status.active = enabled_ && last_active_;
  status.status_message = last_status_;
  status.target_type = "pose";
  status.error.assign(last_error_.data(), last_error_.data() + last_error_.size());
  status.command.assign(last_velocity_.data(), last_velocity_.data() + last_velocity_.size());
  return status;
}

std::vector<double> FramePoseLimitsTask::current_target() const
{
  if (!has_target_) {return {};}
  return {target_position_.x(), target_position_.y(), target_position_.z(),
    target_orientation_.x(), target_orientation_.y(), target_orientation_.z(),
    target_orientation_.w()};
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::FramePoseLimitsTask,
  task_priority_kinematic_control::TaskBase)
