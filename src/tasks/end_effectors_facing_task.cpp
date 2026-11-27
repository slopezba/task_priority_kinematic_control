#include "task_priority_kinematic_control/tasks/end_effectors_facing_task.hpp"

#include <pluginlib/class_list_macros.hpp>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace task_priority_kinematic_control
{

namespace
{

namespace facing = sura_manipulation_actions::facing;
using facing::wrapAngle;

std::vector<double> activation_values_from_parameters(
  const std::map<std::string, rclcpp::Parameter> & parameters)
{
  const auto it = parameters.find("activation");
  if (it == parameters.end()) {
    return {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  }
  if (it->second.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY) {
    const auto values = it->second.as_integer_array();
    return std::vector<double>(values.begin(), values.end());
  }
  if (it->second.get_type() == rclcpp::ParameterType::PARAMETER_BOOL_ARRAY) {
    const auto values = it->second.as_bool_array();
    std::vector<double> out;
    out.reserve(values.size());
    for (const bool value : values) {
      out.push_back(value ? 1.0 : 0.0);
    }
    return out;
  }
  return it->second.as_double_array();
}

bool has_parameter(
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const std::string & name)
{
  const auto it = parameters.find(name);
  return it != parameters.end() &&
         it->second.get_type() != rclcpp::ParameterType::PARAMETER_NOT_SET;
}

Eigen::Vector3d vector_from_values(const std::vector<double> & values)
{
  if (values.size() != 3) {
    return Eigen::Vector3d::Zero();
  }
  return Eigen::Vector3d(values[0], values[1], values[2]);
}

}  // namespace

bool EndEffectorsFacingTask::pure_axis_index(
  const Eigen::Vector3d & axis, int & index, double & sign, std::string & message)
{
  facing::PureAxis out;
  if (!facing::pureAxis(axis, out, message)) {return false;}
  index = out.index;
  sign = out.sign;
  return true;
}

void EndEffectorsFacingTask::restore_goal_reference()
{
  left_reference_axis_ = left_goal_.vector();
  right_reference_axis_ = right_goal_.vector();
  feedforward_.setZero();
}

facing::Reference EndEffectorsFacingTask::reference() const
{
  facing::Reference out;
  out.left_axis = left_reference_axis_;
  out.right_axis = right_reference_axis_;
  out.roll = target_roll_;
  out.distance = target_distance_;
  return out;
}

bool EndEffectorsFacingTask::set_enabled(bool enabled)
{
  feedforward_.setZero();
  return TaskBaseCommon::set_enabled(enabled);
}

void EndEffectorsFacingTask::reset() {restore_goal_reference();}

void EndEffectorsFacingTask::configure(
  const std::string & id,
  const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  left_frame_ = get_string_param(parameters, "left_frame", context.model->left_tip_frame());
  right_frame_ = get_string_param(parameters, "right_frame", context.model->right_tip_frame());
  min_distance_ = get_double_param(parameters, "min_distance", 0.02);

  const auto gains = get_double_array_param(parameters, "gain", {0.5, 0.5, 0.5, 0.5, 0.5, 0.5});
  for (size_t i = 0; i < 6 && i < gains.size(); ++i) {
    gains_(static_cast<Eigen::Index>(i)) = gains[i];
  }
  const auto activation = activation_values_from_parameters(parameters);
  for (size_t i = 0; i < 6 && i < activation.size(); ++i) {
    activation_(static_cast<Eigen::Index>(i)) = activation[i] != 0.0;
  }

  std::string message;
  const Eigen::Vector3d left_axis = vector_from_values(
    get_double_array_param(parameters, "left_pointing_axis", {0.0, 0.0, 1.0}));
  const Eigen::Vector3d right_axis = vector_from_values(
    get_double_array_param(parameters, "right_pointing_axis", {0.0, 0.0, 1.0}));
  if (!facing::pureAxis(left_axis, left_goal_, message)) {
    throw std::invalid_argument("Task '" + id + "': left_pointing_axis: " + message);
  }
  if (!facing::pureAxis(right_axis, right_goal_, message)) {
    throw std::invalid_argument("Task '" + id + "': right_pointing_axis: " + message);
  }
  restore_goal_reference();
  // The hosts declare feedforward_timeout = 0 for every task. A streamed facing
  // feedforward always expires, so zero selects the default.
  feedforward_timeout_ = get_double_param(parameters, "feedforward_timeout", 0.0);
  if (!std::isfinite(feedforward_timeout_) || feedforward_timeout_ < 0.0) {
    throw std::invalid_argument(
            "Task '" + id + "': feedforward_timeout must be finite and non-negative");
  }
  if (feedforward_timeout_ == 0.0) {feedforward_timeout_ = 0.5;}

  capture_roll_pending_ = true;
  if (has_parameter(parameters, "roll_angle")) {
    const double roll = get_double_param(parameters, "roll_angle", 0.0);
    if (std::isfinite(roll)) {
      target_roll_ = wrapAngle(roll);
      capture_roll_pending_ = false;
    }
  }
  capture_distance_pending_ = true;
  if (has_parameter(parameters, "distance")) {
    const double distance = get_double_param(parameters, "distance", 0.0);
    if (std::isfinite(distance) && distance > 0.0) {
      target_distance_ = distance;
      capture_distance_pending_ = false;
    }
  }
}

EndEffectorsFacingTask::Measurement EndEffectorsFacingTask::measure(
  const KinematicsBackend & backend) const
{
  Measurement out;
  const FrameState left = backend.get_frame_state(left_frame_);
  const FrameState right = backend.get_frame_state(right_frame_);
  const Eigen::Vector3d delta = right.pose.translation() - left.pose.translation();
  out.distance = delta.norm();
  out.valid = out.distance >= min_distance_ && out.distance > 0.0;
  if (!out.valid) {
    return out;
  }
  const Eigen::Vector3d d = delta / out.distance;
  const auto roll = facing::rollGeometry(
    d,
    facing::rollReferenceAxis(left.pose.rotation(), left_goal_),
    facing::rollReferenceAxis(right.pose.rotation(), right_goal_));
  out.roll_valid = roll.valid;
  out.roll = roll.roll;
  return out;
}

TaskComputation EndEffectorsFacingTask::update(
  const WholeBodyState &,
  const KinematicsBackend & backend)
{
  TaskComputation computation;
  computation.active = enabled_;
  if (!enabled_) {
    computation.status_message = "disabled";
    return computation;
  }

  const FrameState left = backend.get_frame_state(left_frame_);
  const FrameState right = backend.get_frame_state(right_frame_);
  computation.has_frame_pose = true;
  computation.frame_id = right_frame_;
  computation.frame_pose = left.pose.inverse() * right.pose;

  const Eigen::Index cols = left.jacobian.cols();
  if (right.jacobian.cols() != cols || left.jacobian.rows() < 6 || right.jacobian.rows() < 6) {
    computation.active = false;
    computation.status_message = "invalid jacobians";
    return computation;
  }

  const Eigen::Vector3d delta = right.pose.translation() - left.pose.translation();
  const double r = delta.norm();
  measured_distance_ = r;
  if (r < min_distance_ || r <= 0.0) {
    computation.active = false;
    computation.status_message = "ee too close";
    return computation;
  }
  auto rows_now = facing::evaluate(
    left.pose, right.pose, left.jacobian, right.jacobian, left_goal_, right_goal_,
    reference(), min_distance_);
  if (rows_now.roll_valid) {
    measured_roll_ = rows_now.roll;
    if (capture_roll_pending_) {
      target_roll_ = rows_now.roll;
      capture_roll_pending_ = false;
      rows_now.error(4) = 0.0;
    } else {
      rows_now.error(4) = wrapAngle(target_roll_ - rows_now.roll);
    }
  } else {
    rows_now.error(4) = 0.0;
  }
  if (capture_distance_pending_) {
    target_distance_ = r;
    capture_distance_pending_ = false;
    rows_now.error(5) = 0.0;
  }
  const Eigen::Matrix<double, 6, 1> full_error = rows_now.error;
  const Eigen::MatrixXd & full_jacobian = rows_now.jacobian;
  if (std::chrono::duration<double>(steady_now() - feedforward_time_).count() > feedforward_timeout_) {
    feedforward_.setZero();
  }
  Eigen::Matrix<double, 6, 1> feedforward = feedforward_;
  // An undefined roll is held, so it cannot take a feedforward either.
  if (!rows_now.roll_valid) {feedforward(4) = 0.0;}

  std::vector<Eigen::Index> active_indices;
  active_indices.reserve(6);
  for (Eigen::Index i = 0; i < 6; ++i) {
    if (activation_(i)) {
      active_indices.push_back(i);
    }
  }
  if (active_indices.empty()) {
    computation.active = false;
    computation.status_message = "no active axes";
    return computation;
  }

  const auto rows = static_cast<Eigen::Index>(active_indices.size());
  computation.error = Eigen::VectorXd::Zero(rows);
  computation.feedforward = Eigen::VectorXd::Zero(rows);
  computation.desired_velocity = Eigen::VectorXd::Zero(rows);
  computation.jacobian = Eigen::MatrixXd::Zero(rows, cols);
  for (Eigen::Index row = 0; row < rows; ++row) {
    const Eigen::Index source_row = active_indices[static_cast<size_t>(row)];
    computation.error(row) = full_error(source_row);
    computation.feedforward(row) = feedforward(source_row);
    computation.desired_velocity(row) =
      feedforward(source_row) + gains_(source_row) * full_error(source_row);
    computation.jacobian.row(row) = full_jacobian.row(source_row);
  }

  std::ostringstream status;
  status.setf(std::ios::fixed);
  status.precision(3);
  status << "tracking roll=" << measured_roll_ << (rows_now.roll_valid ? "" : "(undefined)")
         << " distance=" << r;
  computation.status_message = status.str();
  return computation;
}

bool EndEffectorsFacingTask::set_facing_target(
  const msg::FacingTarget & target, std::string & message)
{
  const Eigen::Vector3d left_axis(
    target.left_pointing_axis.x, target.left_pointing_axis.y, target.left_pointing_axis.z);
  const Eigen::Vector3d right_axis(
    target.right_pointing_axis.x, target.right_pointing_axis.y, target.right_pointing_axis.z);
  facing::PureAxis left_goal, right_goal;
  if (!facing::pureAxis(left_axis, left_goal, message)) {
    message = "left_pointing_axis: " + message;
    return false;
  }
  if (!facing::pureAxis(right_axis, right_goal, message)) {
    message = "right_pointing_axis: " + message;
    return false;
  }
  if (!std::isfinite(target.distance) || target.distance <= 0.0) {
    message = "distance must be finite and > 0";
    return false;
  }
  if (std::isinf(target.roll_angle)) {
    message = "roll_angle must be finite or NaN (keep current roll)";
    return false;
  }

  left_goal_ = left_goal;
  right_goal_ = right_goal;
  restore_goal_reference();
  target_distance_ = target.distance;
  capture_distance_pending_ = false;
  if (std::isnan(target.roll_angle)) {
    capture_roll_pending_ = true;
  } else {
    target_roll_ = wrapAngle(target.roll_angle);
    capture_roll_pending_ = false;
  }
  message = "Facing target updated";
  return true;
}

bool EndEffectorsFacingTask::set_facing_reference(
  const sura_manipulation_actions::msg::FacingReference & message_in, std::string & message)
{
  const auto vector = [](const geometry_msgs::msg::Vector3 & value) {
      return Eigen::Vector3d(value.x, value.y, value.z);
    };
  facing::PureAxis left_goal, right_goal;
  if (!facing::pureAxis(vector(message_in.left_goal_axis), left_goal, message) ||
    !facing::pureAxis(vector(message_in.right_goal_axis), right_goal, message))
  {return false;}
  facing::Reference next;
  next.left_axis = vector(message_in.left_pointing_axis);
  next.right_axis = vector(message_in.right_pointing_axis);
  next.roll = message_in.roll_angle;
  next.distance = message_in.distance;
  facing::ReferenceRate rate;
  rate.left_axis = vector(message_in.left_pointing_rate);
  rate.right_axis = vector(message_in.right_pointing_rate);
  rate.roll = message_in.roll_rate;
  rate.distance = message_in.distance_rate;
  if (!next.left_axis.allFinite() || !next.right_axis.allFinite() ||
    std::abs(next.left_axis.norm() - 1.0) > 1e-3 || std::abs(next.right_axis.norm() - 1.0) > 1e-3)
  {
    message = "reference pointing axes must be unit vectors";
    return false;
  }
  if (!std::isfinite(next.roll) || !std::isfinite(next.distance) || next.distance <= 0.0 ||
    !rate.left_axis.allFinite() || !rate.right_axis.allFinite() ||
    !std::isfinite(rate.roll) || !std::isfinite(rate.distance))
  {
    message = "reference roll, distance and rates must be finite, with distance > 0";
    return false;
  }
  next.left_axis.normalize();
  next.right_axis.normalize();
  left_goal_ = left_goal;
  right_goal_ = right_goal;
  left_reference_axis_ = next.left_axis;
  right_reference_axis_ = next.right_axis;
  target_roll_ = wrapAngle(next.roll);
  target_distance_ = next.distance;
  capture_roll_pending_ = false;
  capture_distance_pending_ = false;
  feedforward_ = enabled_ ?
    facing::feedforwardRows(next, rate) : Eigen::Matrix<double, 6, 1>::Zero();
  feedforward_time_ = steady_now();
  message = "Facing reference updated";
  return true;
}

void EndEffectorsFacingTask::capture_measured_target(const KinematicsBackend & backend)
{
  restore_goal_reference();
  const auto measured = measure(backend);
  if (measured.valid) {
    target_distance_ = measured.distance;
    capture_distance_pending_ = false;
  } else {
    capture_distance_pending_ = true;
  }
  if (measured.valid && measured.roll_valid) {
    target_roll_ = measured.roll;
    capture_roll_pending_ = false;
  } else {
    capture_roll_pending_ = true;
  }
}

bool EndEffectorsFacingTask::set_gain(const std::vector<double> & gain, std::string & message)
{
  if (gain.size() != 6) {
    message = "Facing gain must contain 6 values";
    return false;
  }
  for (size_t i = 0; i < gain.size(); ++i) {
    if (gain[i] < 0.0) {
      message = "Facing gain values must be non-negative";
      return false;
    }
  }
  for (size_t i = 0; i < gain.size(); ++i) {
    gains_(static_cast<Eigen::Index>(i)) = gain[i];
  }
  message = "Facing gain updated";
  return true;
}

msg::TaskStatus EndEffectorsFacingTask::build_status() const
{
  auto status = TaskBaseCommon::build_status();
  status.target_type = "facing";
  return status;
}

std::vector<double> EndEffectorsFacingTask::current_target() const
{
  return {
    left_reference_axis_.x(), left_reference_axis_.y(), left_reference_axis_.z(),
    right_reference_axis_.x(), right_reference_axis_.y(), right_reference_axis_.z(),
    target_roll_, target_distance_};
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::EndEffectorsFacingTask,
  task_priority_kinematic_control::TaskBase)
