#include "task_priority_kinematic_control/tasks/end_effectors_facing_task.hpp"

#include <pluginlib/class_list_macros.hpp>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace task_priority_kinematic_control
{

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kPureAxisTolerance = 1e-3;
constexpr double kAlignedEpsilon = 1e-9;
// x^2 + y^2 of the roll atan2 equals |proj(b_left)|^2 * |proj(b_right)|^2.
constexpr double kRollMinSquaredNorm = 1e-4;

double wrap_angle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

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

// World-frame reference axis of an EE for the roll zero (cyclic: x->y, y->z, z->x).
Eigen::Vector3d roll_reference_axis(const Eigen::Matrix3d & rotation, int index, double sign)
{
  return sign * rotation.col((index + 1) % 3);
}

struct RollGeometry
{
  bool valid = false;
  double roll = 0.0;
  double x = 0.0;
  double y = 0.0;
};

RollGeometry roll_geometry(
  const Eigen::Vector3d & d, const Eigen::Vector3d & b_left, const Eigen::Vector3d & b_right)
{
  RollGeometry out;
  out.y = d.dot(b_left.cross(b_right));
  out.x = b_left.dot(b_right) - d.dot(b_left) * d.dot(b_right);
  out.valid = out.x * out.x + out.y * out.y > kRollMinSquaredNorm;
  out.roll = out.valid ? std::atan2(out.y, out.x) : 0.0;
  return out;
}

}  // namespace

bool EndEffectorsFacingTask::pure_axis_index(
  const Eigen::Vector3d & axis, int & index, double & sign, std::string & message)
{
  if (!axis.allFinite()) {
    message = "pointing axis must be finite";
    return false;
  }
  Eigen::Index max_index = 0;
  axis.cwiseAbs().maxCoeff(&max_index);
  const double value = axis(max_index);
  for (Eigen::Index i = 0; i < 3; ++i) {
    const double expected = i == max_index ? 1.0 : 0.0;
    if (std::abs(std::abs(axis(i)) - expected) > kPureAxisTolerance) {
      std::ostringstream stream;
      stream << "pointing axis [" << axis.transpose()
             << "] must be a pure signed axis (+-x, +-y or +-z)";
      message = stream.str();
      return false;
    }
  }
  index = static_cast<int>(max_index);
  sign = value > 0.0 ? 1.0 : -1.0;
  return true;
}

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
  if (!pure_axis_index(left_axis, left_axis_index_, left_axis_sign_, message)) {
    throw std::invalid_argument("Task '" + id + "': left_pointing_axis: " + message);
  }
  if (!pure_axis_index(right_axis, right_axis_index_, right_axis_sign_, message)) {
    throw std::invalid_argument("Task '" + id + "': right_pointing_axis: " + message);
  }
  left_pointing_axis_ = left_axis_sign_ * Eigen::Vector3d::Unit(left_axis_index_);
  right_pointing_axis_ = right_axis_sign_ * Eigen::Vector3d::Unit(right_axis_index_);

  capture_roll_pending_ = true;
  if (has_parameter(parameters, "roll_angle")) {
    const double roll = get_double_param(parameters, "roll_angle", 0.0);
    if (std::isfinite(roll)) {
      target_roll_ = wrap_angle(roll);
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
  const auto roll = roll_geometry(
    d,
    roll_reference_axis(left.pose.rotation(), left_axis_index_, left_axis_sign_),
    roll_reference_axis(right.pose.rotation(), right_axis_index_, right_axis_sign_));
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
  const Eigen::Vector3d d = delta / r;

  const Eigen::MatrixXd left_linear = left.jacobian.topRows(3);
  const Eigen::MatrixXd left_angular = left.jacobian.middleRows(3, 3);
  const Eigen::MatrixXd right_linear = right.jacobian.topRows(3);
  const Eigen::MatrixXd right_angular = right.jacobian.middleRows(3, 3);
  const Eigen::MatrixXd relative_linear = right_linear - left_linear;
  // d_dot = P (v_right - v_left), P = (I - d d^T) / r.
  const Eigen::MatrixXd d_jacobian =
    ((Eigen::Matrix3d::Identity() - d * d.transpose()) / r) * relative_linear;
  // Angular velocity of the line direction: omega_d = d x d_dot.
  const Eigen::MatrixXd line_angular = skew(d) * d_jacobian;

  const Eigen::Matrix3d left_rotation = left.pose.rotation();
  const Eigen::Matrix3d right_rotation = right.pose.rotation();

  Eigen::Matrix<double, 6, 1> full_error = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::MatrixXd full_jacobian = Eigen::MatrixXd::Zero(6, cols);

  // Pointing rows: rotation vector that takes the EE axis a onto the line direction s,
  // projected on the two other EE axes. Current-rate jacobian: u^T (omega_ee - omega_line).
  const auto pointing_rows = [&](
    const Eigen::Matrix3d & rotation, int index, double sign, const Eigen::Vector3d & s,
    const Eigen::MatrixXd & angular, Eigen::Index row)
    {
      const Eigen::Vector3d a = sign * rotation.col(index);
      const Eigen::Vector3d u1 = rotation.col((index + 1) % 3);
      const Eigen::Vector3d u2 = rotation.col((index + 2) % 3);
      const Eigen::Vector3d c = a.cross(s);
      const double sin_angle = c.norm();
      const double cos_angle = a.dot(s);
      Eigen::Vector3d rotation_vector = Eigen::Vector3d::Zero();
      if (sin_angle > kAlignedEpsilon) {
        rotation_vector = std::atan2(sin_angle, cos_angle) * c / sin_angle;
      } else if (cos_angle < 0.0) {
        rotation_vector = kPi * u1;
      }
      full_error(row) = u1.dot(rotation_vector);
      full_error(row + 1) = u2.dot(rotation_vector);
      const Eigen::MatrixXd relative_angular = angular - line_angular;
      full_jacobian.row(row) = u1.transpose() * relative_angular;
      full_jacobian.row(row + 1) = u2.transpose() * relative_angular;
    };
  pointing_rows(left_rotation, left_axis_index_, left_axis_sign_, d, left_angular, 0);
  pointing_rows(right_rotation, right_axis_index_, right_axis_sign_, -d, right_angular, 2);

  // Roll row: phi = atan2(y, x), y = d.(bL x bR), x = bL.bR - (d.bL)(d.bR).
  const Eigen::Vector3d b_left =
    roll_reference_axis(left_rotation, left_axis_index_, left_axis_sign_);
  const Eigen::Vector3d b_right =
    roll_reference_axis(right_rotation, right_axis_index_, right_axis_sign_);
  const auto roll = roll_geometry(d, b_left, b_right);
  if (roll.valid) {
    measured_roll_ = roll.roll;
    if (capture_roll_pending_) {
      target_roll_ = roll.roll;
      capture_roll_pending_ = false;
    }
    const double alpha_left = d.dot(b_left);
    const double alpha_right = d.dot(b_right);
    const Eigen::Vector3d bl_x_br = b_left.cross(b_right);
    const Eigen::RowVectorXd dy =
      bl_x_br.transpose() * d_jacobian +
      b_left.cross(b_right.cross(d)).transpose() * left_angular +
      b_right.cross(d.cross(b_left)).transpose() * right_angular;
    const Eigen::RowVectorXd dx =
      -(alpha_right * b_left + alpha_left * b_right).transpose() * d_jacobian +
      (bl_x_br - alpha_right * b_left.cross(d)).transpose() * left_angular +
      (-bl_x_br - alpha_left * b_right.cross(d)).transpose() * right_angular;
    const double denominator = roll.x * roll.x + roll.y * roll.y;
    full_error(4) = capture_roll_pending_ ? 0.0 : wrap_angle(target_roll_ - roll.roll);
    full_jacobian.row(4) = (roll.x * dy - roll.y * dx) / denominator;
  } else {
    // Roll undefined while a reference axis is parallel to the line: hold, keep the main term.
    full_error(4) = 0.0;
    full_jacobian.row(4) = d.transpose() * (right_angular - left_angular);
  }

  // Distance row.
  if (capture_distance_pending_) {
    target_distance_ = r;
    capture_distance_pending_ = false;
  }
  full_error(5) = target_distance_ - r;
  full_jacobian.row(5) = d.transpose() * relative_linear;

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
  computation.desired_velocity = Eigen::VectorXd::Zero(rows);
  computation.jacobian = Eigen::MatrixXd::Zero(rows, cols);
  for (Eigen::Index row = 0; row < rows; ++row) {
    const Eigen::Index source_row = active_indices[static_cast<size_t>(row)];
    computation.error(row) = full_error(source_row);
    computation.desired_velocity(row) = gains_(source_row) * full_error(source_row);
    computation.jacobian.row(row) = full_jacobian.row(source_row);
  }

  std::ostringstream status;
  status.setf(std::ios::fixed);
  status.precision(3);
  status << "tracking roll=" << measured_roll_ << (roll.valid ? "" : "(undefined)")
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
  int left_index = 0;
  int right_index = 0;
  double left_sign = 1.0;
  double right_sign = 1.0;
  if (!pure_axis_index(left_axis, left_index, left_sign, message)) {
    message = "left_pointing_axis: " + message;
    return false;
  }
  if (!pure_axis_index(right_axis, right_index, right_sign, message)) {
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

  left_axis_index_ = left_index;
  right_axis_index_ = right_index;
  left_axis_sign_ = left_sign;
  right_axis_sign_ = right_sign;
  left_pointing_axis_ = left_sign * Eigen::Vector3d::Unit(left_index);
  right_pointing_axis_ = right_sign * Eigen::Vector3d::Unit(right_index);
  target_distance_ = target.distance;
  capture_distance_pending_ = false;
  if (std::isnan(target.roll_angle)) {
    capture_roll_pending_ = true;
  } else {
    target_roll_ = wrap_angle(target.roll_angle);
    capture_roll_pending_ = false;
  }
  message = "Facing target updated";
  return true;
}

void EndEffectorsFacingTask::capture_measured_target(const KinematicsBackend & backend)
{
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
    left_pointing_axis_.x(), left_pointing_axis_.y(), left_pointing_axis_.z(),
    right_pointing_axis_.x(), right_pointing_axis_.y(), right_pointing_axis_.z(),
    target_roll_, target_distance_};
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::EndEffectorsFacingTask,
  task_priority_kinematic_control::TaskBase)
