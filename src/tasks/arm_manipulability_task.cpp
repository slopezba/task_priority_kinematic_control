#include "task_priority_kinematic_control/tasks/arm_manipulability_task.hpp"

#include <pluginlib/class_list_macros.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace task_priority_kinematic_control
{

void ArmManipulabilityTask::configure(
  const std::string & id, const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  if (!context.model) {throw std::invalid_argument("ArmManipulabilityTask requires a model");}
  const auto & model = *context.model;

  // 'arm' only provides defaults; frame_id / joint_names override it.
  const std::string arm = get_string_param(parameters, "arm", "");
  std::string default_frame;
  std::vector<std::string> default_joints;
  if (arm == "left") {
    default_frame = model.left_tip_frame();
    default_joints = model.left_arm_joints();
  } else if (arm == "right") {
    default_frame = model.right_tip_frame();
    default_joints = model.right_arm_joints();
  } else if (!arm.empty()) {
    throw std::invalid_argument("ArmManipulabilityTask: arm must be 'left' or 'right'");
  }
  // The controller auto-declares frame_id/joint_names empty: empty means "use the arm default".
  frame_id_ = get_string_param(parameters, "frame_id", default_frame);
  if (frame_id_.empty()) {frame_id_ = default_frame;}
  joint_names_ = get_string_array_param(parameters, "joint_names", default_joints);
  if (joint_names_.empty()) {joint_names_ = default_joints;}
  if (frame_id_.empty() || joint_names_.empty()) {
    throw std::invalid_argument(
            "ArmManipulabilityTask: set 'arm' or both 'frame_id' and 'joint_names'");
  }

  joint_indices_.clear();
  columns_.clear();
  for (const auto & name : joint_names_) {
    const int index = model.joint_index(name);
    if (index < 0) {
      throw std::invalid_argument("ArmManipulabilityTask: unknown joint '" + name + "'");
    }
    joint_indices_.push_back(index);
    columns_.push_back(static_cast<Eigen::Index>(model.base_dofs()) + index);
  }

  w_min_ = get_double_param(parameters, "w_min", 0.0025);
  w_des_ = get_double_param(parameters, "w_des", 0.0045);
  gain_ = get_double_param(parameters, "gain_scalar", 1.0);
  max_velocity_ = get_double_param(parameters, "max_velocity", 0.3);
  fd_step_ = get_double_param(parameters, "fd_step", 1e-4);
  gradient_epsilon_ = get_double_param(parameters, "gradient_epsilon", 1e-9);
  for (const double value : {w_min_, w_des_, gain_, max_velocity_, fd_step_, gradient_epsilon_}) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("ArmManipulabilityTask: parameters must be finite");
    }
  }
  if (w_min_ <= 0.0 || w_des_ <= w_min_) {
    throw std::invalid_argument("ArmManipulabilityTask: require 0 < w_min < w_des");
  }
  if (gain_ < 0.0 || max_velocity_ <= 0.0 || fd_step_ <= 0.0 || gradient_epsilon_ < 0.0) {
    throw std::invalid_argument(
            "ArmManipulabilityTask: require gain_scalar >= 0, max_velocity > 0, fd_step > 0 "
            "and gradient_epsilon >= 0");
  }

  dofs_ = static_cast<Eigen::Index>(model.total_dofs());
  frame_.jacobian.setZero(6, dofs_);
  gradient_.setZero(static_cast<Eigen::Index>(joint_names_.size()));
  last_status_.reserve(32);

  publish_marker_rate_ = get_double_param(parameters, "publish_marker_rate", 2.0);
  ellipsoid_scale_ = get_double_param(parameters, "ellipsoid_scale", 0.5);
  const double period_ns = 1e9 / publish_marker_rate_;
  if (!std::isfinite(publish_marker_rate_) || publish_marker_rate_ <= 0 ||
    !std::isfinite(period_ns) || period_ns < 1 ||
    period_ns >= static_cast<double>(std::numeric_limits<int64_t>::max()))
  {
    throw std::invalid_argument(
            "ArmManipulabilityTask: publish_marker_rate must be finite, positive and representable");
  }
  if (!std::isfinite(ellipsoid_scale_) || ellipsoid_scale_ <= 0.0) {
    throw std::invalid_argument("ArmManipulabilityTask: ellipsoid_scale must be finite and positive");
  }
  snapshot_period_ns_ = std::max<int64_t>(1, static_cast<int64_t>(period_ns));
  has_snapshot_ = false;
  marker_valid_ = false;
  set_debug(get_bool_param(parameters, "debug", false));
  debug_channel_.buffer.initialize([&](ArmManipulabilitySnapshot & snapshot) {
      snapshot = ArmManipulabilitySnapshot{};
      snapshot.w_min = w_min_; snapshot.w_des = w_des_; snapshot.ellipsoid_scale = ellipsoid_scale_;
    });
  reset();
}

void ArmManipulabilityTask::prepare_computation(TaskComputation & out) const
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
  out.frame_pose.setIdentity();
  out.status_message.reserve(32);
}

TaskComputation ArmManipulabilityTask::update(
  const WholeBodyState & state, const KinematicsBackend & backend)
{
  TaskComputation out;
  prepare_computation(out);
  update_into(state, backend, out);
  return out;
}

double ArmManipulabilityTask::manipulability(const Eigen::MatrixXd & jacobian) const
{
  if (jacobian.rows() < 3) {return 0.0;}
  Eigen::Matrix3d jjt = Eigen::Matrix3d::Zero();
  for (const auto col : columns_) {
    if (col >= jacobian.cols()) {return 0.0;}
    const Eigen::Vector3d column = jacobian.block<3, 1>(0, col);
    jjt.noalias() += column * column.transpose();
  }
  return std::sqrt(std::max(0.0, jjt.determinant()));
}

bool ArmManipulabilityTask::compute_gradient(
  const WholeBodyState & state, const KinematicsBackend & backend)
{
  perturbed_ = state;
  for (size_t k = 0; k < joint_indices_.size(); ++k) {
    const Eigen::Index index = joint_indices_[k];
    const double q = state.joint_positions(index);
    perturbed_.joint_positions(index) = q + fd_step_;
    const double w_plus = manipulability(backend.compute_frame_state(perturbed_, frame_id_).jacobian);
    perturbed_.joint_positions(index) = q - fd_step_;
    const double w_minus = manipulability(backend.compute_frame_state(perturbed_, frame_id_).jacobian);
    perturbed_.joint_positions(index) = q;
    gradient_(static_cast<Eigen::Index>(k)) = (w_plus - w_minus) / (2.0 * fd_step_);
  }
  return gradient_.allFinite();
}

void ArmManipulabilityTask::update_into(
  const WholeBodyState & state, const KinematicsBackend & backend, TaskComputation & out)
{
  out.jacobian.setZero();
  out.error.setZero();
  out.desired_velocity.setZero();
  out.active = false;
  last_velocity_ = 0.0;
  marker_valid_ = false;
  if (!enabled_) {
    row_active_ = false;
    record_status(out, "disabled");
    return;
  }
  for (const auto index : joint_indices_) {
    if (index >= state.joint_positions.size()) {
      row_active_ = false;
      record_status(out, "waiting_for_joints");
      return;
    }
  }
  if (!backend.read_frame_state(frame_id_, frame_) || frame_.jacobian.rows() != 6 ||
    frame_.jacobian.cols() != dofs_ || !frame_.jacobian.allFinite())
  {
    row_active_ = false;
    record_status(out, "frame_unavailable");
    return;
  }
  out.has_frame_pose = true;
  out.frame_id = frame_id_;
  out.frame_pose = frame_.pose;

  const double w = manipulability(frame_.jacobian);
  last_w_ = w;
  if (debug()) {capture_marker_geometry();}
  if (!row_active_ && w < w_min_) {row_active_ = true;}
  if (row_active_ && w >= w_des_) {row_active_ = false;}
  if (!row_active_) {
    record_status(out, "well_conditioned");
    return;
  }

  try {
    if (!compute_gradient(state, backend)) {
      record_status(out, "invalid_gradient");
      return;
    }
  } catch (const std::exception &) {
    row_active_ = false;
    record_status(out, "backend_unsupported");
    return;
  }
  const double gradient_norm = gradient_.norm();
  if (gradient_norm <= gradient_epsilon_) {
    // Exactly on a symmetric singularity the ascent direction is undefined.
    record_status(out, "zero_gradient");
    return;
  }
  for (size_t k = 0; k < columns_.size(); ++k) {
    out.jacobian(0, columns_[k]) = gradient_(static_cast<Eigen::Index>(k)) / gradient_norm;
  }
  const double normalized_error = (w_des_ - w) / w_des_;
  out.error(0) = normalized_error;
  out.desired_velocity(0) = std::clamp(gain_ * normalized_error, 0.0, max_velocity_);
  last_velocity_ = out.desired_velocity(0);
  out.active = true;
  record_status(out, "raising_manipulability");
}

void ArmManipulabilityTask::capture_marker_geometry()
{
  // Ellipsoid of tip velocities for unit joint speed: axes = eigenvectors of Jp Jp^T,
  // radii = singular values sqrt(eigenvalues). Fixed-size, no allocation.
  Eigen::Matrix3d jjt = Eigen::Matrix3d::Zero();
  for (const auto col : columns_) {
    const Eigen::Vector3d column = frame_.jacobian.block<3, 1>(0, col);
    jjt.noalias() += column * column.transpose();
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(jjt);
  if (solver.info() != Eigen::Success) {return;}
  Eigen::Matrix3d axes = solver.eigenvectors();
  if (axes.determinant() < 0.0) {axes.col(2) = -axes.col(2);}  // proper rotation
  marker_orientation_ = Eigen::Quaterniond(axes).normalized();
  marker_singular_values_ = solver.eigenvalues().cwiseMax(0.0).cwiseSqrt();
  marker_tip_position_ = frame_.pose.translation();
  marker_valid_ = marker_orientation_.coeffs().allFinite() && marker_singular_values_.allFinite() &&
    marker_tip_position_.allFinite();
}

void ArmManipulabilityTask::observe_command(const WholeBodyCommand &, int64_t timestamp_ns)
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
  snapshot.valid = enabled_ && marker_valid_;
  snapshot.tip_position = marker_tip_position_;
  snapshot.orientation = marker_orientation_;
  snapshot.singular_values = marker_singular_values_;
  snapshot.w = last_w_;
  snapshot.w_min = w_min_;
  snapshot.w_des = w_des_;
  snapshot.ellipsoid_scale = ellipsoid_scale_;
  snapshot.active = last_active_;
  snapshot.timestamp_ns = timestamp_ns;
  snapshot.captured = std::chrono::steady_clock::now();
  debug_channel_.buffer.publish();
  has_snapshot_ = true;
  last_snapshot_ns_ = timestamp_ns;
  last_snapshot_visibility_ = visibility;
}

void ArmManipulabilityTask::set_debug(bool enabled)
{
  auto & visibility = debug_channel_.visibility;
  const uint32_t previous = visibility.load(std::memory_order_relaxed);
  if (((previous & 1U) != 0) == enabled) {return;}
  // The epoch prevents redisplaying snapshots from before a visualization toggle.
  visibility.store(((previous + 2U) & ~1U) | (enabled ? 1U : 0U), std::memory_order_relaxed);
}

void ArmManipulabilityTask::record_status(TaskComputation & out, const char * message)
{
  out.status_message = message;
  last_status_ = message;
  last_active_ = out.active;
}

bool ArmManipulabilityTask::set_gain(const std::vector<double> & gain, std::string & message)
{
  if (gain.size() != 1) {
    message = "Manipulability gain must contain 1 value";
    return false;
  }
  return set_gain_scalar(gain.front(), message);
}

bool ArmManipulabilityTask::set_gain_scalar(double gain, std::string & message)
{
  if (!std::isfinite(gain) || gain < 0.0) {
    message = "Manipulability gain must be finite and non-negative";
    return false;
  }
  gain_ = gain;
  message = "Manipulability gain updated";
  return true;
}

void ArmManipulabilityTask::reset()
{
  TaskBaseCommon::reset();
  row_active_ = false;
  last_active_ = false;
  last_velocity_ = 0.0;
}

msg::TaskStatus ArmManipulabilityTask::build_status() const
{
  auto status = TaskBaseCommon::build_status();
  status.active = enabled_ && last_active_;
  status.status_message = last_status_;
  status.joint_names = joint_names_;
  status.error = {last_w_, w_min_, w_des_};
  status.command = {last_velocity_};
  return status;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::ArmManipulabilityTask,
  task_priority_kinematic_control::TaskBase)
