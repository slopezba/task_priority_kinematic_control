#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <stdexcept>
#include <cmath>

namespace task_priority_kinematic_control
{

namespace
{

Eigen::Vector3d quaternion_error(
  const Eigen::Quaterniond & target,
  const Eigen::Quaterniond & current)
{
  const Eigen::Quaterniond q_err = target * current.conjugate();
  Eigen::AngleAxisd aa(q_err);
  return aa.axis() * aa.angle();
}

Eigen::Isometry3d base_pose_from_state(const WholeBodyState & state)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = state.base_position;
  const Eigen::AngleAxisd roll(state.base_rpy.x(), Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch(state.base_rpy.y(), Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw(state.base_rpy.z(), Eigen::Vector3d::UnitZ());
  pose.linear() = (yaw * pitch * roll).toRotationMatrix();
  return pose;
}

Eigen::MatrixXd base_jacobian_for_model(const WholeBodyModel & model)
{
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, model.total_dofs());
  for (size_t i = 0; i < model.active_base_dofs().size(); ++i) {
    const int dof = model.active_base_dofs()[i];
    if (dof >= 0 && dof < 6) {
      jacobian.col(static_cast<Eigen::Index>(i))(dof) = 1.0;
    }
  }
  return jacobian;
}

}  // namespace

void FramePoseTask::configure(
  const std::string & id,
  const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  configure_pose_feedforward(parameters);
  frame_id_ = get_string_param(parameters, "frame_id", context.model->base_frame());
  activation_.setOnes();
  const auto mask = parameters.find("activation");
  if (mask != parameters.end()) {
    const auto values = mask->second.as_integer_array();
    // Empty is the controller's default and preserves full-pose behavior.
    if (!values.empty() && values.size() != 6) {
      throw std::invalid_argument("Frame pose activation must contain 6 values");
    }
    for (size_t i = 0; i < values.size(); ++i) {
      if (values[i] != 0 && values[i] != 1) {
        throw std::invalid_argument("Frame pose activation values must be 0 or 1");
      }
      activation_(static_cast<Eigen::Index>(i)) = values[i];
    }
  }
  const auto gains = get_double_array_param(parameters, "gain", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  roll_pitch_control_ = get_bool_param(parameters, "roll_pitch_control", false);
  allow_roll_pitch_ = get_bool_param(parameters, "allow_roll_pitch", false);
  tilt_threshold_ = get_double_param(parameters, "feedforward_velocity_threshold", 0.001);
  if (roll_pitch_control_ && (frame_id_ != context.model->base_frame() ||
    !activation_.isApprox((PoseVelocity() << 0,0,0,1,1,0).finished()))) {
    throw std::invalid_argument("Roll/pitch control requires the base frame and activation [0,0,0,1,1,0]");
  }
  for (size_t i = 0; i < 6 && i < gains.size(); ++i) {
    gains_(static_cast<Eigen::Index>(i)) = gains[i];
  }
}

TaskComputation FramePoseTask::update(
  const WholeBodyState & state,
  const KinematicsBackend & backend)
{
  TaskComputation computation;
  computation.active = enabled_;
  if (!enabled_) {
    computation.status_message = "disabled";
    return computation;
  }

  Eigen::Isometry3d current_pose = Eigen::Isometry3d::Identity();
  Eigen::MatrixXd jacobian;

  if (frame_id_ == context_.model->base_frame()) {
    current_pose = base_pose_from_state(state);
    jacobian = base_jacobian_for_model(*context_.model);
  } else {
    const FrameState frame = backend.get_frame_state(frame_id_);
    current_pose = frame.pose;
    jacobian = frame.jacobian;
  }
  if (!current_pose.matrix().allFinite()) {
    computation.active = false;
    computation.status_message = "frame pose unavailable";
    return computation;
  }
  computation.has_frame_pose = true;
  computation.frame_id = frame_id_;
  computation.frame_pose = current_pose;
  if (roll_pitch_control_) {
    // Euler tilt rates from spatial (world) angular velocity. Yaw is free.
    const double pitch = state.base_rpy.y();
    const double yaw = state.base_rpy.z();
    const double cp = std::cos(pitch);
    if (std::abs(cp) < 1e-4) {
      computation.active = false;
      computation.status_message = "roll/pitch singularity";
      return computation;
    }
    Eigen::Matrix<double, 2, 3> tilt_map;
    tilt_map << std::cos(yaw) / cp, std::sin(yaw) / cp, 0,
      -std::sin(yaw), std::cos(yaw), 0;
    computation.feedforward = pose_feedforward().cwiseProduct(activation_);
    Eigen::Vector2d velocity = tilt_map * computation.feedforward.tail<3>();
    const bool moving = allow_roll_pitch_ && velocity.norm() > tilt_threshold_;
    if (!allow_roll_pitch_) {
      tilt_target_.setZero();
      computation.feedforward.setZero();
      velocity.setZero();
    } else if (!tilt_initialized_ || moving || tilt_moving_) {
      // Also capture the measured tilt on the first zero/expired sample.
      tilt_target_ = state.base_rpy.head<2>();
    }
    tilt_initialized_ = true;
    tilt_moving_ = moving;
    const Eigen::Quaterniond target(
      Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(tilt_target_.y(), Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(tilt_target_.x(), Eigen::Vector3d::UnitX()));
    pose_goal_.header.frame_id = context_.model->world_frame();
    pose_goal_.pose.position.x = state.base_position.x();
    pose_goal_.pose.position.y = state.base_position.y();
    pose_goal_.pose.position.z = state.base_position.z();
    pose_goal_.pose.orientation.x = target.x();
    pose_goal_.pose.orientation.y = target.y();
    pose_goal_.pose.orientation.z = target.z();
    pose_goal_.pose.orientation.w = target.w();
    has_pose_goal_ = true;
    computation.has_updated_target = true;
    computation.updated_target = pose_goal_.pose;
    computation.error = tilt_target_ - state.base_rpy.head<2>();
    for (Eigen::Index i = 0; i < 2; ++i) {
      computation.error(i) = std::atan2(std::sin(computation.error(i)), std::cos(computation.error(i)));
    }
    computation.desired_velocity = velocity + gains_.segment<2>(3).cwiseProduct(computation.error);
    computation.jacobian = tilt_map * jacobian.bottomRows(3);
    computation.status_message = allow_roll_pitch_ ? "tilt permitted" : "leveling";
    return computation;
  }
  update_pose_target(current_pose, computation, activation_);

  Eigen::Matrix<double, 6, 1> error = Eigen::Matrix<double, 6, 1>::Zero();
  const Eigen::Vector3d target_pos = has_pose_goal_ ?
    Eigen::Vector3d(
      pose_goal_.pose.position.x,
      pose_goal_.pose.position.y,
      pose_goal_.pose.position.z) :
    current_pose.translation();
  error.head<3>() = target_pos - current_pose.translation();

  if (has_pose_goal_) {
    const Eigen::Quaterniond current(current_pose.rotation());
    const Eigen::Quaterniond target(
      pose_goal_.pose.orientation.w,
      pose_goal_.pose.orientation.x,
      pose_goal_.pose.orientation.y,
      pose_goal_.pose.orientation.z);
    error.tail<3>() = quaternion_error(target.normalized(), current.normalized());
  }

  const Eigen::Index rows = static_cast<Eigen::Index>(activation_.sum());
  if (rows == 0) {
    computation.active = false;
    computation.status_message = "no active axes";
    return computation;
  }
  // Remove inactive rows so they leave degrees of freedom to lower priorities.
  computation.error.resize(rows);
  computation.desired_velocity.resize(rows);
  computation.jacobian.resize(rows, jacobian.cols());
  Eigen::Index row = 0;
  for (Eigen::Index axis = 0; axis < 6; ++axis) {
    if (activation_(axis) == 0.0) {continue;}
    computation.error(row) = error(axis);
    computation.desired_velocity(row) = computation.feedforward(axis) + gains_(axis) * error(axis);
    computation.jacobian.row(row) = jacobian.row(axis);
    ++row;
  }
  computation.status_message = "tracking";
  return computation;
}

bool FramePoseTask::set_gain(const std::vector<double> & gain, std::string & message)
{
  if (gain.size() != 6) {
    message = "Frame pose gain must contain 6 values";
    return false;
  }
  for (size_t i = 0; i < gain.size(); ++i) {
    if (gain[i] < 0.0) {
      message = "Frame pose gain values must be non-negative";
      return false;
    }
    gains_(static_cast<Eigen::Index>(i)) = gain[i];
  }
  message = "Frame pose gain updated";
  return true;
}

bool FramePoseTask::set_allow_roll_pitch(bool allow, std::string & message)
{
  if (!roll_pitch_control_) {message = "Roll/pitch policy is not configured"; return false;}
  if (allow != allow_roll_pitch_) {
    clear_feedforward();
    tilt_initialized_ = false;
    tilt_moving_ = false;
  }
  allow_roll_pitch_ = allow;
  if (!allow) {tilt_target_.setZero();}
  message = allow ? "Nonzero roll/pitch allowed" : "Roll/pitch fixed at zero";
  return true;
}

bool FramePoseTask::set_enabled(bool enabled)
{
  if (enabled != enabled_) {tilt_initialized_ = false; tilt_moving_ = false;}
  return PoseTaskCommon::set_enabled(enabled);
}

void FramePoseTask::reset()
{
  PoseTaskCommon::reset();
  tilt_initialized_ = false;
  tilt_moving_ = false;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::FramePoseTask,
  task_priority_kinematic_control::TaskBase)
