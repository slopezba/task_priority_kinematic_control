#include "task_priority_kinematic_control/tasks/joint_limits_task.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <rclcpp/logging.hpp>

#include <pluginlib/class_list_macros.hpp>

namespace task_priority_kinematic_control
{

void JointLimitsTask::configure(
  const std::string & id,
  const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  if (!context.model) {throw std::invalid_argument("JointLimitsTask requires a configured model");}
  joint_names_ = context.model->all_joint_names();
  joint_count_ = static_cast<Eigen::Index>(joint_names_.size());
  dofs_ = static_cast<Eigen::Index>(context.model->total_dofs());
  base_dofs_ = static_cast<Eigen::Index>(context.model->base_dofs());
  limits_marker_publish_rate_ = get_double_param(parameters, "limits_marker_publish_rate", 1.0);
  if (!std::isfinite(limits_marker_publish_rate_) || limits_marker_publish_rate_ <= 0) {
    throw std::invalid_argument("limits_marker_publish_rate must be finite and positive");
  }
  set_publish_limits_marker(get_bool_param(parameters, "publish_limits_marker", false));
  const double legacy_margin = get_double_param(parameters, "margin", 0.1);
  alpha_ = get_double_param(parameters, "alpha", legacy_margin);
  delta_ = get_double_param(parameters, "delta", std::max(0.15, alpha_));
  eps_ = get_double_param(parameters, "eps", 1e-4);
  if (alpha_ < 0.0) {
    alpha_ = 0.0;
  }
  if (delta_ <= alpha_) {
    delta_ = alpha_ + 1e-6;
  }
  if (eps_ < 0.0) {
    eps_ = 0.0;
  }
  gain_ = get_double_param(parameters, "gain_scalar", 0.5);
  const auto lower = get_double_array_param(
    parameters, "lower_limits", std::vector<double>(joint_names_.size(), 0.0));
  const auto upper = get_double_array_param(
    parameters, "upper_limits", std::vector<double>(joint_names_.size(), 0.0));
  lower_limits_ = Eigen::VectorXd::Zero(lower.size());
  upper_limits_ = Eigen::VectorXd::Zero(upper.size());
  for (size_t i = 0; i < lower.size(); ++i) {
    lower_limits_(static_cast<Eigen::Index>(i)) = lower[i];
  }
  for (size_t i = 0; i < upper.size(); ++i) {
    upper_limits_(static_cast<Eigen::Index>(i)) = upper[i];
  }
  lower_active_.assign(joint_names_.size(), false);
  upper_active_.assign(joint_names_.size(), false);
  joint_frames_.assign(joint_names_.size(), std::string{});
  const auto backend = context.backend.lock();
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    if (!backend || !backend->joint_child_frame(joint_names_[i], joint_frames_[i]) || joint_frames_[i].empty()) {
      joint_frames_[i].clear();
      if (publish_limits_marker()) {
        RCLCPP_WARN(rclcpp::get_logger("JointLimitsTask"),
          "Task '%s': marker frame unavailable for joint '%s'; limit control remains enabled",
          id.c_str(), joint_names_[i].c_str());
      }
    }
  }
  base_marker_frame_.jacobian.setZero(6, dofs_);
  joint_marker_frame_.jacobian.setZero(6, dofs_);
  snapshots_.initialize([&](JointLimitsSnapshot & snapshot) {
      snapshot.joints.assign(joint_names_.size(), JointLimitPose{});
      snapshot.enabled = false; snapshot.visualization_enabled = false; snapshot.timestamp_ns = 0;
      snapshot.visualization_state = 0;
    });
}

void JointLimitsTask::prepare_computation(TaskComputation & out) const
{
  out.jacobian.setZero(joint_count_, dofs_);
  out.error.setZero(joint_count_);
  out.desired_velocity.setZero(joint_count_);
  out.active = false; out.stop_arm_motion = false; out.has_frame_pose = false;
  out.frame_id.clear(); out.frame_pose.setIdentity();
  out.status_message.reserve(32);
}

TaskComputation JointLimitsTask::update(
  const WholeBodyState & state, const KinematicsBackend & backend)
{
  TaskComputation out;
  prepare_computation(out);
  update_into(state, backend, out);
  return out;
}

void JointLimitsTask::update_into(
  const WholeBodyState & state, const KinematicsBackend & backend, TaskComputation & out)
{
  out.jacobian.setZero(); out.error.setZero(); out.desired_velocity.setZero();
  out.active = false; out.stop_arm_motion = false;
  if (!enabled_ || state.joint_positions.size() == 0) {
    out.status_message = enabled_ ? "waiting_for_joints" : "disabled";
    capture_markers(state, backend);
    return;
  }
  if (state.joint_positions.size() != joint_count_) {
    out.status_message = "invalid_joint_state";
    capture_markers(state, backend);
    return;
  }

  for (Eigen::Index i = 0; i < joint_count_; ++i) {
    const double q = state.joint_positions(i);
    const double lower = i < lower_limits_.size() ? lower_limits_(i) : 0.0;
    const double upper = i < upper_limits_.size() ? upper_limits_(i) : 0.0;
    double cmd = 0.0;
    if (upper > lower) {
      const size_t joint_index = static_cast<size_t>(i);
      if (!lower_active_[joint_index] && q < lower + alpha_) {
        lower_active_[joint_index] = true;
      }
      if (lower_active_[joint_index] && q >= lower + delta_ - eps_) {
        lower_active_[joint_index] = false;
      }

      if (!upper_active_[joint_index] && q > upper - alpha_) {
        upper_active_[joint_index] = true;
      }
      if (upper_active_[joint_index] && q <= upper - delta_ + eps_) {
        upper_active_[joint_index] = false;
      }

      if (lower_active_[joint_index]) {
        cmd = gain_ * ((lower + delta_) - q);
      } else if (upper_active_[joint_index]) {
        cmd = -gain_ * (q - (upper - delta_));
      }
    }
    if (cmd != 0.0) {
      out.jacobian(i, base_dofs_ + i) = 1.0;
      out.error(i) = cmd;
      out.desired_velocity(i) = cmd;
      out.active = true;
    }
  }
  out.status_message = out.active ? "repelling_limits" : "limits_inactive";
  capture_markers(state, backend);
}

void JointLimitsTask::capture_markers(
  const WholeBodyState & state, const KinematicsBackend & backend)
{
  auto & snapshot = snapshots_.writable();
  snapshot.visualization_state = marker_visibility_state();
  snapshot.visualization_enabled = (snapshot.visualization_state & 1U) != 0;
  snapshot.enabled = enabled_ && state.joint_positions.size() == joint_count_ && joint_count_ > 0;
  if (!snapshot.visualization_enabled) {return;}
  for (auto & joint : snapshot.joints) {joint = JointLimitPose{};}
  if (!snapshot.enabled) {return;}
  bool needs_pose = false;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    snapshot.joints[i].active = lower_active_[i] || upper_active_[i];
    if (snapshot.joints[i].active && !joint_frames_[i].empty() &&
      std::isfinite(state.joint_positions(static_cast<Eigen::Index>(i)))) {needs_pose = true;}
  }
  if (!needs_pose || !backend.read_frame_state(base_frame(), base_marker_frame_) ||
    !base_marker_frame_.pose.matrix().allFinite()) {return;}
  const Eigen::Matrix3d rotation = base_marker_frame_.pose.rotation().transpose();
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    auto & joint = snapshot.joints[i];
    if (!joint.active || joint_frames_[i].empty() ||
      !std::isfinite(state.joint_positions(static_cast<Eigen::Index>(i)))) {continue;}
    if (!backend.read_frame_state(joint_frames_[i], joint_marker_frame_) ||
      !joint_marker_frame_.pose.matrix().allFinite()) {continue;}
    const auto & jacobian = joint_marker_frame_.jacobian;
    const Eigen::Index column = base_dofs_ + static_cast<Eigen::Index>(i);
    if (jacobian.rows() != 6 || column >= jacobian.cols()) {continue;}
    // Revolute joints carry their world axis in the angular column; prismatic joints
    // carry it in the linear column. No new topology queries or frame reads are needed.
    Eigen::Vector3d world_axis = jacobian.block<3, 1>(3, column);
    if (!world_axis.allFinite()) {continue;}
    if (world_axis.squaredNorm() == 0.0) {world_axis = jacobian.block<3, 1>(0, column);}
    if (!world_axis.allFinite()) {continue;}
    joint.axis = rotation * world_axis;
    const double axis_norm = joint.axis.norm();
    if (!std::isfinite(axis_norm) || axis_norm <= 0.0) {continue;}
    joint.axis /= axis_norm;
    joint.position = rotation * (joint_marker_frame_.pose.translation() - base_marker_frame_.pose.translation());
    joint.valid = joint.position.allFinite();
  }
}

void JointLimitsTask::observe_command(const WholeBodyCommand &, int64_t timestamp_ns)
{
  if (!publish_limits_marker()) {return;}
  snapshots_.writable().timestamp_ns = timestamp_ns;
  snapshots_.publish();
}

bool JointLimitsTask::set_gain_scalar(double gain, std::string & message)
{
  if (gain < 0.0) {
    message = "gain_scalar must be non-negative";
    return false;
  }
  gain_ = gain;
  message = "Joint limits scalar gain updated";
  return true;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::JointLimitsTask,
  task_priority_kinematic_control::TaskBase)
