#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"

#include <pluginlib/class_list_macros.hpp>

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
  frame_id_ = get_string_param(parameters, "frame_id", context.model->base_frame());
  const auto gains = get_double_array_param(parameters, "gain", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
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
  computation.has_frame_pose = true;
  computation.frame_id = frame_id_;
  computation.frame_pose = current_pose;

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

  computation.error = error;
  computation.desired_velocity = gains_.asDiagonal() * error;
  computation.jacobian = jacobian;
  computation.status_message = "tracking";
  return computation;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::FramePoseTask,
  task_priority_kinematic_control::TaskBase)
