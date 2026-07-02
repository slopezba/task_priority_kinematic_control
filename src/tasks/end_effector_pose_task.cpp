#include "task_priority_kinematic_control/tasks/end_effector_pose_task.hpp"

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

}  // namespace

void EndEffectorPoseTask::configure(
  const std::string & id,
  const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  frame_id_ = get_string_param(parameters, "frame_id", context.model->left_tip_frame());
  const auto gains = get_double_array_param(parameters, "gain", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  for (size_t i = 0; i < 6 && i < gains.size(); ++i) {
    gains_(static_cast<Eigen::Index>(i)) = gains[i];
  }
}

TaskComputation EndEffectorPoseTask::update(
  const WholeBodyState &,
  const KinematicsBackend & backend)
{
  TaskComputation computation;
  computation.active = enabled_;
  if (!enabled_) {
    computation.status_message = "disabled";
    return computation;
  }

  const FrameState frame = backend.get_frame_state(frame_id_);
  computation.has_frame_pose = true;
  computation.frame_id = frame_id_;
  computation.frame_pose = frame.pose;
  Eigen::Matrix<double, 6, 1> error = Eigen::Matrix<double, 6, 1>::Zero();
  const Eigen::Vector3d target_pos = has_pose_goal_ ?
    Eigen::Vector3d(
      pose_goal_.pose.position.x,
      pose_goal_.pose.position.y,
      pose_goal_.pose.position.z) :
    frame.pose.translation();
  error.head<3>() = target_pos - frame.pose.translation();

  if (has_pose_goal_) {
    const Eigen::Quaterniond current(frame.pose.rotation());
    const Eigen::Quaterniond target(
      pose_goal_.pose.orientation.w,
      pose_goal_.pose.orientation.x,
      pose_goal_.pose.orientation.y,
      pose_goal_.pose.orientation.z);
    error.tail<3>() = quaternion_error(target.normalized(), current.normalized());
  }

  computation.error = error;
  computation.desired_velocity = gains_.asDiagonal() * error;
  computation.jacobian = frame.jacobian;
  computation.status_message = "tracking";
  return computation;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::EndEffectorPoseTask,
  task_priority_kinematic_control::TaskBase)
