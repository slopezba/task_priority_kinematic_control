#include "task_priority_kinematic_control/tasks/end_effector_position_task.hpp"

#include <pluginlib/class_list_macros.hpp>

namespace task_priority_kinematic_control
{

void EndEffectorPositionTask::configure(
  const std::string & id,
  const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  frame_id_ = get_string_param(parameters, "frame_id", context.model->left_tip_frame());
  const auto gains = get_double_array_param(parameters, "gain", {1.0, 1.0, 1.0});
  for (size_t i = 0; i < 3 && i < gains.size(); ++i) {
    gains_(static_cast<Eigen::Index>(i)) = gains[i];
  }
  const auto goal = get_double_array_param(parameters, "default_goal", {0.0, 0.0, 0.0});
  for (size_t i = 0; i < 3 && i < goal.size(); ++i) {
    default_goal_(static_cast<Eigen::Index>(i)) = goal[i];
  }
}

TaskComputation EndEffectorPositionTask::update(
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
  const Eigen::Vector3d target = has_pose_goal_ ?
    Eigen::Vector3d(
      pose_goal_.pose.position.x,
      pose_goal_.pose.position.y,
      pose_goal_.pose.position.z) :
    default_goal_;
  const Eigen::Vector3d current = frame.pose.translation();
  computation.error = target - current;
  computation.desired_velocity = gains_.asDiagonal() * computation.error;
  computation.jacobian = frame.jacobian.topRows(3);
  computation.status_message = "tracking";
  return computation;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::EndEffectorPositionTask,
  task_priority_kinematic_control::TaskBase)
