#include "task_priority_kinematic_control/tasks/end_effector_orientation_task.hpp"

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

void EndEffectorOrientationTask::configure(
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
}

TaskComputation EndEffectorOrientationTask::update(
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
  const Eigen::Quaterniond current(frame.pose.rotation());
  const Eigen::Quaterniond target = has_pose_goal_ ?
    Eigen::Quaterniond(
      pose_goal_.pose.orientation.w,
      pose_goal_.pose.orientation.x,
      pose_goal_.pose.orientation.y,
      pose_goal_.pose.orientation.z) :
    default_goal_;

  computation.error = quaternion_error(target.normalized(), current.normalized());
  computation.desired_velocity = gains_.asDiagonal() * computation.error;
  computation.jacobian = frame.jacobian.bottomRows(3);
  computation.status_message = "tracking";
  return computation;
}

bool EndEffectorOrientationTask::set_gain(const std::vector<double> & gain, std::string & message)
{
  if (gain.size() != 3) {
    message = "Orientation gain must contain 3 values";
    return false;
  }
  for (size_t i = 0; i < gain.size(); ++i) {
    if (gain[i] < 0.0) {
      message = "Orientation gain values must be non-negative";
      return false;
    }
    gains_(static_cast<Eigen::Index>(i)) = gain[i];
  }
  message = "Orientation gain updated";
  return true;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::EndEffectorOrientationTask,
  task_priority_kinematic_control::TaskBase)
