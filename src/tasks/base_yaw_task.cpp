#include "task_priority_kinematic_control/tasks/base_yaw_task.hpp"

#include <pluginlib/class_list_macros.hpp>

namespace task_priority_kinematic_control
{

namespace
{

double wrap_angle(double angle)
{
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

}  // namespace

void BaseYawTask::configure(
  const std::string & id,
  const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  target_yaw_ = get_double_param(parameters, "target_yaw", 0.0);
  gain_ = get_double_param(parameters, "gain_scalar", 1.0);
}

TaskComputation BaseYawTask::update(
  const WholeBodyState & state,
  const KinematicsBackend &)
{
  TaskComputation computation;
  computation.active = enabled_ && state.navigation_valid && context_.model->base_dofs() > 0;
  if (!computation.active) {
    computation.status_message = enabled_ ? "waiting_for_navigation" : "disabled";
    return computation;
  }

  computation.jacobian = Eigen::MatrixXd::Zero(1, context_.model->total_dofs());
  const auto & base_dofs = context_.model->active_base_dofs();
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(base_dofs.size()); ++i) {
    if (base_dofs[static_cast<size_t>(i)] == 5) {
      computation.jacobian(0, i) = 1.0;
    }
  }
  computation.error = Eigen::VectorXd::Constant(1, wrap_angle(target_yaw_ - state.base_rpy.z()));
  computation.desired_velocity = Eigen::VectorXd::Constant(1, gain_ * computation.error(0));
  computation.status_message = "tracking";
  return computation;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::BaseYawTask,
  task_priority_kinematic_control::TaskBase)
