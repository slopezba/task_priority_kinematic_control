#include "task_priority_kinematic_control/tasks/joint_limits_task.hpp"

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
  margin_ = get_double_param(parameters, "margin", 0.1);
  gain_ = get_double_param(parameters, "gain_scalar", 0.5);
  const auto lower = get_double_array_param(
    parameters, "lower_limits", std::vector<double>(context.model->all_joint_names().size(), 0.0));
  const auto upper = get_double_array_param(
    parameters, "upper_limits", std::vector<double>(context.model->all_joint_names().size(), 0.0));
  lower_limits_ = Eigen::VectorXd::Zero(lower.size());
  upper_limits_ = Eigen::VectorXd::Zero(upper.size());
  for (size_t i = 0; i < lower.size(); ++i) {
    lower_limits_(static_cast<Eigen::Index>(i)) = lower[i];
  }
  for (size_t i = 0; i < upper.size(); ++i) {
    upper_limits_(static_cast<Eigen::Index>(i)) = upper[i];
  }
}

TaskComputation JointLimitsTask::update(
  const WholeBodyState & state,
  const KinematicsBackend &)
{
  TaskComputation computation;
  computation.active = enabled_ && state.joint_positions.size() > 0;
  if (!computation.active) {
    computation.status_message = enabled_ ? "waiting_for_joints" : "disabled";
    return computation;
  }

  const Eigen::Index joint_count = state.joint_positions.size();
  computation.jacobian = Eigen::MatrixXd::Zero(joint_count, context_.model->total_dofs());
  computation.error = Eigen::VectorXd::Zero(joint_count);
  computation.desired_velocity = Eigen::VectorXd::Zero(joint_count);

  for (Eigen::Index i = 0; i < joint_count; ++i) {
    const Eigen::Index col = static_cast<Eigen::Index>(context_.model->base_dofs()) + i;
    computation.jacobian(i, col) = 1.0;
    const double q = state.joint_positions(i);
    const double lower = i < lower_limits_.size() ? lower_limits_(i) : 0.0;
    const double upper = i < upper_limits_.size() ? upper_limits_(i) : 0.0;
    double cmd = 0.0;
    if (upper > lower) {
      if (q < lower + margin_) {
        cmd = gain_ * ((lower + margin_) - q);
      } else if (q > upper - margin_) {
        cmd = -gain_ * (q - (upper - margin_));
      }
    }
    computation.error(i) = cmd;
    computation.desired_velocity(i) = cmd;
  }

  computation.status_message = "repelling_limits";
  return computation;
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
