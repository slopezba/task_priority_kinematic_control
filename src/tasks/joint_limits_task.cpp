#include "task_priority_kinematic_control/tasks/joint_limits_task.hpp"

#include <algorithm>

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
  const size_t joint_count = context.model->all_joint_names().size();
  lower_active_ = std::vector<bool>(joint_count, false);
  upper_active_ = std::vector<bool>(joint_count, false);
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

  struct ActiveLimit
  {
    Eigen::Index joint_index;
    double command;
  };

  const Eigen::Index joint_count = state.joint_positions.size();
  if (lower_active_.size() < static_cast<size_t>(joint_count)) {
    lower_active_.resize(static_cast<size_t>(joint_count), false);
  }
  if (upper_active_.size() < static_cast<size_t>(joint_count)) {
    upper_active_.resize(static_cast<size_t>(joint_count), false);
  }

  std::vector<ActiveLimit> active_limits;
  active_limits.reserve(static_cast<size_t>(joint_count));
  for (Eigen::Index i = 0; i < joint_count; ++i) {
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
      active_limits.push_back(ActiveLimit{i, cmd});
    }
  }

  if (active_limits.empty()) {
    computation.active = false;
    computation.status_message = "limits_inactive";
    return computation;
  }

  computation.jacobian = Eigen::MatrixXd::Zero(active_limits.size(), context_.model->total_dofs());
  computation.error = Eigen::VectorXd::Zero(active_limits.size());
  computation.desired_velocity = Eigen::VectorXd::Zero(active_limits.size());

  for (size_t row = 0; row < active_limits.size(); ++row) {
    const auto & active_limit = active_limits[row];
    const Eigen::Index col =
      static_cast<Eigen::Index>(context_.model->base_dofs()) + active_limit.joint_index;
    computation.jacobian(static_cast<Eigen::Index>(row), col) = 1.0;
    computation.error(static_cast<Eigen::Index>(row)) = active_limit.command;
    computation.desired_velocity(static_cast<Eigen::Index>(row)) = active_limit.command;
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
