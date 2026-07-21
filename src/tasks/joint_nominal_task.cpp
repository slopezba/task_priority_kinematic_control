#include "task_priority_kinematic_control/tasks/joint_nominal_task.hpp"

#include <pluginlib/class_list_macros.hpp>

namespace task_priority_kinematic_control
{

namespace
{

std::vector<bool> activation_values_from_parameters(
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const size_t joint_count)
{
  std::vector<bool> activation(joint_count, true);
  const auto it = parameters.find("activation");
  if (it == parameters.end()) {
    return activation;
  }

  if (it->second.get_type() == rclcpp::ParameterType::PARAMETER_BOOL_ARRAY) {
    const auto values = it->second.as_bool_array();
    for (size_t i = 0; i < joint_count && i < values.size(); ++i) {
      activation[i] = values[i];
    }
    return activation;
  }

  if (it->second.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY) {
    const auto values = it->second.as_integer_array();
    for (size_t i = 0; i < joint_count && i < values.size(); ++i) {
      activation[i] = values[i] != 0;
    }
    return activation;
  }

  const auto values = it->second.as_double_array();
  for (size_t i = 0; i < joint_count && i < values.size(); ++i) {
    activation[i] = values[i] != 0.0;
  }
  return activation;
}

}  // namespace

void JointNominalTask::configure(
  const std::string & id,
  const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  joint_names_ = get_string_array_param(parameters, "joint_names", context.model->all_joint_names());
  const auto target = get_double_array_param(parameters, "target", std::vector<double>(joint_names_.size(), 0.0));
  const auto gains = get_double_array_param(parameters, "gain", std::vector<double>(joint_names_.size(), 0.3));
  joint_activation_ = activation_values_from_parameters(parameters, joint_names_.size());
  target_ = Eigen::VectorXd::Zero(joint_names_.size());
  gains_ = Eigen::VectorXd::Constant(joint_names_.size(), 0.3);
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    if (i < target.size()) {
      target_(static_cast<Eigen::Index>(i)) = target[i];
    }
    if (i < gains.size()) {
      gains_(static_cast<Eigen::Index>(i)) = gains[i];
    }
  }
}

TaskComputation JointNominalTask::update(
  const WholeBodyState & state,
  const KinematicsBackend &)
{
  TaskComputation computation;
  computation.active = enabled_ && state.joint_positions.size() > 0;
  if (!computation.active) {
    computation.status_message = enabled_ ? "waiting_for_joints" : "disabled";
    return computation;
  }

  std::vector<size_t> active_rows;
  active_rows.reserve(joint_names_.size());
  for (size_t i = 0; i < joint_activation_.size(); ++i) {
    if (joint_activation_[i]) {
      active_rows.push_back(i);
    }
  }

  if (active_rows.empty()) {
    computation.active = false;
    computation.status_message = "no_active_joints";
    return computation;
  }

  computation.jacobian = Eigen::MatrixXd::Zero(active_rows.size(), context_.model->total_dofs());
  computation.error = Eigen::VectorXd::Zero(active_rows.size());
  computation.desired_velocity = Eigen::VectorXd::Zero(active_rows.size());
  for (size_t row = 0; row < active_rows.size(); ++row) {
    const size_t joint_row = active_rows[row];
    const int joint_idx = context_.model->joint_index(joint_names_[joint_row]);
    if (joint_idx < 0 || joint_idx >= state.joint_positions.size()) {
      continue;
    }
    const Eigen::Index col = static_cast<Eigen::Index>(context_.model->base_dofs() + joint_idx);
    computation.jacobian(static_cast<Eigen::Index>(row), col) = 1.0;
    computation.error(static_cast<Eigen::Index>(row)) =
      target_(static_cast<Eigen::Index>(joint_row)) - state.joint_positions(joint_idx);
    computation.desired_velocity(static_cast<Eigen::Index>(row)) =
      gains_(static_cast<Eigen::Index>(joint_row)) * computation.error(static_cast<Eigen::Index>(row));
  }
  computation.status_message = "tracking";
  return computation;
}

bool JointNominalTask::set_joint_target(const std::vector<double> & target, std::string & message)
{
  if (target.size() != joint_names_.size()) {
    message = "Joint target size does not match configured joint_names size";
    return false;
  }

  for (size_t i = 0; i < target.size(); ++i) {
    target_(static_cast<Eigen::Index>(i)) = target[i];
  }
  message = "Joint target updated";
  return true;
}

bool JointNominalTask::set_gain(const std::vector<double> & gain, std::string & message)
{
  if (gain.size() != joint_names_.size()) {
    message = "Joint gain size does not match configured joint_names size";
    return false;
  }

  for (size_t i = 0; i < gain.size(); ++i) {
    if (gain[i] < 0.0) {
      message = "Joint gain values must be non-negative";
      return false;
    }
    gains_(static_cast<Eigen::Index>(i)) = gain[i];
  }
  message = "Joint gain updated";
  return true;
}

bool JointNominalTask::set_joint_activation(
  const std::vector<bool> & activation,
  std::string & message)
{
  if (activation.size() != joint_names_.size()) {
    message = "Joint activation size does not match configured joint_names size";
    return false;
  }

  joint_activation_ = activation;
  message = "Joint activation updated";
  return true;
}

msg::TaskStatus JointNominalTask::build_status() const
{
  auto status = TaskBaseCommon::build_status();
  status.target_type = "joint_array";
  status.joint_names = joint_names_;
  status.joint_activation = joint_activation_;
  return status;
}

std::vector<double> JointNominalTask::current_target() const
{
  std::vector<double> target(target_.size(), 0.0);
  for (Eigen::Index i = 0; i < target_.size(); ++i) {
    target[static_cast<size_t>(i)] = target_(i);
  }
  return target;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::JointNominalTask,
  task_priority_kinematic_control::TaskBase)
