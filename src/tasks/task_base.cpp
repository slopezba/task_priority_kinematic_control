#include "task_priority_kinematic_control/tasks/task_base.hpp"

namespace task_priority_kinematic_control
{

bool TaskBase::set_pose_goal(const geometry_msgs::msg::PoseStamped &) { return false; }
bool TaskBase::set_joint_target(const std::vector<double> &, std::string & message)
{
  message = "Task does not accept joint targets";
  return false;
}
bool TaskBase::set_enabled(bool) { return false; }
void TaskBase::reset() {}
std::vector<double> TaskBase::current_target() const { return {}; }

void TaskBaseCommon::configure_common(
  const std::string & id,
  const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  id_ = id;
  plugin_name_ = plugin_name;
  group_ = get_string_param(parameters, "group", "default");
  priority_ = static_cast<uint32_t>(get_double_param(parameters, "priority", 0.0));
  enabled_ = get_bool_param(parameters, "enabled", true);
  context_ = context;
}

bool TaskBaseCommon::set_pose_goal(const geometry_msgs::msg::PoseStamped & goal)
{
  pose_goal_ = goal;
  has_pose_goal_ = true;
  return true;
}

bool TaskBaseCommon::set_joint_target(const std::vector<double> &, std::string & message)
{
  message = "Task does not accept joint targets";
  return false;
}

bool TaskBaseCommon::set_enabled(bool enabled)
{
  enabled_ = enabled;
  return true;
}

void TaskBaseCommon::reset() {}
msg::TaskStatus TaskBaseCommon::build_status() const
{
  msg::TaskStatus status;
  status.id = id_;
  status.plugin = plugin_name_;
  status.group = group_;
  status.priority = priority_;
  status.enabled = enabled_;
  status.active = enabled_;
  status.status_message = enabled_ ? "configured" : "disabled";
  status.target_type = "none";
  return status;
}
std::string TaskBaseCommon::id() const { return id_; }
std::string TaskBaseCommon::plugin_name() const { return plugin_name_; }
std::string TaskBaseCommon::group() const { return group_; }
uint32_t TaskBaseCommon::priority() const { return priority_; }
void TaskBaseCommon::set_priority(uint32_t priority) { priority_ = priority; }
bool TaskBaseCommon::enabled() const { return enabled_; }

Eigen::VectorXd TaskBaseCommon::gain_vector(size_t expected_size, double default_gain) const
{
  return Eigen::VectorXd::Constant(expected_size, default_gain);
}

std::string TaskBaseCommon::get_string_param(
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const std::string & name,
  const std::string & default_value) const
{
  const auto it = parameters.find(name);
  if (it == parameters.end()) {
    return default_value;
  }
  return it->second.as_string();
}

double TaskBaseCommon::get_double_param(
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const std::string & name,
  double default_value) const
{
  const auto it = parameters.find(name);
  if (it == parameters.end()) {
    return default_value;
  }
  return it->second.as_double();
}

bool TaskBaseCommon::get_bool_param(
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const std::string & name,
  bool default_value) const
{
  const auto it = parameters.find(name);
  if (it == parameters.end()) {
    return default_value;
  }
  return it->second.as_bool();
}

std::vector<double> TaskBaseCommon::get_double_array_param(
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const std::string & name,
  const std::vector<double> & default_value) const
{
  const auto it = parameters.find(name);
  if (it == parameters.end()) {
    return default_value;
  }
  return it->second.as_double_array();
}

std::vector<std::string> TaskBaseCommon::get_string_array_param(
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const std::string & name,
  const std::vector<std::string> & default_value) const
{
  const auto it = parameters.find(name);
  if (it == parameters.end()) {
    return default_value;
  }
  return it->second.as_string_array();
}

}  // namespace task_priority_kinematic_control
