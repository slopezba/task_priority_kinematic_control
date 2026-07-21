#include "task_priority_kinematic_control/core/task_manager.hpp"

#include <algorithm>
#include <rclcpp/logging.hpp>

namespace task_priority_kinematic_control
{

TaskManager::TaskManager(
  const rclcpp::node_interfaces::NodeParametersInterface::SharedPtr & parameters_interface,
  const rclcpp::Logger & logger)
: parameters_interface_(parameters_interface),
  logger_(logger),
  loader_("task_priority_kinematic_control", "task_priority_kinematic_control::TaskBase")
{
}

void TaskManager::configure(const TaskContext & context)
{
  tasks_.clear();
  rclcpp::Parameter task_ids_param;
  if (!parameters_interface_->get_parameter("task_ids", task_ids_param)) {
    RCLCPP_WARN(logger_, "Task manager configured without a task_ids parameter");
    return;
  }

  const auto task_ids = task_ids_param.as_string_array();
  for (const auto & task_id : task_ids) {
    const auto params = parameters_for_task(task_id);
    const auto plugin_it = params.find("plugin");
    if (plugin_it == params.end()) {
      RCLCPP_WARN(logger_, "Skipping task '%s' because it has no plugin entry", task_id.c_str());
      continue;
    }
    const std::string plugin_name = plugin_it->second.as_string();
    auto task = loader_.createSharedInstance(plugin_name);
    task->configure(task_id, plugin_name, params, context);
    tasks_.push_back(task);
  }
  std::sort(tasks_.begin(), tasks_.end(), [](const auto & a, const auto & b) {
    return a->priority() < b->priority();
  });
}

std::vector<TaskComputation> TaskManager::update_all(
  const WholeBodyState & state,
  const KinematicsBackend & backend)
{
  std::vector<TaskComputation> outputs;
  outputs.reserve(tasks_.size());
  for (const auto & task : tasks_) {
    outputs.push_back(task->update(state, backend));
  }
  return outputs;
}

bool TaskManager::set_task_enabled(const std::string & task_id, bool enabled, std::string & message)
{
  for (auto & task : tasks_) {
    if (task->id() == task_id) {
      task->set_enabled(enabled);
      message = "Task state updated";
      return true;
    }
  }
  message = "Task not found";
  return false;
}

bool TaskManager::set_task_pose_goal(
  const std::string & task_id,
  const geometry_msgs::msg::PoseStamped & goal,
  std::string & message)
{
  for (auto & task : tasks_) {
    if (task->id() == task_id) {
      const bool accepted = task->set_pose_goal(goal);
      message = accepted ? "Goal updated" : "Task rejected pose goal";
      return accepted;
    }
  }
  message = "Task not found";
  return false;
}

bool TaskManager::set_task_joint_target(
  const std::string & task_id,
  const std::vector<double> & target,
  std::string & message)
{
  for (auto & task : tasks_) {
    if (task->id() == task_id) {
      return task->set_joint_target(target, message);
    }
  }
  message = "Task not found";
  return false;
}

bool TaskManager::set_task_joint_trajectory(
  const std::string & task_id,
  const trajectory_msgs::msg::JointTrajectory & trajectory,
  std::string & message)
{
  for (auto & task : tasks_) {
    if (task->id() == task_id) {
      return task->set_joint_trajectory(trajectory, message);
    }
  }
  message = "Task not found";
  return false;
}

bool TaskManager::cancel_task_joint_trajectory(const std::string & task_id, std::string & message)
{
  for (auto & task : tasks_) {
    if (task->id() == task_id) {
      return task->cancel_joint_trajectory(message);
    }
  }
  message = "Task not found";
  return false;
}

JointTrajectoryTaskStatus TaskManager::get_task_joint_trajectory_status(
  const std::string & task_id) const
{
  for (const auto & task : tasks_) {
    if (task->id() == task_id) {
      return task->joint_trajectory_status();
    }
  }
  JointTrajectoryTaskStatus status;
  status.message = "Task not found";
  return status;
}

bool TaskManager::set_task_gain(
  const std::string & task_id,
  const std::vector<double> & gain,
  std::string & message)
{
  for (auto & task : tasks_) {
    if (task->id() == task_id) {
      return task->set_gain(gain, message);
    }
  }
  message = "Task not found";
  return false;
}

bool TaskManager::set_task_gain_scalar(
  const std::string & task_id,
  double gain,
  std::string & message)
{
  for (auto & task : tasks_) {
    if (task->id() == task_id) {
      return task->set_gain_scalar(gain, message);
    }
  }
  message = "Task not found";
  return false;
}

bool TaskManager::set_task_joint_activation(
  const std::string & task_id,
  const std::vector<bool> & activation,
  std::string & message)
{
  for (auto & task : tasks_) {
    if (task->id() == task_id) {
      return task->set_joint_activation(activation, message);
    }
  }
  message = "Task not found";
  return false;
}

void TaskManager::disable_all_tasks()
{
  for (auto & task : tasks_) {
    task->set_enabled(false);
  }
}

bool TaskManager::reorder_tasks(const std::vector<std::string> & ordered_ids, std::string & message)
{
  if (ordered_ids.size() != tasks_.size()) {
    message = "Ordered task list size does not match current task count";
    return false;
  }

  std::vector<std::shared_ptr<TaskBase>> reordered;
  reordered.reserve(tasks_.size());
  for (size_t priority = 0; priority < ordered_ids.size(); ++priority) {
    auto it = std::find_if(tasks_.begin(), tasks_.end(), [&](const auto & task) {
      return task->id() == ordered_ids[priority];
    });
    if (it == tasks_.end()) {
      message = "Unknown task id in reorder request: " + ordered_ids[priority];
      return false;
    }
    (*it)->set_priority(static_cast<uint32_t>(priority));
    reordered.push_back(*it);
  }
  tasks_ = reordered;
  message = "Tasks reordered";
  return true;
}

std::vector<msg::TaskStatus> TaskManager::get_task_statuses() const
{
  std::vector<msg::TaskStatus> statuses;
  statuses.reserve(tasks_.size());
  for (const auto & task : tasks_) {
    auto status = task->build_status();
    if (status.target_type == "none" && task->plugin_name().find("PoseTask") != std::string::npos) {
      status.target_type = "pose";
    }
    statuses.push_back(status);
  }
  return statuses;
}

std::vector<std::shared_ptr<TaskBase>> TaskManager::tasks() const
{
  return tasks_;
}

void TaskManager::reset()
{
  for (auto & task : tasks_) {
    task->reset();
  }
}

std::map<std::string, rclcpp::Parameter> TaskManager::parameters_for_task(const std::string & task_id) const
{
  std::map<std::string, rclcpp::Parameter> result;
  const std::string prefix = "tasks." + task_id;
  const auto listed = parameters_interface_->list_parameters({prefix}, 10);
  for (const auto & full_name : listed.names) {
    rclcpp::Parameter parameter;
    if (!parameters_interface_->get_parameter(full_name, parameter)) {
      continue;
    }
    const std::string expected_prefix = prefix + ".";
    if (full_name.rfind(expected_prefix, 0) != 0) {
      continue;
    }
    const std::string short_name = full_name.substr(expected_prefix.size());
    result.emplace(short_name, parameter);
  }
  return result;
}

}  // namespace task_priority_kinematic_control
