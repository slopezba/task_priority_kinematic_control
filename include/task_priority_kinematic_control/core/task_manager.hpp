#pragma once

#include "task_priority_kinematic_control/core/common.hpp"
#include "task_priority_kinematic_control/tasks/task_base.hpp"

#include "task_priority_kinematic_control/msg/task_status.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/node_interfaces/node_parameters_interface.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace task_priority_kinematic_control
{

class TaskManager
{
public:
  TaskManager(
    const rclcpp::node_interfaces::NodeParametersInterface::SharedPtr & parameters_interface,
    const rclcpp::Logger & logger);

  void configure(const TaskContext & context);

  std::vector<TaskComputation> update_all(
    const WholeBodyState & state,
    const KinematicsBackend & backend);

  bool set_task_enabled(const std::string & task_id, bool enabled, std::string & message);
  bool set_task_pose_goal(
    const std::string & task_id,
    const geometry_msgs::msg::PoseStamped & goal,
    std::string & message);
  bool set_task_joint_target(
    const std::string & task_id,
    const std::vector<double> & target,
    std::string & message);
  bool set_task_gain(
    const std::string & task_id,
    const std::vector<double> & gain,
    std::string & message);
  bool set_task_gain_scalar(
    const std::string & task_id,
    double gain,
    std::string & message);
  void disable_all_tasks();
  bool reorder_tasks(const std::vector<std::string> & ordered_ids, std::string & message);

  std::vector<msg::TaskStatus> get_task_statuses() const;
  std::vector<std::shared_ptr<TaskBase>> tasks() const;
  void reset();

private:
  std::map<std::string, rclcpp::Parameter> parameters_for_task(const std::string & task_id) const;

  rclcpp::node_interfaces::NodeParametersInterface::SharedPtr parameters_interface_;
  rclcpp::Logger logger_;
  pluginlib::ClassLoader<TaskBase> loader_;
  std::vector<std::shared_ptr<TaskBase>> tasks_;
};

}  // namespace task_priority_kinematic_control
