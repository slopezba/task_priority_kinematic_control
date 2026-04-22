#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"

namespace task_priority_kinematic_control
{

class BaseYawTask : public TaskBaseCommon
{
public:
  void configure(
    const std::string & id,
    const std::string & plugin_name,
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const TaskContext & context) override;

  TaskComputation update(
    const WholeBodyState & state,
    const KinematicsBackend & backend) override;

private:
  double target_yaw_ = 0.0;
  double gain_ = 1.0;
};

}  // namespace task_priority_kinematic_control
