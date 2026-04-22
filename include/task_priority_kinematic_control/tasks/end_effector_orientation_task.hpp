#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"

namespace task_priority_kinematic_control
{

class EndEffectorOrientationTask : public TaskBaseCommon
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

protected:
  std::string frame_id_;
  Eigen::Vector3d gains_ = Eigen::Vector3d::Ones();
  Eigen::Quaterniond default_goal_ = Eigen::Quaterniond::Identity();
};

}  // namespace task_priority_kinematic_control
