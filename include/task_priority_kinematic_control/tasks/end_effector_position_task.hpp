#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"

namespace task_priority_kinematic_control
{

class EndEffectorPositionTask : public TaskBaseCommon
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
  Eigen::Vector3d default_goal_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d gains_ = Eigen::Vector3d::Ones();
};

}  // namespace task_priority_kinematic_control
