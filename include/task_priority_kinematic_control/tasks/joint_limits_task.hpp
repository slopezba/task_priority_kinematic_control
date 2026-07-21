#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"

namespace task_priority_kinematic_control
{

class JointLimitsTask : public TaskBaseCommon
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

  bool set_gain_scalar(double gain, std::string & message) override;

private:
  double alpha_ = 0.1;
  double delta_ = 0.15;
  double eps_ = 1e-4;
  double gain_ = 1.0;
  Eigen::VectorXd lower_limits_;
  Eigen::VectorXd upper_limits_;
  std::vector<bool> lower_active_;
  std::vector<bool> upper_active_;
};

}  // namespace task_priority_kinematic_control
