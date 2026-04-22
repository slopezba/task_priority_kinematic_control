#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"

namespace task_priority_kinematic_control
{

class JointNominalTask : public TaskBaseCommon
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

  bool set_joint_target(const std::vector<double> & target, std::string & message) override;
  msg::TaskStatus build_status() const override;
  std::vector<double> current_target() const override;

private:
  std::vector<std::string> joint_names_;
  Eigen::VectorXd target_;
  Eigen::VectorXd gains_;
};

}  // namespace task_priority_kinematic_control
