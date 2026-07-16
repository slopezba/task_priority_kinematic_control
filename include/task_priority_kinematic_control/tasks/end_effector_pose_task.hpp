#pragma once

#include "task_priority_kinematic_control/tasks/end_effector_orientation_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effector_position_task.hpp"

namespace task_priority_kinematic_control
{

class EndEffectorPoseTask : public TaskBaseCommon
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

  bool set_gain(const std::vector<double> & gain, std::string & message) override;

private:
  std::string frame_id_;
  Eigen::Matrix<double, 6, 1> gains_ = Eigen::Matrix<double, 6, 1>::Ones();
};

}  // namespace task_priority_kinematic_control
