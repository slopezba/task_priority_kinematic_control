#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"

namespace task_priority_kinematic_control
{

class EndEffectorsRelativePoseTask : public TaskBaseCommon
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

  bool set_pose_goal(const geometry_msgs::msg::PoseStamped & goal) override;
  bool set_gain(const std::vector<double> & gain, std::string & message) override;
  msg::TaskStatus build_status() const override;
  std::vector<double> current_target() const override;

private:
  std::string reference_frame_;
  std::string controlled_frame_;
  Eigen::Matrix<double, 6, 1> gains_ = Eigen::Matrix<double, 6, 1>::Ones();
  Eigen::Matrix<bool, 6, 1> activation_ = Eigen::Matrix<bool, 6, 1>::Constant(true);
  Eigen::Isometry3d target_relative_pose_ = Eigen::Isometry3d::Identity();
  bool has_target_relative_pose_ = false;
};

}  // namespace task_priority_kinematic_control
