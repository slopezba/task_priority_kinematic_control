#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"
#include <sensor_msgs/msg/joint_state.hpp>
#include <chrono>

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
  bool set_gain(const std::vector<double> & gain, std::string & message) override;
  bool set_joint_activation(const std::vector<bool> & activation, std::string & message) override;
  msg::TaskStatus build_status() const override;
  std::vector<double> current_target() const override;
  bool set_joint_command(const sensor_msgs::msg::JointState & command, std::string & message);
  bool set_enabled(bool enabled) override;
  void reset() override;
  void clear_velocity();

private:
  std::vector<std::string> joint_names_;
  Eigen::VectorXd target_;
  Eigen::VectorXd gains_;
  std::vector<bool> joint_activation_;
  Eigen::VectorXd feedforward_;
  double command_timeout_ = 0.5;
  std::chrono::steady_clock::time_point command_time_;
};

}  // namespace task_priority_kinematic_control
