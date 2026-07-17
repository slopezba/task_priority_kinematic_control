#pragma once

#include "task_priority_kinematic_control/core/common.hpp"
#include "task_priority_kinematic_control/core/whole_body_model.hpp"
#include "task_priority_kinematic_control/kinematics/kinematics_backend.hpp"
#include "task_priority_kinematic_control/msg/task_status.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/parameter.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <map>
#include <memory>
#include <string>

namespace task_priority_kinematic_control
{

struct TaskContext
{
  std::shared_ptr<WholeBodyModel> model;
};

class TaskBase
{
public:
  virtual ~TaskBase() = default;

  virtual void configure(
    const std::string & id,
    const std::string & plugin_name,
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const TaskContext & context) = 0;

  virtual TaskComputation update(
    const WholeBodyState & state,
    const KinematicsBackend & backend) = 0;

  virtual bool set_pose_goal(const geometry_msgs::msg::PoseStamped & goal);
  virtual bool set_joint_target(const std::vector<double> & target, std::string & message);
  virtual bool set_joint_trajectory(
    const trajectory_msgs::msg::JointTrajectory & trajectory,
    std::string & message);
  virtual bool cancel_joint_trajectory(std::string & message);
  virtual JointTrajectoryTaskStatus joint_trajectory_status() const;
  virtual bool set_gain(const std::vector<double> & gain, std::string & message);
  virtual bool set_gain_scalar(double gain, std::string & message);
  virtual bool set_enabled(bool enabled);
  virtual void reset();
  virtual msg::TaskStatus build_status() const = 0;
  virtual std::vector<double> current_target() const;

  virtual std::string id() const = 0;
  virtual std::string plugin_name() const = 0;
  virtual std::string group() const = 0;
  virtual uint32_t priority() const = 0;
  virtual void set_priority(uint32_t priority) = 0;
  virtual bool enabled() const = 0;
};

class TaskBaseCommon : public TaskBase
{
public:
  bool set_pose_goal(const geometry_msgs::msg::PoseStamped & goal) override;
  bool set_joint_target(const std::vector<double> & target, std::string & message) override;
  bool set_joint_trajectory(
    const trajectory_msgs::msg::JointTrajectory & trajectory,
    std::string & message) override;
  bool cancel_joint_trajectory(std::string & message) override;
  JointTrajectoryTaskStatus joint_trajectory_status() const override;
  bool set_gain(const std::vector<double> & gain, std::string & message) override;
  bool set_gain_scalar(double gain, std::string & message) override;
  bool set_enabled(bool enabled) override;
  void reset() override;
  msg::TaskStatus build_status() const override;

  std::string id() const override;
  std::string plugin_name() const override;
  std::string group() const override;
  uint32_t priority() const override;
  void set_priority(uint32_t priority) override;
  bool enabled() const override;

protected:
  void configure_common(
    const std::string & id,
    const std::string & plugin_name,
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const TaskContext & context);

  Eigen::VectorXd gain_vector(size_t expected_size, double default_gain) const;
  std::string get_string_param(
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const std::string & name,
    const std::string & default_value) const;
  double get_double_param(
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const std::string & name,
    double default_value) const;
  bool get_bool_param(
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const std::string & name,
    bool default_value) const;
  std::vector<double> get_double_array_param(
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const std::string & name,
    const std::vector<double> & default_value) const;
  std::vector<std::string> get_string_array_param(
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const std::string & name,
    const std::vector<std::string> & default_value) const;

  std::string id_;
  std::string plugin_name_;
  std::string group_ = "default";
  uint32_t priority_ = 0;
  bool enabled_ = true;
  geometry_msgs::msg::PoseStamped pose_goal_;
  bool has_pose_goal_ = false;
  TaskContext context_;
};

}  // namespace task_priority_kinematic_control
