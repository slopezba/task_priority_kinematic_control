#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"
#include "sura_manipulation_actions/base_trajectory_sampling.hpp"
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>
#include <mutex>

namespace task_priority_kinematic_control
{

// References and twists are always expressed at the base origin in world axes.
class BasePoseTrajectoryTask : public TaskBaseCommon
{
public:
  using Coordinates = sura_manipulation_actions::BaseCoordinates;
  void configure(const std::string &, const std::string &,
    const std::map<std::string, rclcpp::Parameter> &, const TaskContext &) override;
  TaskComputation update(const WholeBodyState &, const KinematicsBackend &) override;
  bool validate_trajectory(const trajectory_msgs::msg::MultiDOFJointTrajectory &,
    std::string &) const;
  bool set_base_trajectory(const trajectory_msgs::msg::MultiDOFJointTrajectory &,
    const std::vector<double> & goal_tolerance, double timeout, std::string &);
  bool cancel_base_trajectory(std::string &);
  void set_motion_limits(const std::vector<double> & max_velocity,
    const std::vector<double> & max_acceleration);
  void hold_measured(const WholeBodyState &);
  JointTrajectoryTaskStatus base_trajectory_status() const;
  bool set_enabled(bool) override;
  bool enabled() const override;
  bool set_gain(const std::vector<double> &, std::string &) override;
  void reset() override;
  msg::TaskStatus build_status() const override;
  std::vector<double> current_target() const override;

private:
  mutable std::mutex mutex_;
  trajectory_msgs::msg::MultiDOFJointTrajectory trajectory_;
  std::vector<sura_manipulation_actions::BaseTrajectoryKnot> knots_;
  Coordinates gains_ = Coordinates::Ones();
  Coordinates tolerance_ = Coordinates::Constant(0.05);
  Coordinates target_ = Coordinates::Zero();
  Coordinates last_error_ = Coordinates::Zero();
  Coordinates max_velocity_ = (Coordinates() << 0.2, 0.2, 0.2, 0.3, 0.3, 0.3).finished();
  Coordinates max_acceleration_ = (Coordinates() << 0.2, 0.2, 0.2, 0.3, 0.3, 0.3).finished();
  Coordinates previous_command_ = Coordinates::Zero();
  int64_t command_time_ns_ = 0;
  int64_t start_ns_ = 0;
  int64_t previous_ns_ = 0;
  double timeout_ = 0.5;
  bool installed_ = false;
  bool succeeded_ = false;
  bool failed_ = false;
  bool holding_ = false;
  bool ros_cycle_seen_ = false;
  std::string message_ = "idle";
};

}  // namespace task_priority_kinematic_control
