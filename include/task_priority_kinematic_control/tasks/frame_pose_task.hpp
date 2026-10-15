#pragma once

#include "task_priority_kinematic_control/tasks/pose_task_common.hpp"

#include "task_priority_kinematic_control/tasks/task_base.hpp"

namespace task_priority_kinematic_control
{

class FramePoseTask : public PoseTaskCommon
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
  bool roll_pitch_control() const {return roll_pitch_control_;}
  bool set_allow_roll_pitch(bool allow, std::string & message);
  bool set_enabled(bool enabled) override;
  void reset() override;

private:
  std::string frame_id_;
  PoseVelocity activation_ = PoseVelocity::Ones();
  bool roll_pitch_control_ = false;
  bool allow_roll_pitch_ = false;
  bool tilt_initialized_ = false;
  bool tilt_moving_ = false;
  Eigen::Vector2d tilt_target_ = Eigen::Vector2d::Zero();
  double tilt_threshold_ = 0.001;
  Eigen::Matrix<double, 6, 1> gains_ = Eigen::Matrix<double, 6, 1>::Ones();
};

}  // namespace task_priority_kinematic_control
