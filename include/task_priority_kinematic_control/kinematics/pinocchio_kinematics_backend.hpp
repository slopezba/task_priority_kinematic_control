#pragma once

#include "task_priority_kinematic_control/kinematics/kinematics_backend.hpp"

namespace task_priority_kinematic_control
{

class PinocchioKinematicsBackend : public KinematicsBackend
{
public:
  void configure(
    const WholeBodyModel & model,
    const std::string & robot_description,
    const rclcpp::Logger & logger) override;

  void update(const WholeBodyState & state) override;

  FrameState get_frame_state(const std::string & frame_id) const override;

  Eigen::Isometry3d get_relative_transform(
    const std::string & from_frame,
    const std::string & to_frame) const override;

  std::string name() const override;

private:
  WholeBodyModel model_;
  rclcpp::Logger logger_ = rclcpp::get_logger("PinocchioKinematicsBackend");
};

}  // namespace task_priority_kinematic_control
