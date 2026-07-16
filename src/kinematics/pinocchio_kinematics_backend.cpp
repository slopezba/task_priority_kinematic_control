#include "task_priority_kinematic_control/kinematics/pinocchio_kinematics_backend.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

namespace task_priority_kinematic_control
{

void PinocchioKinematicsBackend::configure(
  const WholeBodyModel & model,
  const std::string &,
  const rclcpp::Logger & logger)
{
  model_ = model;
  logger_ = logger;
  RCLCPP_WARN(
    logger_,
    "Pinocchio backend is a placeholder in this first implementation. Use the KDL backend for runtime execution.");
}

void PinocchioKinematicsBackend::update(const WholeBodyState &) {}

FrameState PinocchioKinematicsBackend::get_frame_state(const std::string &) const
{
  throw std::runtime_error("Pinocchio backend is not implemented yet");
}

Eigen::Isometry3d PinocchioKinematicsBackend::get_relative_transform(
  const std::string &,
  const std::string &) const
{
  throw std::runtime_error("Pinocchio backend is not implemented yet");
}

const std::vector<CollisionCapsule> & PinocchioKinematicsBackend::collision_capsules() const
{
  return collision_capsules_;
}

std::string PinocchioKinematicsBackend::name() const
{
  return "pinocchio";
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::PinocchioKinematicsBackend,
  task_priority_kinematic_control::KinematicsBackend)
