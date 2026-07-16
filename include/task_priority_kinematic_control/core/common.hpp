#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <map>
#include <string>
#include <vector>

namespace task_priority_kinematic_control
{

enum class SolverMethod
{
  kDls,
  kPinv,
  kSvd
};

struct FrameState
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  Eigen::MatrixXd jacobian;
};

struct WholeBodyState
{
  Eigen::Vector3d base_position = Eigen::Vector3d::Zero();
  Eigen::Vector3d base_rpy = Eigen::Vector3d::Zero();
  Eigen::VectorXd joint_positions;
  Eigen::VectorXd joint_velocities;
  bool navigation_valid = false;
  bool joints_valid = false;
};

struct WholeBodyCommand
{
  Eigen::VectorXd generalized_velocity;
  Eigen::VectorXd base_velocity;
  Eigen::VectorXd left_arm_velocity;
  Eigen::VectorXd right_arm_velocity;
};

struct TaskComputation
{
  Eigen::MatrixXd jacobian;
  Eigen::VectorXd desired_velocity;
  Eigen::VectorXd error;
  Eigen::Isometry3d frame_pose = Eigen::Isometry3d::Identity();
  bool active = false;
  bool has_frame_pose = false;
  std::string frame_id;
  std::string status_message;
};

struct CollisionCapsule
{
  std::string link_name;
  std::string parent_link_name;
  std::string source;
  Eigen::Vector3d local_a = Eigen::Vector3d::Zero();
  Eigen::Vector3d local_b = Eigen::Vector3d::Zero();
  double radius = 0.0;
};

inline Eigen::Matrix3d skew(const Eigen::Vector3d & v)
{
  Eigen::Matrix3d out = Eigen::Matrix3d::Zero();
  out(0, 1) = -v.z();
  out(0, 2) = v.y();
  out(1, 0) = v.z();
  out(1, 2) = -v.x();
  out(2, 0) = -v.y();
  out(2, 1) = v.x();
  return out;
}

}  // namespace task_priority_kinematic_control
