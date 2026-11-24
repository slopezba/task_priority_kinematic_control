#pragma once

#include "task_priority_kinematic_control/msg/facing_target.hpp"
#include "task_priority_kinematic_control/tasks/task_base.hpp"

namespace task_priority_kinematic_control
{

// Makes two end effectors face each other along the line that joins them.
//
// Rows (fixed order, 6 DOF):
//   0-1  left_pointing_axis (left EE frame) aligned with +d (towards the right EE)
//   2-3  right_pointing_axis (right EE frame) aligned with -d (towards the left EE)
//   4    relative roll around d (left -> right) tracks roll_angle
//   5    distance between both EE tracks distance
// where d = (p_right - p_left) / ||p_right - p_left||.
//
// Pointing axes must be pure signed axes (+-x, +-y, +-z). The zero of the roll uses a cyclic
// reference axis per EE: +-x -> +-y, +-y -> +-z, +-z -> +-x (same sign as the pointing axis).
// The roll is the signed angle around d from the left reference axis to the right one, both
// projected onto the plane orthogonal to d.
class EndEffectorsFacingTask : public TaskBaseCommon
{
  friend class StpikTaskAccess;

public:
  void configure(
    const std::string & id,
    const std::string & plugin_name,
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const TaskContext & context) override;

  TaskComputation update(
    const WholeBodyState & state,
    const KinematicsBackend & backend) override;

  bool set_facing_target(const msg::FacingTarget & target, std::string & message);
  bool set_gain(const std::vector<double> & gain, std::string & message) override;
  msg::TaskStatus build_status() const override;
  std::vector<double> current_target() const override;
  // Keeps the pointing axes and captures the measured roll and distance as the new target.
  void capture_measured_target(const KinematicsBackend & backend);

  struct Measurement
  {
    bool valid = false;           // false when the EE are closer than min_distance
    double distance = 0.0;
    bool roll_valid = false;      // false when a roll reference is (almost) parallel to d
    double roll = 0.0;
  };
  Measurement measure(const KinematicsBackend & backend) const;

  // Returns false (and fills message) if axis is not a pure signed axis.
  static bool pure_axis_index(
    const Eigen::Vector3d & axis, int & index, double & sign, std::string & message);

private:
  std::string left_frame_;
  std::string right_frame_;
  Eigen::Vector3d left_pointing_axis_ = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d right_pointing_axis_ = Eigen::Vector3d::UnitZ();
  int left_axis_index_ = 2;
  int right_axis_index_ = 2;
  double left_axis_sign_ = 1.0;
  double right_axis_sign_ = 1.0;
  double target_roll_ = 0.0;
  double target_distance_ = 0.0;
  bool capture_roll_pending_ = true;
  bool capture_distance_pending_ = true;
  double min_distance_ = 0.02;
  Eigen::Matrix<double, 6, 1> gains_ = Eigen::Matrix<double, 6, 1>::Ones();
  Eigen::Matrix<bool, 6, 1> activation_ = Eigen::Matrix<bool, 6, 1>::Constant(true);
  // Last measured values, only for status reporting.
  double measured_roll_ = 0.0;
  double measured_distance_ = 0.0;
};

}  // namespace task_priority_kinematic_control
