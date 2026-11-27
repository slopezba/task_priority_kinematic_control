#pragma once

#include "sura_manipulation_actions/facing_geometry.hpp"
#include "sura_manipulation_actions/msg/facing_reference.hpp"
#include "task_priority_kinematic_control/msg/facing_target.hpp"
#include "task_priority_kinematic_control/tasks/task_base.hpp"

#include <chrono>

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
// Goal pointing axes must be pure signed axes (+-x, +-y, +-z); a streamed FacingReference may
// move the pointing reference through any unit vector and adds an expiring feedforward.
// The geometry is shared with the bimanual planner (facing_geometry.hpp).
// The zero of the roll uses a cyclic reference axis per goal axis: +-x -> +-y, +-y -> +-z, +-z -> +-x (same sign as the pointing axis).
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
  bool set_facing_reference(
    const sura_manipulation_actions::msg::FacingReference & reference, std::string & message);
  bool set_gain(const std::vector<double> & gain, std::string & message) override;
  bool set_enabled(bool enabled) override;
  void reset() override;
  const std::string & left_frame() const {return left_frame_;}
  const std::string & right_frame() const {return right_frame_;}
  double min_distance() const {return min_distance_;}
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
  void restore_goal_reference();
  sura_manipulation_actions::facing::Reference reference() const;

  std::string left_frame_;
  std::string right_frame_;
  sura_manipulation_actions::facing::PureAxis left_goal_;
  sura_manipulation_actions::facing::PureAxis right_goal_;
  // Current pointing reference, equal to the goal axes unless a reference is streamed.
  Eigen::Vector3d left_reference_axis_ = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d right_reference_axis_ = Eigen::Vector3d::UnitZ();
  Eigen::Matrix<double, 6, 1> feedforward_ = Eigen::Matrix<double, 6, 1>::Zero();
  std::chrono::steady_clock::time_point feedforward_time_;
  double feedforward_timeout_ = 0.5;
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
