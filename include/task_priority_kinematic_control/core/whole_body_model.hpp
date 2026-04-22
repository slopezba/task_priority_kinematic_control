#pragma once

#include "task_priority_kinematic_control/core/common.hpp"

#include <Eigen/Core>

#include <string>
#include <vector>

namespace task_priority_kinematic_control
{

class WholeBodyModel
{
public:
  void configure(
    const std::string & world_frame,
    const std::string & base_frame,
    const std::string & left_tip_frame,
    const std::string & right_tip_frame,
    const std::vector<std::string> & left_arm_joints,
    const std::vector<std::string> & right_arm_joints,
    const std::vector<int> & active_base_dofs);

  size_t total_dofs() const;
  size_t base_dofs() const;
  size_t left_arm_dofs() const;
  size_t right_arm_dofs() const;

  const std::string & world_frame() const;
  const std::string & base_frame() const;
  const std::string & left_tip_frame() const;
  const std::string & right_tip_frame() const;
  const std::vector<std::string> & left_arm_joints() const;
  const std::vector<std::string> & right_arm_joints() const;
  const std::vector<int> & active_base_dofs() const;

  size_t left_offset() const;
  size_t right_offset() const;

  int joint_index(const std::string & joint_name) const;
  std::vector<std::string> all_joint_names() const;

private:
  std::string world_frame_;
  std::string base_frame_;
  std::string left_tip_frame_;
  std::string right_tip_frame_;
  std::vector<std::string> left_arm_joints_;
  std::vector<std::string> right_arm_joints_;
  std::vector<int> active_base_dofs_;
  std::vector<std::string> all_joint_names_;
};

}  // namespace task_priority_kinematic_control
