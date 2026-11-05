#include "task_priority_kinematic_control/core/whole_body_model.hpp"

#include <algorithm>

namespace task_priority_kinematic_control
{

void WholeBodyModel::configure(
  const std::string & world_frame,
  const std::string & base_frame,
  const std::string & left_tip_frame,
  const std::string & right_tip_frame,
  const std::vector<std::string> & left_arm_joints,
  const std::vector<std::string> & right_arm_joints,
  const std::vector<int> & active_base_dofs)
{
  world_frame_ = world_frame;
  base_frame_ = base_frame;
  left_tip_frame_ = left_tip_frame;
  right_tip_frame_ = right_tip_frame;
  left_arm_joints_ = left_arm_joints;
  right_arm_joints_ = right_arm_joints;
  active_base_dofs_ = active_base_dofs;
  all_joint_names_ = left_arm_joints_;
  all_joint_names_.insert(all_joint_names_.end(), right_arm_joints_.begin(), right_arm_joints_.end());
}

size_t WholeBodyModel::total_dofs() const { return base_dofs() + left_arm_dofs() + right_arm_dofs(); }
size_t WholeBodyModel::base_dofs() const { return active_base_dofs_.size(); }
size_t WholeBodyModel::left_arm_dofs() const { return left_arm_joints_.size(); }
size_t WholeBodyModel::right_arm_dofs() const { return right_arm_joints_.size(); }
const std::string & WholeBodyModel::world_frame() const { return world_frame_; }
const std::string & WholeBodyModel::base_frame() const { return base_frame_; }
const std::string & WholeBodyModel::left_tip_frame() const { return left_tip_frame_; }
const std::string & WholeBodyModel::right_tip_frame() const { return right_tip_frame_; }
const std::vector<std::string> & WholeBodyModel::left_arm_joints() const { return left_arm_joints_; }
const std::vector<std::string> & WholeBodyModel::right_arm_joints() const { return right_arm_joints_; }
const std::vector<int> & WholeBodyModel::active_base_dofs() const { return active_base_dofs_; }
size_t WholeBodyModel::left_offset() const { return base_dofs(); }
size_t WholeBodyModel::right_offset() const { return base_dofs() + left_arm_dofs(); }

int WholeBodyModel::joint_index(const std::string & joint_name) const
{
  const auto it = std::find(all_joint_names_.begin(), all_joint_names_.end(), joint_name);
  if (it == all_joint_names_.end()) {
    return -1;
  }
  return static_cast<int>(std::distance(all_joint_names_.begin(), it));
}

std::vector<std::string> WholeBodyModel::all_joint_names() const
{
  return all_joint_names_;
}

}  // namespace task_priority_kinematic_control
