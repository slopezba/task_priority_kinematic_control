#pragma once
#include "task_priority_kinematic_control/core/initial_posture.hpp"
#include "task_priority_kinematic_control/core/nominal_trajectory.hpp"
#include "task_priority_kinematic_control/core/cartesian_trajectory_coordinator.hpp"
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <atomic>
#include <optional>
#include <thread>

namespace task_priority_kinematic_control
{
class PostureAction
{
public:
  using Handle = rclcpp_action::ServerGoalHandle<Preparation>;
  PostureAction(
    rclcpp_lifecycle::LifecycleNode::SharedPtr, std::mutex &, WholeBodyState &,
    std::shared_ptr<WholeBodyModel> &, KinematicsBackendPtr &, HierarchySolver &,
    TaskManager &, CartesianTrajectoryCoordinator &, std::function<bool()> ready);
  ~PostureAction();
  void interrupt() {cancel_ = true;}

private:
  void run(const std::shared_ptr<Handle> &);
  PostureOptions settings() const;
  NominalOptions nominal_settings() const;
  void fill_nominal(const NominalPlan &, Preparation::Result &) const;
  void apply(PosturePlan &, const Preparation::Goal &, const StpikCheck::Canceled &,
    bool check_trajectory = true);
  PosturePlan nominal_posture_plan(StpikSnapshot, const Preparation::Goal &, const StpikOptions &,
    const PostureOptions &, const NominalPlan &) const;
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::mutex & mutex_;
  WholeBodyState & state_;
  std::shared_ptr<WholeBodyModel> & model_;
  KinematicsBackendPtr & backend_;
  HierarchySolver & solver_;
  TaskManager & manager_;
  CartesianTrajectoryCoordinator & coordinator_;
  std::function<bool()> ready_;
  std::atomic_bool busy_{false}, cancel_{false};
  std::optional<PosturePlan> plan_;
  std::optional<NominalPlan> nominal_;
  rclcpp_action::Server<Preparation>::SharedPtr server_;
  std::thread worker_;
};
}
