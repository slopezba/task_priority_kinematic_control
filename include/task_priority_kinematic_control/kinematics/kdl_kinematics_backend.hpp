#pragma once

#include "task_priority_kinematic_control/kinematics/kinematics_backend.hpp"

#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/tree.hpp>

#include <memory>
#include <unordered_map>

namespace task_priority_kinematic_control
{

class KDLKinematicsBackend : public KinematicsBackend
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
  struct ChainData
  {
    KDL::Chain chain;
    std::vector<std::string> joint_names;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver;
    std::unique_ptr<KDL::ChainJntToJacSolver> jac_solver;
  };

  static Eigen::Isometry3d to_isometry(const KDL::Frame & frame);
  static Eigen::MatrixXd to_eigen(const KDL::Jacobian & jacobian);
  ChainData build_chain_data(
    const KDL::Tree & tree,
    const std::string & base_frame,
    const std::string & tip_frame) const;
  Eigen::VectorXd q_for_chain(const ChainData & chain, const WholeBodyState & state) const;
  FrameState compute_frame_state(
    const ChainData & chain,
    const WholeBodyState & state,
    const std::string & group_name) const;

  WholeBodyModel model_;
  rclcpp::Logger logger_ = rclcpp::get_logger("KDLKinematicsBackend");
  WholeBodyState last_state_;
  KDL::Tree tree_;
  std::unordered_map<std::string, ChainData> chains_;
  std::unordered_map<std::string, FrameState> cached_frames_;
};

}  // namespace task_priority_kinematic_control
