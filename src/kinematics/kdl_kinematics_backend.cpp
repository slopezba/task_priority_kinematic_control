#include "task_priority_kinematic_control/kinematics/kdl_kinematics_backend.hpp"

#include <kdl/jntarray.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/logging.hpp>

#include <sstream>

namespace task_priority_kinematic_control
{

namespace
{

std::string describe_chain(const KDL::Chain & chain)
{
  std::ostringstream stream;
  for (unsigned int i = 0; i < chain.getNrOfSegments(); ++i) {
    if (i > 0) {
      stream << " -> ";
    }
    const auto & segment = chain.getSegment(i);
    stream << segment.getName();
    const auto & joint = segment.getJoint();
    if (joint.getType() != KDL::Joint::None) {
      stream << "[" << joint.getName() << "]";
    }
  }
  return stream.str();
}

}  // namespace

void KDLKinematicsBackend::configure(
  const WholeBodyModel & model,
  const std::string & robot_description,
  const rclcpp::Logger & logger)
{
  model_ = model;
  logger_ = logger;
  if (!kdl_parser::treeFromString(robot_description, tree_)) {
    throw std::runtime_error("Failed to parse robot_description into a KDL tree");
  }

  chains_.clear();
  for (const auto & entry : tree_.getSegments()) {
    const std::string & frame_id = entry.first;
    if (frame_id == model_.base_frame()) {
      continue;
    }
    try {
      chains_.emplace(frame_id, build_chain_data(tree_, model_.base_frame(), frame_id));
    } catch (const std::exception &) {
      continue;
    }
  }
  RCLCPP_INFO(
    logger_,
    "KDL left chain (%s -> %s): %s",
    model_.base_frame().c_str(),
    model_.left_tip_frame().c_str(),
    describe_chain(chains_.at(model_.left_tip_frame()).chain).c_str());
  RCLCPP_INFO(
    logger_,
    "KDL right chain (%s -> %s): %s",
    model_.base_frame().c_str(),
    model_.right_tip_frame().c_str(),
    describe_chain(chains_.at(model_.right_tip_frame()).chain).c_str());
}

void KDLKinematicsBackend::update(const WholeBodyState & state)
{
  last_state_ = state;
  cached_frames_.clear();
  for (const auto & entry : chains_) {
    const std::string & frame_id = entry.first;
    const std::string group_name = frame_id.rfind("cirtesub/alpha_left/", 0) == 0 ? "left" : "right";
    cached_frames_.emplace(frame_id, compute_frame_state(entry.second, state, group_name));
  }
}

FrameState KDLKinematicsBackend::get_frame_state(const std::string & frame_id) const
{
  const auto it = cached_frames_.find(frame_id);
  if (it == cached_frames_.end()) {
    throw std::runtime_error("Requested frame is not available in the KDL backend cache: " + frame_id);
  }
  return it->second;
}

Eigen::Isometry3d KDLKinematicsBackend::get_relative_transform(
  const std::string & from_frame,
  const std::string & to_frame) const
{
  const auto from = get_frame_state(from_frame).pose;
  const auto to = get_frame_state(to_frame).pose;
  return from.inverse() * to;
}

std::string KDLKinematicsBackend::name() const
{
  return "kdl";
}

Eigen::Isometry3d KDLKinematicsBackend::to_isometry(const KDL::Frame & frame)
{
  Eigen::Isometry3d out = Eigen::Isometry3d::Identity();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      out.linear()(row, col) = frame.M(row, col);
    }
    out.translation()(row) = frame.p(row);
  }
  return out;
}

Eigen::MatrixXd KDLKinematicsBackend::to_eigen(const KDL::Jacobian & jacobian)
{
  Eigen::MatrixXd out(6, jacobian.columns());
  for (unsigned int col = 0; col < jacobian.columns(); ++col) {
    for (unsigned int row = 0; row < 6; ++row) {
      out(row, col) = jacobian(row, col);
    }
  }
  return out;
}

KDLKinematicsBackend::ChainData KDLKinematicsBackend::build_chain_data(
  const KDL::Tree & tree,
  const std::string & base_frame,
  const std::string & tip_frame) const
{
  ChainData data;
  if (!tree.getChain(base_frame, tip_frame, data.chain)) {
    throw std::runtime_error("Failed to extract KDL chain from " + base_frame + " to " + tip_frame);
  }
  for (unsigned int i = 0; i < data.chain.getNrOfSegments(); ++i) {
    const auto & joint = data.chain.getSegment(i).getJoint();
    if (joint.getType() != KDL::Joint::None) {
      data.joint_names.push_back(joint.getName());
    }
  }
  data.fk_solver = std::make_unique<KDL::ChainFkSolverPos_recursive>(data.chain);
  data.jac_solver = std::make_unique<KDL::ChainJntToJacSolver>(data.chain);
  return data;
}

Eigen::VectorXd KDLKinematicsBackend::q_for_chain(
  const ChainData & chain,
  const WholeBodyState & state) const
{
  Eigen::VectorXd q = Eigen::VectorXd::Zero(chain.joint_names.size());
  for (size_t i = 0; i < chain.joint_names.size(); ++i) {
    const int joint_idx = model_.joint_index(chain.joint_names[i]);
    if (joint_idx >= 0 && joint_idx < state.joint_positions.size()) {
      q(static_cast<Eigen::Index>(i)) = state.joint_positions(joint_idx);
    }
  }
  return q;
}

FrameState KDLKinematicsBackend::compute_frame_state(
  const ChainData & chain,
  const WholeBodyState & state,
  const std::string & group_name) const
{
  FrameState frame_state;
  const Eigen::VectorXd q = q_for_chain(chain, state);
  KDL::JntArray q_kdl(chain.joint_names.size());
  for (size_t i = 0; i < chain.joint_names.size(); ++i) {
    q_kdl(static_cast<unsigned int>(i)) = q(static_cast<Eigen::Index>(i));
  }

  KDL::Frame frame;
  chain.fk_solver->JntToCart(q_kdl, frame);
  frame_state.pose = to_isometry(frame);

  KDL::Jacobian jacobian(chain.joint_names.size());
  chain.jac_solver->JntToJac(q_kdl, jacobian);
  const Eigen::MatrixXd manipulator_jacobian = to_eigen(jacobian);

  frame_state.jacobian = Eigen::MatrixXd::Zero(6, model_.total_dofs());
  const Eigen::Vector3d p = frame_state.pose.translation();
  Eigen::Matrix<double, 6, 6> base_jacobian = Eigen::Matrix<double, 6, 6>::Zero();
  base_jacobian.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity();
  base_jacobian.topRightCorner<3, 3>() = -skew(p);
  base_jacobian.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Identity();

  for (size_t i = 0; i < model_.active_base_dofs().size(); ++i) {
    const int dof = model_.active_base_dofs()[i];
    if (dof >= 0 && dof < 6) {
      frame_state.jacobian.col(static_cast<Eigen::Index>(i)) = base_jacobian.col(dof);
    }
  }

  const size_t offset = group_name == "left" ? model_.left_offset() : model_.right_offset();
  frame_state.jacobian.block(0, static_cast<Eigen::Index>(offset), 6, manipulator_jacobian.cols()) =
    manipulator_jacobian;
  return frame_state;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::KDLKinematicsBackend,
  task_priority_kinematic_control::KinematicsBackend)
