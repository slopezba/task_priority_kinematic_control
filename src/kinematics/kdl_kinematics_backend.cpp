#include "task_priority_kinematic_control/kinematics/kdl_kinematics_backend.hpp"

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

Eigen::Isometry3d base_pose_from_state(const WholeBodyState & state)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = state.base_position;
  const Eigen::AngleAxisd roll(state.base_rpy.x(), Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch(state.base_rpy.y(), Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw(state.base_rpy.z(), Eigen::Vector3d::UnitZ());
  pose.linear() = (yaw * pitch * roll).toRotationMatrix();
  return pose;
}

Eigen::Vector3d to_eigen(const KDL::Vector & vector)
{
  return Eigen::Vector3d(vector.x(), vector.y(), vector.z());
}

bool is_revolute(KDL::Joint::JointType joint_type)
{
  switch (joint_type) {
    case KDL::Joint::RotAxis:
    case KDL::Joint::RotX:
    case KDL::Joint::RotY:
    case KDL::Joint::RotZ:
      return true;
    default:
      return false;
  }
}

bool is_prismatic(KDL::Joint::JointType joint_type)
{
  switch (joint_type) {
    case KDL::Joint::TransAxis:
    case KDL::Joint::TransX:
    case KDL::Joint::TransY:
    case KDL::Joint::TransZ:
      return true;
    default:
      return false;
  }
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
    cached_frames_.emplace(frame_id, compute_frame_state(frame_id, entry.second, state));
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
  const std::string &,
  const ChainData & chain,
  const WholeBodyState & state) const
{
  FrameState frame_state;
  const Eigen::VectorXd q = q_for_chain(chain, state);

  KDL::Frame base_to_tip_kdl = KDL::Frame::Identity();
  std::vector<Eigen::Vector3d> joint_origins;
  std::vector<Eigen::Vector3d> joint_axes;
  std::vector<KDL::Joint::JointType> joint_types;
  joint_origins.reserve(chain.joint_names.size());
  joint_axes.reserve(chain.joint_names.size());
  joint_types.reserve(chain.joint_names.size());

  size_t q_index = 0;
  for (unsigned int segment_index = 0; segment_index < chain.chain.getNrOfSegments(); ++segment_index) {
    const auto & segment = chain.chain.getSegment(segment_index);
    const auto & joint = segment.getJoint();
    double q_value = 0.0;

    if (joint.getType() != KDL::Joint::None) {
      if (q_index < static_cast<size_t>(q.size())) {
        q_value = q(static_cast<Eigen::Index>(q_index));
      }

      const Eigen::Isometry3d base_to_segment_root = to_isometry(base_to_tip_kdl);
      const Eigen::Vector3d joint_origin =
        base_to_segment_root * to_eigen(joint.JointOrigin());
      const Eigen::Vector3d joint_axis =
        (base_to_segment_root.rotation() * to_eigen(joint.JointAxis())).normalized();
      joint_origins.push_back(joint_origin);
      joint_axes.push_back(joint_axis);
      joint_types.push_back(joint.getType());
      ++q_index;
    }

    base_to_tip_kdl = base_to_tip_kdl * segment.pose(q_value);
  }

  const Eigen::Isometry3d base_to_tip = to_isometry(base_to_tip_kdl);
  const Eigen::Isometry3d world_to_base = base_pose_from_state(state);
  frame_state.pose = world_to_base * base_to_tip;

  frame_state.jacobian = Eigen::MatrixXd::Zero(6, model_.total_dofs());
  const Eigen::Vector3d p = frame_state.pose.translation() - state.base_position;
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

  for (size_t chain_col = 0; chain_col < chain.joint_names.size(); ++chain_col) {
    const int joint_idx = model_.joint_index(chain.joint_names[chain_col]);
    if (joint_idx < 0) {
      continue;
    }
    const Eigen::Index model_col =
      static_cast<Eigen::Index>(model_.base_dofs() + static_cast<size_t>(joint_idx));
    if (model_col < frame_state.jacobian.cols() &&
      chain_col < joint_origins.size() &&
      chain_col < joint_axes.size() &&
      chain_col < joint_types.size())
    {
      const Eigen::Vector3d world_axis = world_to_base.rotation() * joint_axes[chain_col];
      const Eigen::Vector3d world_joint_origin =
        state.base_position + world_to_base.rotation() * joint_origins[chain_col];
      if (is_revolute(joint_types[chain_col])) {
        frame_state.jacobian.col(model_col).head<3>() =
          world_axis.cross(frame_state.pose.translation() - world_joint_origin);
        frame_state.jacobian.col(model_col).tail<3>() = world_axis;
      } else if (is_prismatic(joint_types[chain_col])) {
        frame_state.jacobian.col(model_col).head<3>() = world_axis;
      }
    }
  }
  return frame_state;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::KDLKinematicsBackend,
  task_priority_kinematic_control::KinematicsBackend)
