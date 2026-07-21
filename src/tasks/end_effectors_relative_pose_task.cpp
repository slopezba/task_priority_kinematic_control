#include "task_priority_kinematic_control/tasks/end_effectors_relative_pose_task.hpp"

#include <pluginlib/class_list_macros.hpp>

namespace task_priority_kinematic_control
{

namespace
{

Eigen::Vector3d quaternion_error(
  const Eigen::Quaterniond & target,
  const Eigen::Quaterniond & current)
{
  const Eigen::Quaterniond q_err = target * current.conjugate();
  Eigen::AngleAxisd aa(q_err);
  return aa.axis() * aa.angle();
}

Eigen::Isometry3d pose_goal_to_isometry(const geometry_msgs::msg::PoseStamped & goal)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(
    goal.pose.position.x,
    goal.pose.position.y,
    goal.pose.position.z);
  const Eigen::Quaterniond q(
    goal.pose.orientation.w,
    goal.pose.orientation.x,
    goal.pose.orientation.y,
    goal.pose.orientation.z);
  pose.linear() = q.normalized().toRotationMatrix();
  return pose;
}

Eigen::Isometry3d relative_pose_from_values(const std::vector<double> & values)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  if (values.size() != 7) {
    return pose;
  }
  pose.translation() = Eigen::Vector3d(values[0], values[1], values[2]);
  const Eigen::Quaterniond q(values[6], values[3], values[4], values[5]);
  pose.linear() = q.normalized().toRotationMatrix();
  return pose;
}

std::vector<double> activation_values_from_parameters(
  const std::map<std::string, rclcpp::Parameter> & parameters)
{
  const auto it = parameters.find("activation");
  if (it == parameters.end()) {
    return {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  }
  if (it->second.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY) {
    const auto values = it->second.as_integer_array();
    return std::vector<double>(values.begin(), values.end());
  }
  if (it->second.get_type() == rclcpp::ParameterType::PARAMETER_BOOL_ARRAY) {
    const auto values = it->second.as_bool_array();
    std::vector<double> out;
    out.reserve(values.size());
    for (const bool value : values) {
      out.push_back(value ? 1.0 : 0.0);
    }
    return out;
  }
  return it->second.as_double_array();
}

}  // namespace

void EndEffectorsRelativePoseTask::configure(
  const std::string & id,
  const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  reference_frame_ = get_string_param(parameters, "reference_frame", context.model->left_tip_frame());
  controlled_frame_ = get_string_param(parameters, "controlled_frame", context.model->right_tip_frame());

  const auto gains = get_double_array_param(parameters, "gain", {1.0, 1.0, 1.0, 0.7, 0.7, 0.7});
  for (size_t i = 0; i < 6 && i < gains.size(); ++i) {
    gains_(static_cast<Eigen::Index>(i)) = gains[i];
  }

  const auto activation = activation_values_from_parameters(parameters);
  for (size_t i = 0; i < 6 && i < activation.size(); ++i) {
    activation_(static_cast<Eigen::Index>(i)) = activation[i] != 0.0;
  }

  const auto default_pose = get_double_array_param(parameters, "default_relative_pose", {});
  if (default_pose.size() == 7) {
    target_relative_pose_ = relative_pose_from_values(default_pose);
    has_target_relative_pose_ = true;
  }
}

TaskComputation EndEffectorsRelativePoseTask::update(
  const WholeBodyState &,
  const KinematicsBackend & backend)
{
  TaskComputation computation;
  computation.active = enabled_;
  if (!enabled_) {
    computation.status_message = "disabled";
    return computation;
  }

  const FrameState reference = backend.get_frame_state(reference_frame_);
  const FrameState controlled = backend.get_frame_state(controlled_frame_);
  const Eigen::Isometry3d current_relative_pose = reference.pose.inverse() * controlled.pose;

  if (!has_target_relative_pose_) {
    target_relative_pose_ = current_relative_pose;
    has_target_relative_pose_ = true;
  }

  Eigen::Matrix<double, 6, 1> full_error = Eigen::Matrix<double, 6, 1>::Zero();
  full_error.head<3>() =
    target_relative_pose_.translation() - current_relative_pose.translation();
  full_error.tail<3>() = quaternion_error(
    Eigen::Quaterniond(target_relative_pose_.rotation()).normalized(),
    Eigen::Quaterniond(current_relative_pose.rotation()).normalized());

  Eigen::MatrixXd full_jacobian = Eigen::MatrixXd::Zero(6, reference.jacobian.cols());
  if (controlled.jacobian.cols() == reference.jacobian.cols() &&
    reference.jacobian.rows() >= 6 && controlled.jacobian.rows() >= 6)
  {
    const Eigen::Matrix3d ref_rotation_world_to_local = reference.pose.rotation().transpose();
    const Eigen::Vector3d relative_translation_world =
      controlled.pose.translation() - reference.pose.translation();
    const auto ref_linear = reference.jacobian.topRows(3);
    const auto ref_angular = reference.jacobian.bottomRows(3);
    const auto controlled_linear = controlled.jacobian.topRows(3);
    const auto controlled_angular = controlled.jacobian.bottomRows(3);

    full_jacobian.topRows(3) =
      ref_rotation_world_to_local *
      (controlled_linear - ref_linear + skew(relative_translation_world) * ref_angular);
    full_jacobian.bottomRows(3) =
      ref_rotation_world_to_local * (controlled_angular - ref_angular);
  }

  std::vector<Eigen::Index> active_indices;
  active_indices.reserve(6);
  for (Eigen::Index i = 0; i < 6; ++i) {
    if (activation_(i)) {
      active_indices.push_back(i);
    }
  }

  computation.has_frame_pose = true;
  computation.frame_id = controlled_frame_;
  computation.frame_pose = current_relative_pose;
  if (active_indices.empty()) {
    computation.active = false;
    computation.status_message = "no active axes";
    return computation;
  }

  computation.error = Eigen::VectorXd::Zero(active_indices.size());
  computation.desired_velocity = Eigen::VectorXd::Zero(active_indices.size());
  computation.jacobian = Eigen::MatrixXd::Zero(
    static_cast<Eigen::Index>(active_indices.size()), full_jacobian.cols());
  for (size_t row = 0; row < active_indices.size(); ++row) {
    const Eigen::Index source_row = active_indices[row];
    computation.error(static_cast<Eigen::Index>(row)) = full_error(source_row);
    computation.desired_velocity(static_cast<Eigen::Index>(row)) =
      gains_(source_row) * full_error(source_row);
    computation.jacobian.row(static_cast<Eigen::Index>(row)) = full_jacobian.row(source_row);
  }
  computation.status_message = "tracking";
  return computation;
}

bool EndEffectorsRelativePoseTask::set_pose_goal(const geometry_msgs::msg::PoseStamped & goal)
{
  target_relative_pose_ = pose_goal_to_isometry(goal);
  has_target_relative_pose_ = true;
  pose_goal_ = goal;
  has_pose_goal_ = true;
  return true;
}

bool EndEffectorsRelativePoseTask::set_gain(
  const std::vector<double> & gain,
  std::string & message)
{
  if (gain.size() != 6) {
    message = "Relative pose gain must contain 6 values";
    return false;
  }
  for (size_t i = 0; i < gain.size(); ++i) {
    if (gain[i] < 0.0) {
      message = "Relative pose gain values must be non-negative";
      return false;
    }
    gains_(static_cast<Eigen::Index>(i)) = gain[i];
  }
  message = "Relative pose gain updated";
  return true;
}

msg::TaskStatus EndEffectorsRelativePoseTask::build_status() const
{
  auto status = TaskBaseCommon::build_status();
  status.target_type = "pose";
  return status;
}

std::vector<double> EndEffectorsRelativePoseTask::current_target() const
{
  const Eigen::Quaterniond q(target_relative_pose_.rotation());
  return {
    target_relative_pose_.translation().x(),
    target_relative_pose_.translation().y(),
    target_relative_pose_.translation().z(),
    q.x(),
    q.y(),
    q.z(),
    q.w()};
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::EndEffectorsRelativePoseTask,
  task_priority_kinematic_control::TaskBase)
