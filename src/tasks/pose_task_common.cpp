#include "task_priority_kinematic_control/tasks/pose_task_common.hpp"
#include <cmath>
#include <stdexcept>

namespace task_priority_kinematic_control
{
void PoseTaskCommon::configure_pose_feedforward(
  const std::map<std::string, rclcpp::Parameter> & parameters)
{
  auto_update_target_pose_ = get_bool_param(parameters, "auto_update_target_pose", true);
  feedforward_velocity_threshold_ = get_double_param(
    parameters, "feedforward_velocity_threshold", 0.001);
  if (!std::isfinite(feedforward_velocity_threshold_) || feedforward_velocity_threshold_ < 0.0) {
    throw std::invalid_argument("feedforward_velocity_threshold must be finite and non-negative");
  }
  accept_feedforward_ = enabled_;
  feedforward_timeout_ = get_double_param(parameters, "feedforward_timeout", 0.0);
  if (!std::isfinite(feedforward_timeout_) || feedforward_timeout_ < 0.0) {
    throw std::invalid_argument("feedforward_timeout must be finite and non-negative");
  }
  reset();
}

bool PoseTaskCommon::set_feedforward_option(
  const std::string & field, const rclcpp::Parameter & parameter, std::string & message)
{
  if (field == "auto_update_target_pose" && parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL) {
    const bool automatic = parameter.as_bool();
    if (automatic && !auto_update_target_pose_) {initialize_target_ = true;}
    auto_update_target_pose_ = automatic;
    message = "Pose target update option changed";
    return true;
  }
  if (field == "feedforward_velocity_threshold" && parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
    const double threshold = parameter.as_double();
    if (std::isfinite(threshold) && threshold >= 0.0) {
      feedforward_velocity_threshold_ = threshold;
      message = "Pose feedforward threshold changed";
      return true;
    }
  }
  message = "Invalid pose feedforward option: expected bool or finite non-negative threshold";
  return false;
}

uint64_t PoseTaskCommon::feedforward_epoch() const
{
  std::scoped_lock lock(feedforward_mutex_);
  return feedforward_epoch_;
}

bool PoseTaskCommon::set_feedforward(const PoseVelocity & velocity, uint64_t epoch)
{
  std::scoped_lock lock(feedforward_mutex_);
  if (!accept_feedforward_ || epoch != feedforward_epoch_) {return false;}
  if (!velocity.allFinite()) {feedforward_.setZero(); return false;}
  feedforward_ = velocity;
  feedforward_time_ = std::chrono::steady_clock::now();
  return true;
}

void PoseTaskCommon::clear_feedforward()
{
  std::scoped_lock lock(feedforward_mutex_);
  ++feedforward_epoch_;
  feedforward_.setZero();
}

void PoseTaskCommon::capture_measured_target(const Eigen::Isometry3d & pose)
{
  clear_feedforward();
  pose_goal_.header.frame_id = context_.model->world_frame();
  pose_goal_.pose.position.x = pose.translation().x();
  pose_goal_.pose.position.y = pose.translation().y();
  pose_goal_.pose.position.z = pose.translation().z();
  const Eigen::Quaterniond q(pose.rotation());
  pose_goal_.pose.orientation.x = q.x(); pose_goal_.pose.orientation.y = q.y();
  pose_goal_.pose.orientation.z = q.z(); pose_goal_.pose.orientation.w = q.w();
  has_pose_goal_ = true;
  initialize_target_ = false;
}

bool PoseTaskCommon::set_enabled(bool enabled)
{
  std::scoped_lock lock(feedforward_mutex_);
  if (enabled != enabled_) {
    ++feedforward_epoch_;
    feedforward_.setZero();
    initialize_target_ = true;
  }
  accept_feedforward_ = enabled;
  enabled_ = enabled;
  return true;
}

void PoseTaskCommon::reset()
{
  clear_feedforward();
  initialize_target_ = true;
}

void PoseTaskCommon::update_pose_target(
  const Eigen::Isometry3d & pose, TaskComputation & computation,
  const PoseVelocity & activation)
{
  computation.feedforward = pose_feedforward().cwiseProduct(activation);
  if (auto_update_target_pose_ &&
    (initialize_target_ || computation.feedforward.norm() > feedforward_velocity_threshold_))
  {
    pose_goal_.header.frame_id = context_.model->world_frame();
    pose_goal_.pose.position.x = pose.translation().x();
    pose_goal_.pose.position.y = pose.translation().y();
    pose_goal_.pose.position.z = pose.translation().z();
    const Eigen::Quaterniond q(pose.rotation());
    pose_goal_.pose.orientation.x = q.x();
    pose_goal_.pose.orientation.y = q.y();
    pose_goal_.pose.orientation.z = q.z();
    pose_goal_.pose.orientation.w = q.w();
    has_pose_goal_ = true;
    computation.has_updated_target = true;
    computation.updated_target = pose_goal_.pose;
  }
  initialize_target_ = false;
}

PoseVelocity PoseTaskCommon::pose_feedforward() const
{
  std::scoped_lock lock(feedforward_mutex_);
  if (feedforward_timeout_ > 0.0 && std::chrono::duration<double>(
      std::chrono::steady_clock::now() - feedforward_time_).count() > feedforward_timeout_) {
    return PoseVelocity::Zero();
  }
  return feedforward_;
}

std::vector<double> PoseTaskCommon::current_target() const
{
  if (!has_pose_goal_) {return {};}
  const auto & p = pose_goal_.pose;
  return {p.position.x, p.position.y, p.position.z,
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w};
}
}  // namespace task_priority_kinematic_control
