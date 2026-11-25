#include "task_priority_kinematic_control/tasks/pose_task_common.hpp"
#include <cmath>
#include <stdexcept>

namespace task_priority_kinematic_control
{
namespace pose_math
{
Eigen::Vector3d quaternion_error(const Eigen::Quaterniond & target, const Eigen::Quaterniond & current)
{
  const Eigen::Quaterniond q_err = target * current.conjugate();
  Eigen::AngleAxisd aa(q_err);
  return aa.axis() * aa.angle();
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

Eigen::MatrixXd base_jacobian_for_model(const WholeBodyModel & model)
{
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, model.total_dofs());
  for (size_t i = 0; i < model.active_base_dofs().size(); ++i) {
    const int dof = model.active_base_dofs()[i];
    if (dof >= 0 && dof < 6) {
      jacobian.col(static_cast<Eigen::Index>(i))(dof) = 1.0;
    }
  }
  return jacobian;
}

void apply_tool_offset(FrameState & frame, const Eigen::Isometry3d & offset)
{
  const Eigen::Vector3d p = frame.pose.rotation() * offset.translation();
  Eigen::Matrix3d skew;
  skew << 0.0, -p.z(), p.y(),
    p.z(), 0.0, -p.x(),
    -p.y(), p.x(), 0.0;
  if (frame.jacobian.rows() >= 6) {
    frame.jacobian.topRows(3) -= skew * frame.jacobian.bottomRows(3);
  }
  frame.pose = frame.pose * offset;
}

FrameState compute_tip_state(const KinematicsBackend & backend, const WholeBodyState & state,
  const std::string & frame, const TaskBase * task)
{
  FrameState result = backend.compute_frame_state(state, frame);
  if (const auto * pose_task = dynamic_cast<const PoseTaskCommon *>(task)) {
    if (const auto offset = pose_task->active_offset()) {apply_tool_offset(result, *offset);}
  }
  return result;
}
}  // namespace pose_math

PoseTaskCommon::PoseTaskCommon(const PoseTaskCommon & other)
: TaskBaseCommon(other)
{
  {
    std::scoped_lock offset_lock(other.offset_mutex_);
    offset_supported_ = other.offset_supported_;
    use_offset_ = other.use_offset_;
    offset_ = other.offset_;
    offset_frame_ = other.offset_frame_;
    // Copies run offline (STPIK probes, with their own clock and for far longer than offset_timeout):
    // the offset is frozen as it was at copy time, valid only if it was fresh then.
    offset_valid_ = other.offset_fresh_locked();
    offset_time_ = other.offset_time_;
    offset_timeout_ = 0.0;
    force_capture_target_ = other.force_capture_target_;
  }
  std::scoped_lock lock(other.feedforward_mutex_);
  feedforward_ = other.feedforward_;
  feedforward_epoch_ = other.feedforward_epoch_;
  accept_feedforward_ = other.accept_feedforward_;
  auto_update_target_pose_ = other.auto_update_target_pose_;
  feedforward_velocity_threshold_ = other.feedforward_velocity_threshold_;
  initialize_target_ = other.initialize_target_;
  feedforward_timeout_ = other.feedforward_timeout_;
  feedforward_time_ = other.feedforward_time_;
}

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

void PoseTaskCommon::configure_offset(
  const std::map<std::string, rclcpp::Parameter> & parameters, const std::string & frame_id,
  bool supported)
{
  const bool use = get_bool_param(parameters, "use_offset", false);
  const double timeout = get_double_param(parameters, "offset_timeout", 0.5);
  if (!std::isfinite(timeout) || timeout < 0.0) {
    throw std::invalid_argument("offset_timeout must be finite and non-negative");
  }
  if (use && !supported) {
    throw std::invalid_argument("use_offset is not supported for the base frame");
  }
  std::scoped_lock lock(offset_mutex_);
  offset_supported_ = supported;
  offset_frame_ = frame_id;
  offset_timeout_ = timeout;
  offset_valid_ = false;
  // Enabled from the YAML: the task waits (inactive) until an offset arrives; it never falls back to
  // the plain frame.
  use_offset_ = use;
  force_capture_target_ = false;
}

bool PoseTaskCommon::offset_fresh_locked() const
{
  return offset_valid_ && (offset_timeout_ <= 0.0 ||
    std::chrono::duration<double>(steady_now() - offset_time_).count() <= offset_timeout_);
}

bool PoseTaskCommon::use_offset() const
{
  std::scoped_lock lock(offset_mutex_);
  return use_offset_;
}

bool PoseTaskCommon::offset_ready() const
{
  std::scoped_lock lock(offset_mutex_);
  return offset_fresh_locked();
}

std::string PoseTaskCommon::offset_frame() const
{
  std::scoped_lock lock(offset_mutex_);
  return offset_frame_;
}

bool PoseTaskCommon::set_offset(const Eigen::Isometry3d & offset)
{
  if (!offset.matrix().allFinite()) {return false;}
  std::scoped_lock lock(offset_mutex_);
  offset_ = offset;
  offset_time_ = steady_now();
  offset_valid_ = true;
  return true;
}

bool PoseTaskCommon::set_use_offset(bool use, std::string & message)
{
  {
    std::scoped_lock lock(offset_mutex_);
    if (!offset_supported_) {
      message = "This task cannot control an offset point";
      return false;
    }
    if (use == use_offset_) {
      message = "Offset use unchanged";
      return true;
    }
    if (use && !offset_fresh_locked()) {
      message = "No fresh offset received; cannot enable use_offset";
      return false;
    }
    use_offset_ = use;
    force_capture_target_ = true;
  }
  clear_feedforward();
  message = use ? "Controlling the offset point" : "Controlling the task frame";
  return true;
}

std::optional<Eigen::Isometry3d> PoseTaskCommon::active_offset() const
{
  std::scoped_lock lock(offset_mutex_);
  if (!use_offset_ || !offset_fresh_locked()) {return std::nullopt;}
  return offset_;
}

bool PoseTaskCommon::apply_controlled_offset(FrameState & frame, std::string & message) const
{
  Eigen::Isometry3d offset;
  {
    std::scoped_lock lock(offset_mutex_);
    if (!use_offset_) {return true;}
    if (!offset_fresh_locked()) {
      message = offset_valid_ ? "offset stale" : "waiting for offset";
      return false;
    }
    offset = offset_;
  }
  pose_math::apply_tool_offset(frame, offset);
  return true;
}

bool PoseTaskCommon::set_feedforward_option(
  const std::string & field, const rclcpp::Parameter & parameter, std::string & message)
{
  if (field == "use_offset" && parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL) {
    return set_use_offset(parameter.as_bool(), message);
  }
  if (field == "offset_timeout" && parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
    const double timeout = parameter.as_double();
    if (std::isfinite(timeout) && timeout >= 0.0) {
      std::scoped_lock lock(offset_mutex_);
      offset_timeout_ = timeout;
      message = "Offset timeout changed";
      return true;
    }
  }
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
  message = "Invalid pose option: expected bool or finite non-negative threshold/timeout";
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
  feedforward_time_ = steady_now();
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
  bool force_capture = false;
  {
    std::scoped_lock lock(offset_mutex_);
    force_capture = force_capture_target_;
    force_capture_target_ = false;
  }
  if ((auto_update_target_pose_ || force_capture) &&
    (initialize_target_ || force_capture ||
    computation.feedforward.norm() > feedforward_velocity_threshold_))
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
      steady_now() - feedforward_time_).count() > feedforward_timeout_) {
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
