#pragma once
#include "task_priority_kinematic_control/tasks/self_collision_avoidance_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/stabilize_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/frame_pose_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/frame_position_norm_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/frame_orientation_norm_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/arm_manipulability_task.hpp"
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <thread>

namespace task_priority_kinematic_control
{
visualization_msgs::msg::MarkerArray capsule_markers(
  const std::vector<ManualCapsule> & definitions, const CollisionSnapshot & snapshot,
  const std::string & frame, double length_epsilon,
  const std::vector<ManualPlane> & planes = {});
visualization_msgs::msg::MarkerArray delete_capsule_markers(
  const std::vector<ManualCapsule> & definitions, const std::string & frame,
  const std::vector<ManualPlane> & planes = {});

visualization_msgs::msg::MarkerArray joint_limits_markers(
  const std::vector<std::string> & joint_names, const JointLimitsSnapshot & snapshot,
  const std::string & frame, const std::string & task_id);
visualization_msgs::msg::MarkerArray delete_joint_limits_markers(
  const std::vector<std::string> & joint_names, const std::string & frame, const std::string & task_id);

// FramePoseLimitsTask debug: allowed position box and orientation rings (world frame).
visualization_msgs::msg::MarkerArray frame_pose_limits_markers(
  const FramePoseLimitsSnapshot & snapshot, const std::string & frame,
  const std::string & task_id, double publish_rate);
visualization_msgs::msg::MarkerArray delete_frame_pose_limits_markers(
  const std::string & frame, const std::string & task_id);

// FramePositionNormLimitsTask debug: sphere / cylinder / slab region (world frame).
visualization_msgs::msg::MarkerArray frame_position_norm_markers(
  const FramePositionNormSnapshot & snapshot, const std::string & frame,
  const std::string & task_id, double publish_rate);
visualization_msgs::msg::MarkerArray delete_frame_position_norm_markers(
  const std::string & frame, const std::string & task_id);

// FrameOrientationNormLimitsTask debug: tilt cone / yaw ring and angle gauge (world frame).
visualization_msgs::msg::MarkerArray frame_orientation_norm_markers(
  const FrameOrientationNormSnapshot & snapshot, const std::string & frame,
  const std::string & task_id, double publish_rate);
visualization_msgs::msg::MarkerArray delete_frame_orientation_norm_markers(
  const std::string & frame, const std::string & task_id);

// ArmManipulabilityTask debug: manipulability ellipsoid at the tip coloured by state (world frame).
visualization_msgs::msg::MarkerArray arm_manipulability_markers(
  const ArmManipulabilitySnapshot & snapshot, const std::string & frame,
  const std::string & task_id, double publish_rate);
visualization_msgs::msg::MarkerArray delete_arm_manipulability_markers(
  const std::string & frame, const std::string & task_id);

// Allocate message storage once; updates reuse marker, string and point buffers.
class StabilizeLimitsMarkerRenderer
{
public:
  StabilizeLimitsMarkerRenderer(const std::string & frame, const std::string & task_id,
    double publish_rate = 2.0);
  const visualization_msgs::msg::MarkerArray & update(const StabilizeLimitsSnapshot & snapshot);
  const visualization_msgs::msg::MarkerArray & clear();
private:
  void prepare_geometry(const StabilizeLimitsSnapshot & snapshot);
  visualization_msgs::msg::MarkerArray markers_;
  Eigen::Vector2d lower_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d upper_ = Eigen::Vector2d::Zero();
  double alpha_ = 0;
  double delta_ = 0;
  bool geometry_ready_ = false;
};

// One dedicated non-control executor/timer for collision and limit tasks in a host.
class CapsuleObserver
{
public:
  CapsuleObserver(const std::vector<std::shared_ptr<TaskBase>> & tasks,
    const std::string & host_name, const rclcpp::Context::SharedPtr & context);
  ~CapsuleObserver();
  CapsuleObserver(const CapsuleObserver &) = delete;
  CapsuleObserver & operator=(const CapsuleObserver &) = delete;
private:
  struct Entry
  {
    std::shared_ptr<SelfCollisionAvoidanceTask> task;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher;
    const CollisionSnapshot * snapshot = nullptr;
    bool visible = false;
    bool previously_stopped = false;
    std::chrono::steady_clock::time_point last_publish{};
    std::chrono::steady_clock::time_point last_warning{};
  };
  struct LimitsEntry
  {
    std::shared_ptr<JointLimitsTask> task;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher;
    const JointLimitsSnapshot * snapshot = nullptr;
    bool visible = false;
    std::chrono::steady_clock::time_point last_publish{};
  };
  struct TiltLimitsEntry
  {
    std::shared_ptr<StabilizeLimitsTask> task;
    std::unique_ptr<StabilizeLimitsMarkerRenderer> renderer;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher;
    const StabilizeLimitsSnapshot * snapshot = nullptr;
    bool visible = false;
    std::chrono::steady_clock::time_point next_publish{};
    std::chrono::steady_clock::duration publish_period{};
  };
  struct PoseLimitsEntry
  {
    std::shared_ptr<FramePoseLimitsTask> task;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher;
    const FramePoseLimitsSnapshot * snapshot = nullptr;
    bool visible = false;
    std::chrono::steady_clock::time_point next_publish{};
    std::chrono::steady_clock::duration publish_period{};
  };
  // Generic debug-marker entry for tasks with a snapshot channel and debug flag.
  template<class TaskT, class SnapshotT>
  struct DebugMarkerEntry
  {
    std::shared_ptr<TaskT> task;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher;
    const SnapshotT * snapshot = nullptr;
    bool visible = false;
    std::chrono::steady_clock::time_point next_publish{};
    std::chrono::steady_clock::duration publish_period{};
  };
  using PositionNormEntry = DebugMarkerEntry<FramePositionNormLimitsTask, FramePositionNormSnapshot>;
  using OrientationNormEntry =
    DebugMarkerEntry<FrameOrientationNormLimitsTask, FrameOrientationNormSnapshot>;
  using ManipulabilityEntry = DebugMarkerEntry<ArmManipulabilityTask, ArmManipulabilitySnapshot>;
  void tick();
  std::vector<Entry> entries_;
  std::vector<LimitsEntry> limits_entries_;
  std::vector<TiltLimitsEntry> tilt_limits_entries_;
  std::vector<PoseLimitsEntry> pose_limits_entries_;
  std::vector<PositionNormEntry> position_norm_entries_;
  std::vector<OrientationNormEntry> orientation_norm_entries_;
  std::vector<ManipulabilityEntry> manipulability_entries_;
  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::thread worker_;
};

// Shared declaration/validation helpers keep ros2_control and runtime parameter conventions equal.
void declare_collision_parameters(
  const rclcpp::node_interfaces::NodeParametersInterface::SharedPtr & parameters,
  const std::string & prefix);
void declare_joint_limits_marker_parameters(
  const rclcpp::node_interfaces::NodeParametersInterface::SharedPtr & parameters,
  const std::string & prefix);
void declare_stabilize_limits_debug_parameters(
  const rclcpp::node_interfaces::NodeParametersInterface::SharedPtr & parameters,
  const std::string & prefix);
// Rejects topology changes and applies visualization toggles atomically. The observer
// never takes the host task mutex; parameter callbacks retain their existing serialization.
rcl_interfaces::msg::SetParametersResult validate_collision_parameter_updates(
  const std::vector<rclcpp::Parameter> & parameters,
  const std::vector<std::shared_ptr<TaskBase>> & tasks,
  bool apply_visualization = true);
rcl_interfaces::msg::SetParametersResult validate_joint_limits_marker_parameter_updates(
  const std::vector<rclcpp::Parameter> & parameters,
  const std::vector<std::shared_ptr<TaskBase>> & tasks,
  bool apply_visualization = true);
}  // namespace task_priority_kinematic_control
