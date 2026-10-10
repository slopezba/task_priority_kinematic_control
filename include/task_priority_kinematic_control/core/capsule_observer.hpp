#pragma once
#include "task_priority_kinematic_control/tasks/self_collision_avoidance_task.hpp"
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <thread>

namespace task_priority_kinematic_control
{
visualization_msgs::msg::MarkerArray capsule_markers(
  const std::vector<ManualCapsule> & definitions, const CollisionSnapshot & snapshot,
  const std::string & frame, double length_epsilon);
visualization_msgs::msg::MarkerArray delete_capsule_markers(
  const std::vector<ManualCapsule> & definitions, const std::string & frame);

// One dedicated non-control executor/timer for all collision tasks in a host.
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
  void tick();
  std::vector<Entry> entries_;
  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::thread worker_;
};

// Shared declaration/validation helpers keep ros2_control and runtime parameter conventions equal.
void declare_collision_parameters(
  const rclcpp::node_interfaces::NodeParametersInterface::SharedPtr & parameters,
  const std::string & prefix);
// Rejects topology changes and applies visualization toggles atomically. The observer
// never takes the host task mutex; parameter callbacks retain their existing serialization.
rcl_interfaces::msg::SetParametersResult validate_collision_parameter_updates(
  const std::vector<rclcpp::Parameter> & parameters,
  const std::vector<std::shared_ptr<TaskBase>> & tasks,
  bool apply_visualization = true);
}  // namespace task_priority_kinematic_control
