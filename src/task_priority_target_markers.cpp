#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "interactive_markers/interactive_marker_server.hpp"
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/interactive_marker.hpp"
#include "visualization_msgs/msg/interactive_marker_control.hpp"
#include "visualization_msgs/msg/interactive_marker_feedback.hpp"
#include "visualization_msgs/msg/marker.hpp"

namespace task_priority_kinematic_control
{
namespace
{

using PoseStamped = geometry_msgs::msg::PoseStamped;
using InteractiveMarker = visualization_msgs::msg::InteractiveMarker;
using InteractiveMarkerControl = visualization_msgs::msg::InteractiveMarkerControl;
using InteractiveMarkerFeedback = visualization_msgs::msg::InteractiveMarkerFeedback;
using Marker = visualization_msgs::msg::Marker;

struct Color
{
  float r;
  float g;
  float b;
  float a;
};

std::string strip_trailing_slash(std::string value)
{
  while (value.size() > 1 && value.back() == '/') {
    value.pop_back();
  }
  return value;
}

bool is_finite(const geometry_msgs::msg::Pose & pose)
{
  return
    std::isfinite(pose.position.x) &&
    std::isfinite(pose.position.y) &&
    std::isfinite(pose.position.z) &&
    std::isfinite(pose.orientation.x) &&
    std::isfinite(pose.orientation.y) &&
    std::isfinite(pose.orientation.z) &&
    std::isfinite(pose.orientation.w);
}

void normalize_quaternion(geometry_msgs::msg::Pose & pose)
{
  const double norm = std::sqrt(
    pose.orientation.x * pose.orientation.x +
    pose.orientation.y * pose.orientation.y +
    pose.orientation.z * pose.orientation.z +
    pose.orientation.w * pose.orientation.w);
  if (norm < 1e-9 || !std::isfinite(norm)) {
    pose.orientation.x = 0.0;
    pose.orientation.y = 0.0;
    pose.orientation.z = 0.0;
    pose.orientation.w = 1.0;
    return;
  }
  pose.orientation.x /= norm;
  pose.orientation.y /= norm;
  pose.orientation.z /= norm;
  pose.orientation.w /= norm;
}

bool poses_near(const geometry_msgs::msg::Pose & a, const geometry_msgs::msg::Pose & b)
{
  constexpr double kPositionEpsilon = 1e-6;
  constexpr double kOrientationEpsilon = 1e-6;
  return
    std::abs(a.position.x - b.position.x) < kPositionEpsilon &&
    std::abs(a.position.y - b.position.y) < kPositionEpsilon &&
    std::abs(a.position.z - b.position.z) < kPositionEpsilon &&
    std::abs(a.orientation.x - b.orientation.x) < kOrientationEpsilon &&
    std::abs(a.orientation.y - b.orientation.y) < kOrientationEpsilon &&
    std::abs(a.orientation.z - b.orientation.z) < kOrientationEpsilon &&
    std::abs(a.orientation.w - b.orientation.w) < kOrientationEpsilon;
}

Marker mesh_marker(
  const std::string & resource,
  const Color & color,
  double scale = 1.0,
  const geometry_msgs::msg::Pose & pose = geometry_msgs::msg::Pose())
{
  Marker marker;
  marker.type = Marker::MESH_RESOURCE;
  marker.mesh_resource = resource;
  marker.mesh_use_embedded_materials = false;
  marker.pose = pose;
  normalize_quaternion(marker.pose);
  marker.scale.x = scale;
  marker.scale.y = scale;
  marker.scale.z = scale;
  marker.color.r = color.r;
  marker.color.g = color.g;
  marker.color.b = color.b;
  marker.color.a = color.a;
  return marker;
}

Marker sphere_marker(const Color & color, double scale)
{
  Marker marker;
  marker.type = Marker::SPHERE;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = scale;
  marker.scale.y = scale;
  marker.scale.z = scale;
  marker.color.r = color.r;
  marker.color.g = color.g;
  marker.color.b = color.b;
  marker.color.a = color.a;
  return marker;
}

geometry_msgs::msg::Quaternion quaternion_from_rpy(
  const double roll,
  const double pitch,
  const double yaw)
{
  const double cr = std::cos(roll * 0.5);
  const double sr = std::sin(roll * 0.5);
  const double cp = std::cos(pitch * 0.5);
  const double sp = std::sin(pitch * 0.5);
  const double cy = std::cos(yaw * 0.5);
  const double sy = std::sin(yaw * 0.5);

  geometry_msgs::msg::Quaternion quaternion;
  quaternion.w = cr * cp * cy + sr * sp * sy;
  quaternion.x = sr * cp * cy - cr * sp * sy;
  quaternion.y = cr * sp * cy + sr * cp * sy;
  quaternion.z = cr * cp * sy - sr * sp * cy;
  return quaternion;
}

InteractiveMarkerControl visual_control(std::vector<Marker> markers)
{
  InteractiveMarkerControl control;
  control.always_visible = true;
  control.markers = std::move(markers);
  return control;
}

InteractiveMarkerControl axis_control(
  const std::string & name,
  uint8_t interaction_mode,
  double x,
  double y,
  double z)
{
  InteractiveMarkerControl control;
  control.name = name;
  control.orientation.w = 1.0;
  control.orientation.x = x;
  control.orientation.y = y;
  control.orientation.z = z;
  control.orientation_mode = InteractiveMarkerControl::FIXED;
  control.interaction_mode = interaction_mode;
  return control;
}

void add_6dof_controls(InteractiveMarker & marker)
{
  marker.controls.push_back(
    axis_control("move_x", InteractiveMarkerControl::MOVE_AXIS, 1.0, 0.0, 0.0));
  marker.controls.push_back(
    axis_control("move_y", InteractiveMarkerControl::MOVE_AXIS, 0.0, 0.0, 1.0));
  marker.controls.push_back(
    axis_control("move_z", InteractiveMarkerControl::MOVE_AXIS, 0.0, 1.0, 0.0));
  marker.controls.push_back(
    axis_control("rotate_roll", InteractiveMarkerControl::ROTATE_AXIS, 1.0, 0.0, 0.0));
  marker.controls.push_back(
    axis_control("rotate_pitch", InteractiveMarkerControl::ROTATE_AXIS, 0.0, 0.0, 1.0));
  marker.controls.push_back(
    axis_control("rotate_yaw", InteractiveMarkerControl::ROTATE_AXIS, 0.0, 1.0, 0.0));
}

rclcpp::QoS target_publisher_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
}

}  // namespace

class TaskPriorityTargetMarkers : public rclcpp::Node
{
public:
  TaskPriorityTargetMarkers()
  : Node("task_priority_target_markers")
  {
    world_frame_ = declare_parameter<std::string>("world_frame", "world_ned");
    task_topic_prefix_ = strip_trailing_slash(
      declare_parameter<std::string>(
        "task_topic_prefix", "/cirtesub/controller/task_priority/tasks"));
    interactive_marker_namespace_ = strip_trailing_slash(
      declare_parameter<std::string>(
        "interactive_marker_namespace", ""));
    interactive_marker_suffix_ = strip_trailing_slash(
      declare_parameter<std::string>("interactive_marker_suffix", "interactive_marker"));
    target_task_ids_ = declare_parameter<std::vector<std::string>>(
      "target_task_ids", {"base_pose", "left_pose", "right_pose"});
    base_task_id_ = declare_parameter<std::string>("base_task_id", "base_pose");
    left_task_id_ = declare_parameter<std::string>("left_task_id", "left_pose");
    right_task_id_ = declare_parameter<std::string>("right_task_id", "right_pose");
    marker_scale_ = declare_parameter<double>("marker_scale", 0.45);
    base_marker_scale_ = declare_parameter<double>("base_marker_scale", 0.9);
    end_effector_marker_scale_ = declare_parameter<double>("end_effector_marker_scale", 0.35);
    base_visual_scale_ = declare_parameter<double>("base_visual_scale", 1.0);
    end_effector_visual_scale_ = declare_parameter<double>("end_effector_visual_scale", 1.0);
    fallback_sphere_scale_ = declare_parameter<double>("fallback_sphere_scale", 0.18);

    configure_tasks();
    RCLCPP_INFO(
      get_logger(),
      "Task priority target markers ready for %zu task namespaces",
      tasks_.size());
  }

private:
  struct TaskMarker
  {
    std::string task_id;
    std::string marker_name;
    std::string frame_id;
    std::string interactive_marker_namespace;
    PoseStamped last_target;
    bool has_target = false;
    rclcpp::Time last_target_stamp{0, 0, RCL_ROS_TIME};
    std::unique_ptr<interactive_markers::InteractiveMarkerServer> server;
    rclcpp::Publisher<PoseStamped>::SharedPtr publisher;
    rclcpp::Subscription<PoseStamped>::SharedPtr subscription;
  };

  void configure_tasks()
  {
    for (const auto & task_id : target_task_ids_) {
      if (task_id.empty()) {
        continue;
      }
      TaskMarker task;
      task.task_id = task_id;
      task.marker_name = task_id + "_target";
      task.frame_id = world_frame_;
      task.interactive_marker_namespace = interactive_marker_namespace_for_task(task_id);
      task.server = std::make_unique<interactive_markers::InteractiveMarkerServer>(
        task.interactive_marker_namespace,
        get_node_base_interface(),
        get_node_clock_interface(),
        get_node_logging_interface(),
        get_node_topics_interface(),
        get_node_services_interface());
      const std::string target_topic = task_topic_prefix_ + "/" + task_id + "/target";
      task.publisher = create_publisher<PoseStamped>(target_topic, target_publisher_qos());

      const auto inserted = tasks_.emplace(task_id, std::move(task));
      auto & stored_task = inserted.first->second;
      marker_to_task_[stored_task.marker_name] = task_id;
      stored_task.subscription = create_subscription<PoseStamped>(
        target_topic,
        rclcpp::SystemDefaultsQoS(),
        [this, task_id](const PoseStamped::SharedPtr msg)
        {
          handle_target(task_id, msg);
        });

      RCLCPP_INFO(
        get_logger(),
        "Watching task target '%s' on '%s'; marker namespace '%s'",
        task_id.c_str(),
        target_topic.c_str(),
        stored_task.interactive_marker_namespace.c_str());
    }
  }

  std::string interactive_marker_namespace_for_task(const std::string & task_id) const
  {
    if (interactive_marker_namespace_.empty()) {
      return task_topic_prefix_ + "/" + task_id + "/" + interactive_marker_suffix_;
    }

    constexpr char kTaskIdToken[] = "{task_id}";
    const auto token_position = interactive_marker_namespace_.find(kTaskIdToken);
    if (token_position != std::string::npos) {
      std::string resolved = interactive_marker_namespace_;
      resolved.replace(token_position, std::string(kTaskIdToken).size(), task_id);
      return strip_trailing_slash(resolved);
    }

    if (target_task_ids_.size() == 1) {
      return interactive_marker_namespace_;
    }

    return interactive_marker_namespace_ + "/" + task_id;
  }

  void handle_target(const std::string & task_id, const PoseStamped::SharedPtr msg)
  {
    if (!msg || !is_finite(msg->pose)) {
      return;
    }

    PoseStamped normalized = *msg;
    normalize_quaternion(normalized.pose);
    if (normalized.header.frame_id.empty()) {
      normalized.header.frame_id = world_frame_;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto task_it = tasks_.find(task_id);
    if (task_it == tasks_.end()) {
      return;
    }

    auto & task = task_it->second;
    const rclcpp::Time target_stamp(normalized.header.stamp);
    if (task.has_target && target_stamp < task.last_target_stamp) {
      RCLCPP_DEBUG(
        get_logger(),
        "Ignoring stale target for task '%s'",
        task_id.c_str());
      return;
    }

    const bool unchanged =
      task.has_target &&
      task.frame_id == normalized.header.frame_id &&
      poses_near(task.last_target.pose, normalized.pose);
    task.last_target = normalized;
    task.last_target_stamp = target_stamp;
    task.frame_id = normalized.header.frame_id;
    task.has_target = true;

    if (!unchanged) {
      rebuild_marker_locked(task);
    }
  }

  void handle_feedback(const InteractiveMarkerFeedback::ConstSharedPtr & feedback)
  {
    if (!feedback || feedback->event_type != InteractiveMarkerFeedback::POSE_UPDATE) {
      return;
    }
    if (!is_finite(feedback->pose)) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const auto marker_it = marker_to_task_.find(feedback->marker_name);
    if (marker_it == marker_to_task_.end()) {
      return;
    }
    auto task_it = tasks_.find(marker_it->second);
    if (task_it == tasks_.end()) {
      return;
    }

    auto & task = task_it->second;
    PoseStamped target;
    target.header.stamp = now();
    target.header.frame_id = task.frame_id.empty() ? world_frame_ : task.frame_id;
    target.pose = feedback->pose;
    normalize_quaternion(target.pose);

    task.last_target = target;
    task.last_target_stamp = rclcpp::Time(target.header.stamp);
    task.has_target = true;
    if (task.server) {
      if (!task.server->setPose(task.marker_name, target.pose, target.header)) {
        RCLCPP_DEBUG(
          get_logger(),
          "Could not update interactive marker pose for task '%s'",
          task.task_id.c_str());
      }
      task.server->applyChanges();
    }
    task.publisher->publish(target);
  }

  void rebuild_marker_locked(TaskMarker & task)
  {
    if (!task.server || !task.has_target) {
      return;
    }

    task.server->clear();
    InteractiveMarker marker = build_marker(task);
    task.server->insert(
      marker,
      std::bind(&TaskPriorityTargetMarkers::handle_feedback, this, std::placeholders::_1));
    task.server->applyChanges();
  }

  InteractiveMarker build_marker(const TaskMarker & task) const
  {
    InteractiveMarker marker;
    marker.header.frame_id = task.frame_id.empty() ? world_frame_ : task.frame_id;
    marker.header.stamp = now();
    marker.name = task.marker_name;
    marker.description = task.task_id + " target";
    marker.scale = scale_for_task(task.task_id);
    marker.pose = task.last_target.pose;
    normalize_quaternion(marker.pose);

    if (task.task_id == base_task_id_) {
      add_base_visual(marker);
    } else if (task.task_id == left_task_id_) {
      add_gripper_visual(marker, Color{0.0F, 0.9F, 1.0F, 0.55F});
    } else if (task.task_id == right_task_id_) {
      add_gripper_visual(marker, Color{0.2F, 1.0F, 0.35F, 0.55F});
    } else {
      const Color color{1.0F, 1.0F, 1.0F, 0.65F};
      marker.controls.push_back(visual_control({
        sphere_marker(color, fallback_sphere_scale_)}));
    }

    add_6dof_controls(marker);
    return marker;
  }

  void add_base_visual(InteractiveMarker & marker) const
  {
    const Color body_color{1.0F, 0.72F, 0.22F, 0.45F};
    geometry_msgs::msg::Pose mesh_pose;
    mesh_pose.orientation = quaternion_from_rpy(0.0, -1.57, 0.0);
    marker.controls.push_back(visual_control({
      mesh_marker(
        "package://cirtesub_description/meshes/cirtesubdae.dae",
        body_color,
        base_visual_scale_,
        mesh_pose)}));
  }

  void add_gripper_visual(InteractiveMarker & marker, const Color & color) const
  {
    geometry_msgs::msg::Pose left_finger_pose;
    left_finger_pose.orientation.w = 1.0;
    left_finger_pose.position.y = 0.0155;
    left_finger_pose.position.z = 0.0069;

    geometry_msgs::msg::Pose right_finger_pose;
    right_finger_pose.orientation.w = 1.0;
    right_finger_pose.position.y = -0.0155;
    right_finger_pose.position.z = 0.0069;

    marker.controls.push_back(visual_control({
      mesh_marker(
        "package://alpha_description/meshes/end_effectors/RS1-124.stl",
        color,
        end_effector_visual_scale_),
      mesh_marker(
        "package://alpha_description/meshes/end_effectors/RS1-130.stl",
        color,
        end_effector_visual_scale_,
        left_finger_pose),
      mesh_marker(
        "package://alpha_description/meshes/end_effectors/RS1-139.stl",
        color,
        end_effector_visual_scale_,
        right_finger_pose)}));
  }

  double scale_for_task(const std::string & task_id) const
  {
    if (task_id == base_task_id_) {
      return base_marker_scale_;
    }
    if (task_id == left_task_id_ || task_id == right_task_id_) {
      return end_effector_marker_scale_;
    }
    return marker_scale_;
  }

  std::mutex mutex_;
  std::map<std::string, TaskMarker> tasks_;
  std::map<std::string, std::string> marker_to_task_;
  std::string world_frame_;
  std::string task_topic_prefix_;
  std::string interactive_marker_namespace_;
  std::string interactive_marker_suffix_;
  std::vector<std::string> target_task_ids_;
  std::string base_task_id_;
  std::string left_task_id_;
  std::string right_task_id_;
  double marker_scale_{0.45};
  double base_marker_scale_{0.9};
  double end_effector_marker_scale_{0.35};
  double base_visual_scale_{1.0};
  double end_effector_visual_scale_{1.0};
  double fallback_sphere_scale_{0.18};
};

}  // namespace task_priority_kinematic_control

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<task_priority_kinematic_control::TaskPriorityTargetMarkers>());
  rclcpp::shutdown();
  return 0;
}
